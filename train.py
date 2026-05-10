"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      -> torch.Tensor  shape [1, out_len]  (token indices)           │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      -> float  (corpus-level BLEU score, 0-100)                     │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) -> None  │
  │  load_checkpoint(path, model, optimizer, scheduler)        -> int   │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import time

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset


#  LABEL SMOOTHING LOSS
class LabelSmoothingLoss(nn.Module):
    """
        Label smoothing as in "Attention Is All You Need".  
        Instead of a hard one-hot target, mass is redistributed:  
        - true class : 1 - eps
        - all others : eps / (vocab_size - 2)   (excluding pad and true class)
        - `<pad>`    : 0.0                      (never rewarded)
        Args:
            vocab_size (int)  : Number of output classes.
            pad_idx    (int)  : Index of <pad> token — receives 0 probability.
            smoothing  (float): Smoothing factor epsilon (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing  # mass placed on the true class
        self.fill_val = smoothing / (vocab_size - 2)  # spread evenly over remaining tokens

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)
        Returns:
            Scalar mean loss, ignoring <pad> positions.
        """
        # build smoothed target distribution
        smooth_dist = torch.full_like(logits, self.fill_val)
        smooth_dist[:, self.pad_idx] = 0.0  # pad always gets 0
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # true class gets 1-eps
        # zero out rows where the gold token is <pad> — those don't contribute to loss
        pad_mask = target == self.pad_idx
        smooth_dist[pad_mask] = 0.0
        # KL divergence — equivalent to cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_dist * log_probs).sum(dim=-1)  # [N]
        # average over non-pad tokens only
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


#   TRAINING LOOP
def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Run one epoch of training or evaluation.
    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (pass None during eval).
        scheduler  : NoamScheduler instance (pass None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu', 'cuda' or 'mps'.
    Returns:
        avg_loss : Average per-token loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    t0 = time.time()
    with context:
        for src, tgt in data_iter:
            src = src.to(device)  # [B, src_len]
            tgt = tgt.to(device)  # [B, tgt_len]

            # teacher forcing: feed tgt[:-1] as input, predict tgt[1:] as output
            # this ensures that past token mistakes do not propagate forward
            tgt_in = tgt[:, :-1]  # <sos> w1 w2 ... wN
            tgt_out = tgt[:, 1:]  #       w1 w2 ... wN <eos>
            
            # build masks
            src_mask = make_src_mask(src, pad_idx=1).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=1).to(device)

            # forward pass -> [B, tgt_len-1, vocab_size]
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # flatten for loss computation
            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)  # [B*T, Vocab size]
            targets_flat = tgt_out.reshape(B * T)  # [B*T]
            loss = loss_fn(logits_flat, targets_flat)

            if is_train: # weight update
                optimizer.zero_grad()  # type: ignore
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # type: ignore
                if scheduler is not None:
                    scheduler.step()
                wandb.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],  # type: ignore
                    }
                )
            # accumulate: we count only non-pad tokens
            n_tokens = (tgt_out != 1).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    phase = "train" if is_train else "val"
    wandb.log(
        {
            f"{phase}/epoch_loss": avg_loss,
            f"{phase}/epoch_time_s": elapsed,
            "epoch": epoch_num,
        }
    )

    return avg_loss, elapsed


# GREEDY DECODING
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.  
    Encodes source once, then at each step decodes the full output
    sequence so far and picks the argmax at the last position only.  
    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.
    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # encode source once, memory is reused for every decoding step
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]
        # initialise output with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)  # [1, 1]
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            # decode current sequence -> [1, cur_len, vocab_size]
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            # argmax over vocab at the last position only
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys  # [1, out_len]


# BLEU EVALUATION
_SPECIALS = {"<sos>", "<eos>", "<pad>", "<unk>"}


def _ids_to_str(ids: list[int], tgt_vocab) -> str:
    """Convert token ids to a clean string, stripping special tokens."""
    tokens = [
        tgt_vocab.lookup_token(i)
        for i in ids
        if tgt_vocab.lookup_token(i) not in _SPECIALS
    ]
    return " ".join(tokens)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    Args:
        model           : Trained Transformer (eval mode).
        test_dataloader : DataLoader over the test split (src, tgt) pairs.
        tgt_vocab       : Vocab object with lookup_token(idx) method.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.
    Returns:
        bleu_score : Corpus-level BLEU (float, range 0-100).
    """
    model.eval()
    hypotheses = []
    list_of_references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0).to(device)
                src_mask = make_src_mask(src_i, pad_idx=1).to(device)

                ys = greedy_decode(
                    model=model,
                    src=src_i,
                    src_mask=src_mask,
                    max_len=max_len,
                    start_symbol=tgt_vocab.sos_idx,
                    end_symbol=tgt_vocab.eos_idx,
                    device=device,
                )

                hyp_str = _ids_to_str(ys.squeeze(0).tolist(), tgt_vocab)
                ref_str = _ids_to_str(tgt[i].tolist(), tgt_vocab)

                hypotheses.append(hyp_str.split())
                list_of_references.append([ref_str.split()])

    smoothing_function = SmoothingFunction().method1 # Add epsilon counts to precision with 0 counts
    bleu_score = corpus_bleu(
        list_of_references,
        hypotheses,
        smoothing_function=smoothing_function,
    )
    return bleu_score * 100 # type: ignore


# CHECKPOINT UTILITIES  (autograder loads your model from disk)
def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model, optimizer, scheduler state to disk.  
    Args:
        model     : Transformer instance.  
        optimizer : Optimizer instance.  
        scheduler : NoamScheduler instance.  
        epoch     : Current epoch number.  
        path      : File path to write (default 'checkpoint.pt').  
    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'  
    model_config must contain all kwargs needed to reconstruct Transformer(**model_config).
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.model_config,  # needed to reconstruct Transformer(**model_config)
        },
        path,
    )
    print(f"[checkpoint] epoch {epoch} saved -> {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).
    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"[checkpoint] epoch {ckpt['epoch']} loaded <- {path}")
    return ckpt["epoch"]

## experimental (below function was written for local testing purposes ONLY. I have used greedy decoding only in evaluate_bleau func)
def beam_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
    beam_size: int = 4,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]
        # each beam: (score, token_ids)
        beams = [(0.0, [start_symbol])]
        completed = []
        for _ in range(max_len - 1):
            candidates = []
            for score, seq in beams:
                if seq[-1] == end_symbol:
                    completed.append((score, seq))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
                logits = model.decode(memory, src_mask, ys, tgt_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # [1, vocab]
                topk_scores, topk_ids = log_probs[0].topk(beam_size)
                for s, idx in zip(topk_scores.tolist(), topk_ids.tolist()):
                    candidates.append((score + s, seq + [idx]))
            if not candidates:
                break
            # keep top beam_size, length-normalised
            candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = candidates[:beam_size]
            if all(seq[-1] == end_symbol for _, seq in beams):
                break
        completed += beams
        completed.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        best = completed[0][1]
    return torch.tensor([best], dtype=torch.long, device=device)

#   EXPERIMENT ENTRY POINT
def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val / test splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (beta1=0.9, beta2=0.98, eps=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(..., is_train=True)
                   run_epoch(..., is_train=False)
                   save_checkpoint(...)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test/bleu': bleu})
    """
    # hyperparameters
    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 128,
        "num_epochs": 30,
        "warmup_steps": 4000,
        "label_smooth": 0.1,
        "min_freq": 2,
    }

    # init W&B
    wandb.init(project="da6401_assignment3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    print(f"[device] {device}")

    # dataset + vocab
    print("[data] loading Multi30k ...")
    m30k = Multi30kDataset()
    m30k.build_vocab(min_freq=cfg.min_freq)
    m30k.process_data()

    src_vocab = m30k.src_vocab
    tgt_vocab = m30k.tgt_vocab
    wandb.config.update({"src_vocab_size": len(src_vocab), "tgt_vocab_size": len(tgt_vocab)})  # type: ignore

    # dataloaders
    train_loader = m30k.get_dataloader("train", batch_size=cfg.batch_size, shuffle=True)
    val_loader = m30k.get_dataloader("validation", batch_size=cfg.batch_size, shuffle=False)
    test_loader = m30k.get_dataloader("test", batch_size=cfg.batch_size, shuffle=False)

    # model
    model = Transformer(
        src_vocab_size=len(src_vocab),  # type: ignore
        tgt_vocab_size=len(tgt_vocab),  # type: ignore
        d_model=cfg.d_model, num_heads=cfg.num_heads,
        N=cfg.N, d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable parameters : {n_params:,}")

    # optimizer — betas and eps from Section 5.3 of the paper
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Noam scheduler scales this; base lr=1 is intentional
        betas=(0.9, 0.98), eps=1e-9,
    )

    # Noam LR scheduler
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # label smoothing loss
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),  # type: ignore
        pad_idx=tgt_vocab.pad_idx,  # type: ignore
        smoothing=cfg.label_smooth,)

    # training loop
    best_val_loss = float("inf")
    best_ckpt_path = "best_checkpoint.pt"

    for epoch in range(cfg.num_epochs):
        print(f"\n[epoch {epoch + 1}/{cfg.num_epochs}]")
        train_loss, train_elapsed = run_epoch(
            train_loader, model,
            loss_fn, optimizer, scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
        val_loss, val_elapsed = run_epoch(
            val_loader, model,
            loss_fn, None, None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )

        print(f"  train loss : {train_loss:.4f} (took {train_elapsed:.4f}s)  |  val loss : {val_loss:.4f} (took {val_elapsed:.4f}s)")
        # save latest checkpoint every epoch (useful for resuming)
        save_checkpoint(model, optimizer, scheduler, epoch, path="latest_checkpoint.pt")

        # save best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt_path)
            print(f"  * best val loss {best_val_loss:.4f} — checkpoint saved")

    # final BLEU on test set using best checkpoint
    print("\n[eval] loading best checkpoint ...")
    load_checkpoint(best_ckpt_path, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"[eval] test BLEU : {bleu:.2f}")
    wandb.log({"test/bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
