"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   -> Tensor          │
  │  PositionalEncoding.forward(x)               -> Tensor          │
  │  make_src_mask(src, pad_idx)                 -> BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 -> BoolTensor      │
  │  Transformer.encode(src, src_mask)           -> Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  -> Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Multi30kDataset

multi30k = Multi30kDataset()
multi30k.build_vocab()
DE_NLP = multi30k.de_nlp
EN_NLP = multi30k.en_nlp
SRC_VOCAB = multi30k.src_vocab
TGT_VOCAB = multi30k.tgt_vocab


# STANDALONE ATTENTION FUNCTION:
# Exposed at module level so the autograder can import and test it independently of MultiHeadAttention
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
            (..., seq_q, seq_k).
            Positions where mask is True are MASKED OUT
            (set to -inf before softmax).
    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.shape[-1]
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # apply mask if needed
    if mask is not None:
        raw_scores = raw_scores.masked_fill(mask, float("-inf"))
    attention_matrix = F.softmax(raw_scores, dim=-1)
    if dropout is not None:
        attention_matrix = dropout(attention_matrix)
    output = torch.matmul(attention_matrix, V)
    return (output, attention_matrix)


# MASK HELPERS:
# Exposed at module level so they can be tested independently and reused inside Transformer.forward
def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).  
    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)
    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  -> position is a PAD token (will be masked out)
        False -> real token
    """
    # (src == pad_idx) is of shape [batch, src_len]
    # unsqueeze twice -> [batch, 1, 1, src_len]  (so we can broadcast over heads and query dim)
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.  
    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)
    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True -> position is masked out (PAD or future token)
    """
    tgt_len = tgt.shape[-1]
    # padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # True wherever <pad>
    
    # causal mask (look-ahead): upper triangle (excluding diagonal) -> [1, 1, tgt_len, tgt_len]
    # torch.triu with diagonal=1 gives True for positions that should be masked
    causal_mask = (
        torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,  # one level above diagonal we will mask
        ).unsqueeze(0).unsqueeze(0))  # [1, 1, tgt_len, tgt_len]

    # combine both: position is masked if it's PAD or it's a future token
    return pad_mask | causal_mask  # [batch, 1, tgt_len, tgt_len]


#  MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.
        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # depth per head

        self.dropout = nn.Dropout(p=dropout)

        # rather than doing a split and project, we do a project and split for efficiency
        self.W_Q = nn.Linear(d_model, d_model) # projects to num_heads × d_k (= d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model) # output projection

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True -> masked out (attend nowhere)
        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]
        seq_v = value.shape[1]  # same as seq_k

        # project
        query = self.W_Q(query)
        key = self.W_K(key)
        value = self.W_V(value)

        # now split into heads (batch, seq, d_model) -> (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
        query = query.reshape(batch_size, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, seq_v, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot product attention
        value_updated, _ = scaled_dot_product_attention(query, key, value, mask, self.dropout)  # (batch, num_heads, seq_q, d_k)

        # (batch, num_heads, seq, d_k) -> (batch, seq, num_heads, d_k) -> (batch, seq_q, num_heads*d_k) -> (batch, seq_q, d_model)
        value_updated = value_updated.transpose(1, 2)
        out = value_updated.reshape(batch_size, seq_q, self.d_model)

        # output projection
        out = self.W_O(out)

        return out


#   POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.
    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # build PE matrix: [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Compute the division term: 10000^(2i/d_model)
        # Using log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model], add batch dimension for broadcasting

        # register as buffer: moves with .to(device) but not a trainable parameter
        self.register_buffer("pe", pe)
        self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]. 
        Returns:
            Tensor of same shape [batch, seq_len, d_model]   
            x := x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


#  FEED-FORWARD NETWORK
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:
        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.pointwise_ffn1 = nn.Linear(d_model, d_ff)
        self.pointwise_ffn2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.pointwise_ffn2(self.dropout(F.relu(self.pointwise_ffn1(x))))


#  ENCODER LAYER
class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x -> [Self-Attention -> Add & Norm] -> [FFN -> Add & Norm]
    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # we use post-LN as per the paper
        # sub-layer 1: self-attention + residual + norm
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask)))
        # sub-layer 2: FFN + residual + norm
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


#   DECODER LAYER
class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x -> [Masked Self-Attn -> Add & Norm]
          -> [Cross-Attn(memory) -> Add & Norm]
          -> [FFN -> Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # masked MHA
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # cross MHA
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        # sub-layer 1: masked self-attention (query/key/value all come from x)
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        # sub-layer 2: cross-attention (query from x, key/value from encoder memory)
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, src_mask)))
        # sub-layer 3: FFN
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


#  ENCODER & DECODER STACKS
class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # deep-copy so each layer has independent weights
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


#   FULL TRANSFORMER
class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = len(SRC_VOCAB), # type: ignore
        tgt_vocab_size: int = len(TGT_VOCAB), # type: ignore
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = "best_checkpoint.pth",
    ) -> None:
        super().__init__()
        # save config for checkpoint reconstruction
        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self.model_state_dict = None
        # https://drive.google.com/file/d/1I82sTYwxdD6agqiEUHexkSleqkz4Dx4V/view?usp=sharing
        self.google_drive_id: str = "1I82sTYwxdD6agqiEUHexkSleqkz4Dx4V" 
        if checkpoint_path is not None:
            gdown.download(id=self.google_drive_id, output=checkpoint_path, quiet=False) # type: ignore
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            cfg = ckpt["model_config"]
            self.model_state_dict = ckpt["model_state_dict"]
            # for the cases when we train on config different than that mentioned in the paper
            # override constructor args with what the checkpoint was actually trained with
            src_vocab_size = cfg["src_vocab_size"]
            tgt_vocab_size = cfg["tgt_vocab_size"]
            d_model = cfg["d_model"]
            N = cfg["N"]
            num_heads = cfg["num_heads"]
            d_ff = cfg["d_ff"]
            dropout = cfg["dropout"]
        # embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        # positional encoding (shared for src and tgt)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        # encoder stack
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        # decoder stack
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)
        # final linear projection -> vocabulary logits
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        # load the weights
        if self.model_state_dict is not None:
            self.load_state_dict(self.model_state_dict)
        else:
            self._init_weights()  # as per the paper (Xavier initialization)

    def _init_weights(self) -> None:
        """Xavier uniform init for all linear and embedding parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # AUTOGRADER HOOKS ── keep these signatures exactly
    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.  
        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]
        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # scale embeddings by sqrt d_model
        x = self.src_embed(src) * math.sqrt(self.model_config["d_model"])
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.  
        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.model_config["d_model"])
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.  
        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, device="cpu", max_len: int = 100) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.  
        Args:
            src_sentence: The raw German text.
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
        with torch.no_grad():
            # Tokenize and numericalize
            tokens = [tok.text.lower() for tok in DE_NLP.tokenizer(src_sentence)]
            src_ids = (
                [SRC_VOCAB.sos_idx] # type: ignore
                + [SRC_VOCAB[tok] for tok in tokens] # type: ignore
                + [SRC_VOCAB.eos_idx] # type: ignore
            )
            src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
            src_mask = make_src_mask(src, pad_idx=SRC_VOCAB.pad_idx) # type: ignore

            # Greedy decode
            ys = greedy_decode(
                model=self,
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                start_symbol=TGT_VOCAB.sos_idx, # type: ignore
                end_symbol=TGT_VOCAB.eos_idx, # type: ignore
                device=device,
            )

        # convert indices back to tokens, strip specials
        SPECIALS = {"<sos>", "<eos>", "<pad>", "<unk>"}
        tokens_out = [
            TGT_VOCAB.lookup_token(idx) # type: ignore
            for idx in ys.squeeze(0).tolist()
            if TGT_VOCAB.lookup_token(idx) not in SPECIALS # type: ignore
        ]
        return " ".join(tokens_out)


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
        device       : 'cpu', 'mps' or 'cuda'.
    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        # encode source once — memory is reused for every decoding step
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


if __name__ == "__main__":
    # small config to test shapes end-to-end
    B, SRC_LEN, TGT_LEN = 2, 10, 8
    SRC_VOCAB, TGT_VOCAB = 100, 120
    D_MODEL, N, HEADS, D_FF = 32, 2, 4, 64

    model = Transformer(SRC_VOCAB, TGT_VOCAB, D_MODEL, N, HEADS, D_FF, dropout=0.0)
    model.eval()

    src = torch.randint(2, SRC_VOCAB, (B, SRC_LEN))
    tgt = torch.randint(2, TGT_VOCAB, (B, TGT_LEN))
    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    print(f"src_mask shape  : {src_mask.shape}")  # [2, 1, 1, 10]
    print(f"tgt_mask shape  : {tgt_mask.shape}")  # [2, 1, 8, 8]
    
    logits = model(src, tgt, src_mask, tgt_mask)
    print(f"logits shape    : {logits.shape}")  # [2, 8, 120]
    
    # test encode / decode separately
    memory = model.encode(src, src_mask)
    print(f"memory shape    : {memory.shape}")  # [2, 10, 32]
    logits2 = model.decode(memory, src_mask, tgt, tgt_mask)
    print(f"decode shape    : {logits2.shape}")  # [2, 8, 120]
    print("\nAll shapes correct!")
