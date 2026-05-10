from collections import Counter
from typing import Callable, Iterable, Optional
from pathlib import Path
import spacy
from spacy.cli.download import download
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS

CACHE_DIR = Path("./data")
CACHE_DIR.mkdir(exist_ok=True)


class Vocab:
    """Minimal vocabulary helper compatible with common torchtext-style calls"""

    def __init__(
        self, tokens: Iterable[str], specials: Optional[list[str]] = None
    ) -> None:
        specials = specials or SPECIAL_TOKENS
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        for token in list(specials) + list(tokens):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        self.default_index = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def lookup_token(self, index: int) -> str:
        return self.itos[index]

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self.lookup_token(int(index)) for index in indices]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> dict[str, int]:
        return self.stoi

    def get_itos(self) -> list[str]:
        return self.itos

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]


# loading spacy tokenizer (as per language)
def _load_spacy_tokenizer(language: str):
    """loading spacy tokenizer (as per languages: german(de) and english(en))"""
    model_name = {"de": "de_core_news_sm", "en": "en_core_web_sm"}[language]
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"[spacy] '{model_name}' not found — downloading...")
        download(model_name)
        return spacy.load(model_name)


def _tokenize(nlp, text: str) -> list[str]:
    """lowercase-tokenize a string using spaCy (nlp = tokenizer for a particular language)"""
    return [token.text.lower() for token in nlp.tokenizer(text)]


class Multi30kDataset:
    def __init__(self, split="train"):
        """Loads the Multi30k dataset and prepares tokenizers."""
        self.split = split
        # Load dataset from Hugging Face: https://huggingface.co/datasets/bentrevett/multi30k

        if (CACHE_DIR / "bentrevett___multi30k").exists():
            print(f"loading from cached dir: ./{CACHE_DIR}")
            self.dataset = load_dataset(path=str(CACHE_DIR / "bentrevett___multi30k"))
        else:
            print(f"downloading from HF hub, saving to {CACHE_DIR}")
            self.dataset = load_dataset("bentrevett/multi30k", cache_dir=str(CACHE_DIR))

        # load spacy tokenizers
        self.de_nlp = _load_spacy_tokenizer("de")
        self.en_nlp = _load_spacy_tokenizer("en")

        # filled by build_vocab()
        self.src_vocab: Optional[Vocab] = None
        self.tgt_vocab: Optional[Vocab] = None

        # filled by process_data()
        self.processed: dict[str, list[tuple[list[int], list[int]]]] = {}

    def build_vocab(self, min_freq: int = 2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including: `<unk>`, `<pad>`, `<sos>`, `<eos>`
        Args:
            min_freq: Minimum token frequency to be included in the vocab`
                      Rare tokens fall back to `<unk>` at numericalization time
        """
        de_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for row in self.dataset["train"]:
            de_counter.update(_tokenize(self.de_nlp, row["de"]))  # type: ignore
            en_counter.update(_tokenize(nlp=self.en_nlp, text=row["en"]))  # type: ignore

        # keep only tokens that appear at least min_freq times
        de_tokens = [tok for tok, cnt in de_counter.items() if cnt >= min_freq]
        en_tokens = [tok for tok, cnt in en_counter.items() if cnt >= min_freq]

        self.src_vocab = Vocab(de_tokens)
        self.tgt_vocab = Vocab(en_tokens)

        print(f"[vocab] DE vocab size : {len(self.src_vocab)}")
        print(f"[vocab] EN vocab size : {len(self.tgt_vocab)}")

    def _numericalize(
        self,
        tokens: list[str],
        vocab: Vocab,
    ) -> list[int]:
        """Convert a token list to integer ids, wrapped with `<sos>` and `<eos>`."""
        return [vocab.sos_idx] + [vocab[tok] for tok in tokens] + [vocab.eos_idx]

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        Result stored in self.processed:
            { "train": [(src_ids, tgt_ids), ...],
              "validation": [...],
              "test": [...] }
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() before process_data().")

        for split in ("train", "validation", "test"):
            pairs: list[tuple[list[int], list[int]]] = []
            for row in self.dataset[split]:
                src_tokens = _tokenize(self.de_nlp, row["de"])  # type: ignore
                tgt_tokens = _tokenize(self.en_nlp, row["en"])  # type: ignore

                src_ids = self._numericalize(src_tokens, self.src_vocab)
                tgt_ids = self._numericalize(tgt_tokens, self.tgt_vocab)

                pairs.append((src_ids, tgt_ids))

            self.processed[split] = pairs
            print(f"[data] {split:>10} — {len(pairs)} sentence pairs")

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 128,
        shuffle: bool = None, # type: ignore
    ) -> DataLoader:
        """
        Return a DataLoader for the requested split.
        Args:
            split      : "train", "validation", or "test".
            batch_size : Number of sentence pairs per batch.
            shuffle    : Defaults to True for train, False otherwise.
        """
        if split not in self.processed:
            raise RuntimeError(
                f"Split '{split}' not found. Please call .process_data()"
            )
        if shuffle is None:
            shuffle = split == "train"

        torch_dataset = _TranslationDataset(self.processed[split])
        pad_idx = self.src_vocab.pad_idx  # same index in both vocabs (always 1) # type: ignore

        return DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_make_collate_fn(pad_idx),
        )


class _TranslationDataset(Dataset):
    """Thin wrapper so the list of (src_ids, tgt_ids) pairs works with `DataLoader` class"""

    def __init__(self, pairs: list[tuple[list[int], list[int]]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def _make_collate_fn(pad_idx: int) -> Callable:
    """
    Returns a collate function that pads variable-length sequences (we provide `<pad>` token index).
    Each batch yields:
        src : [batch_size, max_src_len]  - padded source token ids
        tgt : [batch_size, max_tgt_len]  - padded target token ids
    """

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)

        # pad_sequence expects a list of 1-D tensors; pads along dim=0
        # batch_first=True -> output shape [B, max_len]
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx) # type: ignore
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx) # type: ignore

        return src_padded, tgt_padded

    return collate_fn


if __name__ == "__main__":
    m30k = Multi30kDataset()
    print("Dataset loaded.")

    m30k.build_vocab(min_freq=2)
    m30k.process_data()

    train_loader = m30k.get_dataloader("train", batch_size=128)
    val_loader = m30k.get_dataloader("validation", batch_size=128)
    test_loader = m30k.get_dataloader("test", batch_size=1)

    # check one batch
    src, tgt = next(iter(train_loader))
    print(f"\nBatch shapes  ->  src: {src.shape}  tgt: {tgt.shape}")
    print(f"src[0] decoded: {m30k.src_vocab.lookup_tokens(src[0].tolist())}") # type: ignore
    print(f"tgt[0] decoded: {m30k.tgt_vocab.lookup_tokens(tgt[0].tolist())}") # type: ignore
