"""
data.py — Data loading pipeline for the Text Style Transfer VAE.

Two dataset classes:
  1. StyleDataset     — For training (non-parallel). Combines modern + Shakespeare
                        sentences into a flat list with style labels. This is what
                        the professor's train() function expects.
  2. ParallelEvalDataset — For evaluation only. Keeps the sentence pairing so we
                           can compute BLEU between generated output and ground truth.

Tokenization strategy:
  - The .nltktok files contain word-tokenized text (spaces between all tokens,
    including punctuation: "I have a mind to strike thee ere thou speak'st .")
  - We rejoin tokens into strings with " ".join() and let LLaMA's BPE tokenizer
    handle everything. The custom vocab code from transformer_vae.ipynb is discarded.
  - LLaMA's tokenizer handles padding, BOS/EOS, and subword splitting natively.
  - The extra spaces around punctuation from nltk are consistent across both styles,
    so the model sees the same patterns on both sides. Post-processing at inference
    time strips these artifacts.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import DataConfig, ModelConfig


# ──────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────

def load_sentences(filepath: str | Path) -> list[str]:
    """
    Load a .nltktok file and rejoin tokens into plain strings.

    Each line is a space-separated list of nltk-tokenized words.
    We join them back into a string that the LLaMA tokenizer can process.

    Example line in file:  "I have a mind to strike thee ere thou speak'st ."
    Returned string:       "I have a mind to strike thee ere thou speak'st ."
    (Same — the join is a no-op since it's already space-separated.)
    """
    filepath = Path(filepath)
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


# ──────────────────────────────────────────────────────────
# Training dataset (non-parallel)
# ──────────────────────────────────────────────────────────

class StyleDataset(Dataset):
    """
    Flat dataset of sentences with style labels for non-parallel training.

    Combines all modern sentences (label=0) and all Shakespeare sentences
    (label=1) into one shuffled dataset. The model never sees which modern
    sentence corresponds to which Shakespeare sentence — it just learns
    that some sentences are modern and some are Shakespeare.

    Each item returns a dict with:
        - "text": raw string (will be tokenized in collate_fn)
        - "doc_id": style label (0=modern, 1=Shakespeare)

    This matches the batch format expected by the train() function in
    transformer_vae_original.ipynb: batch["text"] and batch["doc_id"].
    """

    def __init__(self, modern_sents: list[str], original_sents: list[str],
                 data_config: DataConfig):
        self.texts: list[str] = []
        self.labels: list[int] = []

        for sent in modern_sents:
            self.texts.append(sent)
            self.labels.append(data_config.modern_label)

        for sent in original_sents:
            self.texts.append(sent)
            self.labels.append(data_config.shakespeare_label)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.texts[idx],
            "doc_id": self.labels[idx],
        }


# ──────────────────────────────────────────────────────────
# Evaluation dataset (parallel)
# ──────────────────────────────────────────────────────────

class ParallelEvalDataset(Dataset):
    """
    Paired dataset for evaluation only.

    Keeps the 1:1 correspondence between modern and Shakespeare sentences
    so we can:
      1. Encode the modern sentence → extract z_c
      2. Combine with average Shakespeare z_s → decode
      3. Compare generated output to ground-truth Shakespeare (BLEU)

    NOT used during training.
    """

    def __init__(self, modern_sents: list[str], original_sents: list[str]):
        assert len(modern_sents) == len(original_sents), (
            f"Parallel data must be aligned: got {len(modern_sents)} modern "
            f"vs {len(original_sents)} original"
        )
        self.modern = modern_sents
        self.original = original_sents

    def __len__(self) -> int:
        return len(self.modern)

    def __getitem__(self, idx: int) -> dict:
        return {
            "modern": self.modern[idx],
            "original": self.original[idx],
        }


# ──────────────────────────────────────────────────────────
# Collate functions
# ──────────────────────────────────────────────────────────

def make_train_collate_fn(tokenizer: AutoTokenizer, max_seq_len: int):
    """
    Returns a collate function that tokenizes raw strings on the fly.

    Why tokenize in collate_fn rather than in __getitem__?
    - Dynamic padding: each batch is padded to its own max length, not the
      global max_seq_len. This saves compute on short batches.
    - No need to store 37K pre-tokenized tensors in memory.
    - The tokenizer call is batched (faster than per-sample).

    The returned batch dict contains:
        - "input_ids":      (batch, seq_len) — token IDs for the model
        - "attention_mask": (batch, seq_len) — 1 for real tokens, 0 for padding
        - "doc_ids":        (batch,)         — style labels
    """

    def collate_fn(samples: list[dict]) -> dict:
        texts = [s["text"] for s in samples]
        doc_ids = torch.tensor([s["doc_id"] for s in samples], dtype=torch.long)

        # Tokenize the whole batch at once
        encoded = tokenizer(
            texts,
            padding=True,               # Pad to longest in batch
            truncation=True,             # Truncate to max_seq_len
            max_length=max_seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],           # (batch, seq_len)
            "attention_mask": encoded["attention_mask"],  # (batch, seq_len)
            "doc_ids": doc_ids,                          # (batch,)
        }

    return collate_fn


def make_eval_collate_fn(tokenizer: AutoTokenizer, max_seq_len: int):
    """
    Collate function for the parallel evaluation dataset.

    Tokenizes both modern and original sentences so we can:
    - Encode the modern sentence through the VAE
    - Have ground-truth original token IDs for BLEU comparison
    """

    def collate_fn(samples: list[dict]) -> dict:
        modern_texts = [s["modern"] for s in samples]
        original_texts = [s["original"] for s in samples]

        modern_enc = tokenizer(
            modern_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        original_enc = tokenizer(
            original_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

        return {
            "modern_input_ids": modern_enc["input_ids"],
            "modern_attention_mask": modern_enc["attention_mask"],
            "original_input_ids": original_enc["input_ids"],
            "original_attention_mask": original_enc["attention_mask"],
            "modern_texts": modern_texts,
            "original_texts": original_texts,
        }

    return collate_fn


# ──────────────────────────────────────────────────────────
# Convenience: build all dataloaders
# ──────────────────────────────────────────────────────────

def build_dataloaders(
    data_config: DataConfig,
    model_config: ModelConfig,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int = 2,
) -> dict:
    """
    Build train, validation, and test dataloaders.

    Returns a dict with keys: "train", "valid", "test_eval"
    - "train" and "valid" use StyleDataset (non-parallel, shuffled)
    - "test_eval" uses ParallelEvalDataset (parallel, for BLEU evaluation)
    """
    # Load raw sentences
    train_modern = load_sentences(data_config.get_path(data_config.train_modern))
    train_original = load_sentences(data_config.get_path(data_config.train_original))
    valid_modern = load_sentences(data_config.get_path(data_config.valid_modern))
    valid_original = load_sentences(data_config.get_path(data_config.valid_original))
    test_modern = load_sentences(data_config.get_path(data_config.test_modern))
    test_original = load_sentences(data_config.get_path(data_config.test_original))

    print(f"Loaded data — Train: {len(train_modern)} modern + {len(train_original)} shakespeare")
    print(f"              Valid: {len(valid_modern)} modern + {len(valid_original)} shakespeare")
    print(f"              Test:  {len(test_modern)} parallel pairs")

    # Training dataset: flat, non-parallel
    train_dataset = StyleDataset(train_modern, train_original, data_config)
    valid_dataset = StyleDataset(valid_modern, valid_original, data_config)

    # Test dataset: parallel for BLEU evaluation
    test_eval_dataset = ParallelEvalDataset(test_modern, test_original)

    # Collate functions
    train_collate = make_train_collate_fn(tokenizer, model_config.max_seq_len)
    eval_collate = make_eval_collate_fn(tokenizer, model_config.max_seq_len)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,                # Shuffle is critical — we want modern and
            num_workers=num_workers,      # Shakespeare sentences interleaved randomly
            collate_fn=train_collate,     # so each batch has a mix of both styles.
            pin_memory=True,
            drop_last=True,              # Drop last incomplete batch for stable
        ),                               # contrastive loss (needs consistent batch size)
        "valid": DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collate,
            pin_memory=True,
        ),
        "test_eval": DataLoader(
            test_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=eval_collate,
            pin_memory=True,
        ),
    }

    return loaders
