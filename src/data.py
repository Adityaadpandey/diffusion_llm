"""TinyStories dataset loading and tokenization with sequence packing.

Supports both BERT-style WordPiece (default) and GPT-2/LLaMA byte-level BPE
tokenizers. BPE tokenizes text into subword units that map better onto open-ended
generation, which usually produces noticeably better samples than WordPiece.
"""

from __future__ import annotations

from typing import Iterator

import torch
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataset(
    dataset_name: str = "roneneldan/TinyStories",
    tokenizer_name: str = "bert-base-uncased",
    max_seq_len: int = 128,
    val_split: float = 0.005,
    text_column: str = "text",
    cache_dir: str | None = None,
    num_proc: int = 4,
    streaming: bool = False,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset, AutoTokenizer]:
    """Load TinyStories, tokenize, pack into fixed-length chunks, split train/val.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer_name: HuggingFace tokenizer identifier. Use a BPE tokenizer such
            as "gpt2" for open-ended generation quality.
        max_seq_len: chunk length in tokens.
        val_split: fraction held out for validation.
        text_column: column containing raw text.
        cache_dir: optional HF cache directory.
        num_proc: number of processes for tokenization/packing.
    Returns:
        train_dataset, val_dataset, tokenizer
    """
    tokenizer = load_tokenizer(tokenizer_name)

    # BPE tokenizers are append-only when truncating from the right: the EOS
    # token, if present, is preserved. WordPiece tokenizers get a separator.
    append_eos = tokenizer.eos_token is not None and tokenizer.eos_token != ""
    sep = None if append_eos else tokenizer.sep_token

    if streaming:
        raw = load_dataset(dataset_name, split="train", streaming=True, cache_dir=cache_dir)
        if max_samples is not None:
            raw = raw.take(max_samples)

        def tokenize_and_pad(example):
            ids = tokenizer(
                example[text_column],
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
            )["input_ids"]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            ids = ids + [pad_id] * (max_seq_len - len(ids))
            return {"input_ids": ids[:max_seq_len]}

        packed = raw.map(tokenize_and_pad, remove_columns=raw.column_names)
        packed = packed.with_format("torch")
        # Materialize a list for splitting; only meant for small streaming tests.
        items = list(packed)
        packed = HFDataset.from_dict({"input_ids": [it["input_ids"] for it in items]})
        packed = packed.with_format("torch")
    else:
        raw = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))

        def tokenize_batch(batch):
            texts = batch[text_column]
            if sep is not None:
                texts = [text + sep for text in texts]
            return tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_len,
                return_attention_mask=False,
            )

        tokenized = raw.map(
            tokenize_batch,
            batched=True,
            remove_columns=raw.column_names,
            num_proc=num_proc,
            desc="Tokenizing",
        )

        def pack_chunks(examples):
            concatenated = []
            for ids in examples["input_ids"]:
                concatenated.extend(ids)

            total_length = (len(concatenated) // max_seq_len) * max_seq_len
            concatenated = concatenated[:total_length]

            return {
                "input_ids": [
                    concatenated[i : i + max_seq_len]
                    for i in range(0, total_length, max_seq_len)
                ]
            }

        packed = tokenized.map(
            pack_chunks,
            batched=True,
            batch_size=1000,
            remove_columns=tokenized.column_names,
            num_proc=1,
            desc="Packing",
        )
        packed = packed.with_format("torch")

    if val_split and val_split > 0:
        split = packed.train_test_split(test_size=val_split, shuffle=True, seed=42)
        return split["train"], split["test"], tokenizer
    return packed, None, tokenizer


class TextDataset(Dataset):
    """Simple wrapper around a HF dataset returning input_ids."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.dataset[idx]
        if isinstance(item, dict):
            ids = item["input_ids"]
        else:
            ids = item
        return torch.as_tensor(ids, dtype=torch.long)

    def __getitems__(self, indices: list[int]) -> list[torch.Tensor]:
        return [self.__getitem__(i) for i in indices]


def get_dataloaders(
    train_ds,
    val_ds,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader | None, DataLoader | None]:
    if pin_memory is None:
        pin_memory = torch.backends.mps.is_available() is False

    train_loader = None
    if train_ds is not None:
        train_loader = DataLoader(
            TextDataset(train_ds),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            TextDataset(val_ds),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    return train_loader, val_loader
