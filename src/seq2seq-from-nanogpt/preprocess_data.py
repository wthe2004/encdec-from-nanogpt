from datasets import load_dataset
from tokenizers import Tokenizer
import torch
import os
from datetime import datetime
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import tyro
from bpe_tokenizer import get_bpe_tokenizer
from config import TrainArgs
from model import TransformerDecoder
from torch.utils.data import DataLoader, random_split, Dataset


def tokenize_item(example):
    return {
        "src_tokens": [tokenizer.encode(example["src"]).ids],
        "tgt_tokens": [tokenizer.encode(example["ref"]).ids],
    }


def tokenize_batch(examples):
    return {
        "src_tokens": [tokenizer.encode(text).ids for text in examples["src"]],
        "tgt_tokens": [tokenizer.encode(text).ids for text in examples["ref"]],
    }


class IWSLTDataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_length=512):
        self.data = dataset[split]  # already tokenized
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "src_tokens": self.data[idx]["src_tokens"][: self.max_length],
            "tgt_tokens": self.data[idx]["tgt_tokens"][: self.max_length],
        }


def collate_fn(batch):
    src_tokens = [item["src_tokens"] for item in batch]
    tgt_tokens = [item["tgt_tokens"] for item in batch]

    max_src_len = max(len(tokens) for tokens in src_tokens)
    max_tgt_len = max(len(tokens) for tokens in tgt_tokens)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_src_tokens = []
    padded_tgt_tokens = []

    for src, tgt in zip(src_tokens, tgt_tokens):
        padded_src = src + [pad_token_id] * (max_src_len - len(src))
        padded_src_tokens.append(padded_src)
        padded_tgt = tgt + [pad_token_id] * (max_tgt_len - len(tgt))
        padded_tgt_tokens.append(padded_tgt)

    return {
        "src_tokens": torch.tensor(padded_src_tokens, dtype=torch.long),
        "tgt_tokens": torch.tensor(padded_tgt_tokens, dtype=torch.long),
        "src_padding_mask": (
            torch.tensor(padded_src_tokens, dtype=torch.long) != pad_token_id
        ).long(),
    }


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)

    torch.manual_seed(args.seed)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.exp_name}__seed{args.seed}__{run_timestamp}"
    save_path = os.path.join(args.out_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"All results will be saved in: {save_path}")

    # --- 数据加载与预处理 (使用BPE) ---
    tokenizer = get_bpe_tokenizer(
        args.dataset_name, args.vocab_size, args.tokenizer_path, args.lp
    )
    vocab_size = tokenizer.get_vocab_size()

    dataset = load_dataset(args.dataset_name).filter(lambda x: x["lp"] == args.lp)

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=list(dataset[args.split_name].features.keys()),
        desc="Tokenizing dataset",
    )
    print(tokenized_dataset)

    train_ratio = 1.0 - args.test_ratio - args.val_ratio
    total_size = len(tokenized_dataset[args.split_name])
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * args.val_ratio)
    test_size = total_size - train_size - val_size

    train_data, val_data, test_data = random_split(
        IWSLTDataset(
            tokenized_dataset,
            args.split_name,
            tokenizer,
            max_length=args.model.block_size,
        ),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
        num_workers=8,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
        num_workers=8,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=args.device == "cuda",
        num_workers=8,
    )
    print(
        f"Dataset split into {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples."
    )
