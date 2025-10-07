# train.py
import torch
import os
from datetime import datetime
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import tyro
from bpe_tokenizer import get_bpe_tokenizer

# 从我们自己的模块中导入
from config import TrainArgs
from model import TransformerDecoder
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split


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

    dataset = load_dataset(args.dataset_name).filter(lambda x: x["lp"] == "en-de")
    train_data = dataset["train"]

    def preprocess(examples):
        pass
