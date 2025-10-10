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


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)

    torch.manual_seed(args.seed)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.exp_name}__seed{args.seed}__{run_timestamp}"
    save_path = os.path.join(args.out_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"All results will be saved in: {save_path}")

    # --- 数据加载与预处理 (使用BPE) ---
    tokenizer = get_bpe_tokenizer(args.train_file, args.vocab_size, args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()  # 获取实际的词汇表大小

    with open(args.train_file, "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, test_data = data[:n], data[n:]

    # --- 模型初始化 ---
    model = TransformerDecoder(
        vocab_size, args.model, pad_token_id=tokenizer.token_to_id("[PAD]")
    ).to(args.device)

    # --- 辅助函数 ---
    def get_batch(split):
        """Generates a small batch of data of inputs x and targets y."""
        data = train_data if split == "train" else test_data
        ix = torch.randint(len(data) - args.model.block_size, (args.batch_size,))
        x = torch.stack([data[i : i + args.model.block_size] for i in ix]).to(
            args.device
        )
        y = torch.stack([data[i + 1 : i + args.model.block_size + 1] for i in ix]).to(
            args.device
        )
        return x, y

    @torch.no_grad()
    def estimate_loss():
        """Estimates the loss for both train and validation splits."""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # --- 训练过程 (逻辑不变) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("Starting training...")
    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            checkpoint_path = os.path.join(save_path, f"model_iter_{iter}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to: {checkpoint_path}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_model_path = os.path.join(save_path, "model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
