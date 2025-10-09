# train.py
import torch
import torch.nn.functional as F
import os
from datetime import datetime
import tyro
from bpe_tokenizer import get_bpe_tokenizer

# 从我们自己的模块中导入
from config import TrainArgs
from model import Seq2SeqTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from preprocess_data import IWSLTDataset, create_tokenize_batch, create_collate_fn


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
        create_tokenize_batch(tokenizer),
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

    pad_token_id = tokenizer.token_to_id("[PAD]")
    sos_token_id = tokenizer.token_to_id("[CLS]")
    eos_token_id = tokenizer.token_to_id("[SEP]")
    collate_fn = create_collate_fn(pad_token_id)

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

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        args=args.model,
        pad_token_id=pad_token_id,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
    ).to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def estimate_loss():
        """Estimates the loss for both train and validation splits."""
        out = {}
        model.eval()
        for split, loader in [("train", train_loader), ("val", val_loader)]:
            losses = []
            num_batches = min(args.eval_iters, len(loader))
            for i, batch in enumerate(loader):
                if i >= num_batches:
                    break
                src = batch["src_tokens"].to(args.device)
                tgt = batch["tgt_tokens"].to(args.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                logits = model(src, tgt_input)
                B, T, C = logits.size()
                logits_flat = logits.view(B * T, C)
                tgt_flat = tgt_output.contiguous().view(B * T)
                loss = torch.nn.functional.cross_entropy(
                    logits_flat, tgt_flat, ignore_index=pad_token_id
                )
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # --- 训练循环 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("Starting training...")

    global_step = 1

    for epoch in range(args.max_epochs):  # 假设你的 TrainArgs 有 max_epochs
        print(f"\n=== Epoch {epoch + 1}/{args.max_epochs} ===")

        for batch_idx, batch in enumerate(train_loader):
            # 评估和保存
            if global_step % args.eval_interval == 0:
                losses = estimate_loss()
                print(
                    f"step {global_step}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                checkpoint_path = os.path.join(
                    save_path, f"model_step_{global_step}.pt"
                )
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_args": args.model,
                        "tokenizer_str": tokenizer.to_str(),
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved: {checkpoint_path}")

            # 获取数据
            src = batch["src_tokens"].to(args.device)
            tgt = batch["tgt_tokens"].to(args.device)

            # Teacher forcing
            tgt_input = tgt[:, :-1]  # 去掉最后一个 token
            tgt_output = tgt[:, 1:]  # 去掉第一个 token

            # 前向传播
            logits = model(src, tgt_input)

            # 计算损失
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            tgt_flat = tgt_output.reshape(B * T)

            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=pad_token_id)

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 梯度裁剪（可选但推荐）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            global_step += 1

            # 打印训练进度
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

    # 保存最终模型
    final_model_path = os.path.join(save_path, "model_final.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_args": args.model,
            "tokenizer_str": tokenizer.to_str(),
        },
        final_model_path,
    )
    print(f"\nFinal model saved to: {final_model_path}")

    # 最终评估
    final_losses = estimate_loss()
    print(f"Final train loss: {final_losses['train']:.4f}")
    print(f"Final val loss: {final_losses['val']:.4f}")
