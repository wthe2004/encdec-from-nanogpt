# model.py
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelArgs


class Head(nn.Module):
    """
    One head of self-attention.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_size = args.n_embd // args.num_heads
        self.key = nn.Linear(args.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(args.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(args.n_embd, self.head_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask=None):
        # print(x.shape)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # 计算注意力分数 ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        )  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        if mask is not None:
            wei = wei.masked_fill(mask[:, 0, :T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        # 对 value 进行加权聚合
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    args,
                )
                for _ in range(args.num_heads)
            ]
        )
        self.proj = nn.Linear(args.n_embd, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask=None):
        # print(f"MultiHeadAttention input x shape: {x.shape}")
        head_outputs = [h(x=x, mask=mask) for h in self.heads]
        # print(f"Head outputs shapes: {[h.shape for h in head_outputs]}")
        out = torch.cat(head_outputs, dim=-1)
        # print(f"Concatenated output shape: {out.shape}")
        out = self.dropout(self.proj(out))
        # print(f"MultiHeadAttention output shape: {out.shape}")
        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.ReLU(),
            nn.Linear(4 * args.n_embd, args.n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.sa = MultiHeadAttention(args)
        self.ffwd = FeedForward(args)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

    def forward(self, x, mask=None):
        # print(f"Block input: {x.shape}")
        sa_out = self.sa(self.ln1(x), mask=mask)
        x = x + sa_out
        # print(f"After attention: {x.shape}")
        x = x + self.ffwd(self.ln2(x))
        # print(f"Block output: {x.shape}\n")
        return x


class TransformerDecoder(nn.Module):
    """
    The main Transformer Decoder model. Renamed from BigramLanguageModel for clarity.
    """

    def __init__(self, vocab_size: int, args: ModelArgs, pad_token_id: int = 0):
        super().__init__()
        self.args = args
        self.pad_token_id = pad_token_id

        self.token_embedding_table = nn.Embedding(vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)

        self.register_buffer(
            "tril", torch.tril(torch.ones(args.block_size, args.block_size).bool())
        )

        self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = nn.Linear(args.n_embd, vocab_size)

    def _create_combined_mask(self, idx):
        B, T = idx.shape

        # Padding mask
        padding_mask = (idx != self.pad_token_id).reshape(B, 1, 1, T)

        # Causal mask
        causal_mask = self.tril[:T, :T].reshape(1, 1, T, T)

        # Combined mask
        combined_mask = padding_mask & causal_mask  # (B,1,T,T)
        return combined_mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        mask = self._create_combined_mask(idx)

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) 位置编码加上词嵌入

        for block in self.blocks:
            x = block(x=x, mask=mask)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given a context.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 裁剪 idx 以确保不超过 block_size
            idx_cond = idx[:, -self.args.block_size :]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步
            logits = logits[:, -1, :]  # 变为 (B, C)
            # 应用 softmax 得到概率
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将采样的索引附加到运行序列中
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        self.train()
        return idx
