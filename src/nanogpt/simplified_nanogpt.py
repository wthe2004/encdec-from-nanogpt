import torch
import torch.nn as nn
from torch.nn import functional as F

import os
from datetime import datetime

# ==============================================================================
# Hyperparameters & Configuration
# ==============================================================================
batch_size = 64  # 一次并行处理多少个独立的序列
block_size = 256  # 用于预测的最大上下文长度
max_iters = 50  # 最大训练迭代次数
eval_interval = max_iters // 5  # 评估损失的频率
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384  # 词嵌入维度
n_layers = 6  # Transformer Block 的层数
n_head = 6  # 注意力头的数量 (此变量在原代码中未被使用)
dropout = 0.2
num_heads = 4  # 实际在模型中使用的注意力头数量
seed = 1337
exp_name = "tinyshakespeare_char_gpt"
out_dir = "runs"
# ==============================================================================


# ==============================================================================
# Model Definition
# ==============================================================================
class Head(nn.Module):
    """一个自注意力头 (one head of self-attention)"""

    tril: torch.Tensor

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # 下三角矩阵
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        # 计算注意力分数 ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # 对 value 进行加权聚合
        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """并行计算的多个注意力头"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """一个简单的线性层加非线性激活函数"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_head=4):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 每个 token 直接从查找表中读取下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=num_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) 位置编码加上词嵌入
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是 (B, T) 形状的当前上下文索引数组
        for _ in range(max_new_tokens):
            # 裁剪 idx 以确保不超过 block_size
            idx_cond = idx[:, -block_size:]
            # 获取预测
            logits, loss = self(idx_cond)
            # 只关注最后一个时间步
            logits = logits[:, -1, :]  # 变为 (B, C)
            # 应用 softmax 得到概率
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 将采样的索引附加到运行序列中
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ==============================================================================


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":

    torch.manual_seed(seed)
    print(f"using device: {device}")

    # 创建一个漂亮的、唯一的保存路径，模仿 CleanRL 风格
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{exp_name}__seed{seed}__{run_timestamp}"
    save_path = os.path.join(out_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"所有结果将保存在: {save_path}")

    # ----------------
    # 数据加载与预处理
    # ----------------
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open("./ref/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 获取文本中所有不重复的字符
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))

    # 创建字符到整数的映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # 编码器: 输入字符串, 输出整数列表
    decode = lambda l: "".join([itos[i] for i in l])  # 解码器: 输入整数列表, 输出字符串

    # 划分训练集和验证集
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # 前90%为训练集, 剩下为验证集
    train_data = data[:n]
    test_data = data[n:]

    # ----------------
    # 模型初始化
    # ----------------
    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # ----------------
    # 辅助函数 (定义在此处以捕获局部作用域中的变量)
    # ----------------
    def get_batch(split):
        """生成一小批输入x和目标y"""
        data = train_data if split == "train" else test_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        """评估模型在训练集和验证集上的损失"""
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # ----------------
    # 训练过程
    # ----------------
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # 每隔一段时间评估一次损失
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            checkpoint_path = os.path.join(save_path, f"model_iter_{iter}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"model saved in: {checkpoint_path}")

        # 采样一批数据
        xb, yb = get_batch("train")

        # 评估损失
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_model_path = os.path.join(save_path, "model_iter_{max_iters}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"final model saved in: {final_model_path}")

    # ----------------
    # 生成文本
    # ----------------
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# ==============================================================================
