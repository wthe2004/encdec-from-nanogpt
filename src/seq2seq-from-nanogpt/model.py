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

    def forward(self, query, key, value, mask=None):
        # print(x.shape)
        # self attention: q=k=v=x
        # cross-attention: q=target, k=v=source
        k = self.key(key)
        q = self.query(query)
        v = self.value(value)
        # 计算注意力分数 ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * (self.head_size**-0.5)
        )  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        # 对 value 进行加权聚合
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

    def forward(self, query, key, value, mask=None):
        # print(f"MultiHeadAttention input x shape: {x.shape}")
        head_outputs = [
            h(query=query, key=key, value=value, mask=mask) for h in self.heads
        ]
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


class EncoderBlock(nn.Module):
    """
    Transformer block: self-attention + feedforward, with layernorm and residuals.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = MultiHeadAttention(args)
        self.ffwd = FeedForward(args)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

    def forward(self, x, mask=None):
        x = x + self.self_attn(
            query=self.ln1(x), key=self.ln1(x), value=self.ln1(x), mask=mask
        )
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer block: masked self-attention + cross-attention + feedforward, with layernorm and residuals.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = MultiHeadAttention(args)
        self.cross_attn = MultiHeadAttention(args)
        self.ffwd = FeedForward(args)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.ln3 = nn.LayerNorm(args.n_embd)
        self.ln_enc = nn.LayerNorm(args.n_embd)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        x = x + self.self_attn(
            query=self.ln1(x), key=self.ln1(x), value=self.ln1(x), mask=tgt_mask
        )
        x = x + self.cross_attn(
            query=self.ln2(x),
            key=self.ln_enc(enc_output),
            value=self.ln_enc(enc_output),
            mask=src_mask,
        )
        x = x + self.ffwd(self.ln3(x))
        return x


class Seq2SeqTransformer(nn.Module):
    """
    The main Transformer Decoder model. Renamed from BigramLanguageModel for clarity.
    """

    def __init__(self, vocab_size: int, args: ModelArgs, pad_token_id: int = 0):
        super().__init__()
        self.args = args
        self.pad_token_id = pad_token_id

        # encoder components
        self.src_token_embedding = nn.Embedding(vocab_size, args.n_embd)
        self.src_position_embedding = nn.Embedding(args.block_size, args.n_embd)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(args) for _ in range(args.n_layers)]
        )
        self.encoder_ln = nn.LayerNorm(args.n_embd)

        # decoder components
        self.tgt_token_embedding = nn.Embedding(vocab_size, args.n_embd)
        self.tgt_position_embedding = nn.Embedding(args.block_size, args.n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(args) for _ in range(args.n_layers)]
        )
        self.decoder_ln = nn.LayerNorm(args.n_embd)

        # Output projection layer
        self.lm_head = nn.Linear(args.n_embd, vocab_size)

    def create_padding_mask(self, seq):
        # seq: (B, T)
        mask = seq != self.pad_token_id
        return mask.view(seq.size(0), 1, 1, seq.size(1))
        # (B, 1, 1, T) -> broadcast to (B, num_heads, T, T) in attention

    def create_look_ahead_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).to(torch.bool)  # (T, T)
        return mask.view(1, 1, size, size)  # (1, 1, T, T) for broadcasting

    def encode(self, src, src_mask=None):
        B, T = src.size()

        tok_emb = self.src_token_embedding(src)  # (B,T,n_embd)
        pos = torch.arange(0, T, device=src.device).view(1, T)  # (1,T)
        pos_emb = self.src_position_embedding(pos)  # (1,T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd)

        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)

        x = self.encoder_ln(x)
        return x  # (B,T,n_embd)

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        _, T = tgt.size()

        tok_emb = self.tgt_token_embedding(tgt)  # (B,T,n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=tgt.device).view(
            1, T
        )  # (1,T)
        pos_emb = self.tgt_position_embedding(pos)  # (1,T,n_embd)
        x = tok_emb + pos_emb  # (B,T,n_embd)

        for block in self.decoder_blocks:
            x = block(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)

        x = self.decoder_ln(x)
        return x  # (B,T,n_embd)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Take in and process masked src and target sequences.
        B means batch size, T means sequence length.
        Args:
            src: (B, T_src)
            tgt: (B, T_tgt)
            src_mask: (B, 1, 1, T_src)
            tgt_mask: (B, 1, T_tgt, T_tgt)
        Returns:
            logits: (B, T_tgt, vocab_size)
        """
        if src_mask is None:
            src_mask = self.create_padding_mask(src)  # (B, 1, 1, T_src)

        if tgt_mask is None:
            T = tgt.size(1)
            causal_mask = self.create_look_ahead_mask(T).to(tgt.device)  # (1, 1, T, T)
            padding_mask = self.create_padding_mask(tgt).to(tgt.device)  # (B, 1, 1, T)
            padding_mask = padding_mask.expand(-1, 1, T, -1)  # (B, 1, T, T)

            tgt_mask = causal_mask & padding_mask  # (B, 1, T, T)

        enc_output = self.encode(src, src_mask)  # (B, T_src, n_embd)
        dec_output = self.decode(
            tgt, enc_output, tgt_mask, src_mask
        )  # (B, T_tgt, n_embd)
        logits = self.lm_head(dec_output)  # (B, T_tgt, vocab_size)
        return logits

    def generate(
        self,
        src,
        max_new_tokens,
        top_k=None,
        temperature=1.0,
        sos_token_id=1,
        eos_token_id=2,
    ):
        """
        Generate new tokens given a source sequence.
        Args:
            src: (B, T_src) source input sequence
            max_new_tokens: int, number of tokens to generate
            top_k: int or None, if not None, use top-k sampling
            temperature: float, temperature for sampling
            sos_token_id: int, start-of-sequence token id
            eos_token_id: int, end-of-sequence token id
        Returns:
            generated: (B, T_src + max_new_tokens)
        """
        B, T_src = src.size()
        device = src.device
        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_mask)  # (B, T_src, n_embd)
        tgt = torch.full(
            (B, 1), sos_token_id, dtype=torch.long, device=device
        )  # (B, 1), start with <sos>

        for _ in range(max_new_tokens):
            _, T = tgt.size()

            if T > self.args.block_size:
                tgt_input = tgt[:, -self.args.block_size :]  # crop to block size
            else:
                tgt_input = tgt
            _, T = tgt_input.size()

            causal_mask = self.create_look_ahead_mask(T).to(device)  # (1, 1, T, T)
            dec_output = self.decode(
                tgt_input, enc_output, tgt_mask=causal_mask, src_mask=src_mask
            )  # (B, T, n_embd)
            logits = self.lm_head(dec_output)  # (B, T, vocab_size
            logits = (
                logits[:, -1, :] / temperature
            )  # (B, vocab_size), focus on last token
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tgt = torch.cat((tgt, next_token), dim=1)
            if (next_token == eos_token_id).all():
                break
        return tgt  # (B, T_src + max_new_tokens)
