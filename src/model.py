import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func
from tokenizers import Tokenizer


class FFLayer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        dff = emb_dim * 8 // 3
        self.W1 = nn.Linear(emb_dim, dff)
        self.W2 = nn.Linear(dff, emb_dim)
        self.W3 = nn.Linear(emb_dim, dff)

    def forward(self, x):
        y = self.silu(self.W1(x))
        y = y * self.W3(x)
        y = self.W2(y)
        return y

    @staticmethod
    def silu(x):
        return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, emb_size, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(emb_size))
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x.pow(2)) + self.eps)
        x = (x / rms) * self.g
        return x.to(in_dtype)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self, input_dim, emb_dim=512, max_seq_len=1050, n_heads=8, p_dropout=0.1
    ):
        super().__init__()
        # self.Wq = nn.Linear(input_dim, emb_dim, bias=True)
        # self.Wk = nn.Linear(input_dim, emb_dim, bias=True)
        # self.Wv = nn.Linear(input_dim, emb_dim, bias=True)
        self.W = nn.Linear(input_dim, emb_dim * 3, bias=True)
        self.Wo = nn.Linear(emb_dim, emb_dim)
        self.ff = FFLayer(emb_dim)
        self.ln_mha = RMSNorm(emb_dim)
        self.ln_ff = RMSNorm(emb_dim)
        self.scale = 1 / np.sqrt(emb_dim)
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        attn_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)) == 0
        self.register_buffer("_causal_mask", attn_mask)

    def forward(self, x):
        y = self.mha(x)
        y_prenorm = y + x

        y = self.mlp(y_prenorm)
        y = y_prenorm + y

        return y

    def mlp(self, x):
        y = self.ln_ff(x)
        y = self.ff(y)
        y = F.dropout(y, p=self.p_dropout)

        return y

    def mha(self, x):
        b, t, e = x.shape
        s = e // self.n_heads
        dtype = x.dtype

        # x = self.ln_mha(x)
        # q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        # q = q.view(b, t, self.n_heads, s).transpose(1, 2)
        # k = k.view(b, t, self.n_heads, s).transpose(1, 2)
        # v = v.view(b, t, self.n_heads, s).transpose(1, 2)

        x = self.ln_mha(x)
        qkv = self.W(x)
        qkv = qkv.view(b, t, self.n_heads, 3 * s)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.split(s, dim=-1)

        attn = q @ k.transpose(-1, -2)
        attn = attn.to(torch.float32)
        attn = attn * self.scale
        attn = attn.masked_fill(self._causal_mask[:t, :t], -torch.inf)
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.p_dropout)
        attn = attn.to(dtype)
        y = attn @ v
        y = y.transpose(-2, -3).reshape(b, t, e)
        y = self.Wo(y)
        y = F.dropout(y, p=self.p_dropout)

        return y


class FlashSelfAttentionBlock(nn.Module):
    def __init__(
        self, input_dim, emb_dim=512, max_seq_len=1050, n_heads=8, p_dropout=0.1
    ):
        super().__init__()
        self.W = nn.Linear(input_dim, emb_dim * 3, bias=True)
        self.Wo = nn.Linear(emb_dim, emb_dim)
        self.ff = FFLayer(emb_dim)
        self.ln_mha = RMSNorm(emb_dim)
        self.ln_ff = RMSNorm(emb_dim)
        self.scale = 1 / np.sqrt(emb_dim)
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        attn_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)) == 0
        self.register_buffer("_causal_mask", attn_mask)

    def forward(self, x):
        y = self.mha(x)
        y_prenorm = y + x

        y = self.mlp(y_prenorm)
        y = y_prenorm + y

        return y

    def mlp(self, x):
        y = self.ln_ff(x)
        y = self.ff(y)
        y = F.dropout(y, p=self.p_dropout)

        return y

    def mha(self, x):
        b, t, e = x.shape
        s = e // self.n_heads

        x = self.ln_mha(x)
        qkv = self.W(x)

        qkv = qkv.view(b, t, self.n_heads, 3, s)
        qkv = qkv.transpose(2, 3)
        y = flash_attn_qkvpacked_func(qkv, dropout_p=self.p_dropout)
        y = y.reshape(b, t, e)
        y = self.Wo(y)
        y = F.dropout(y, p=self.p_dropout)

        return y


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocab_size=8124,
        emb_dim=1024,
        attn_blocks=24,
        max_seq_len=1024,
        n_heads=16,
        p_dropout=0.1,
        attn_type="vanilla",
        tokenizer_path="tokenizer.json",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        if attn_type == "vanilla":
            self.attn = nn.Sequential(
                *[
                    CausalSelfAttentionBlock(
                        emb_dim, emb_dim, max_seq_len, n_heads, p_dropout
                    )
                    for _ in range(attn_blocks)
                ]
            )
        elif attn_type == "flash":
            self.attn = self.attn = nn.Sequential(
                *[
                    FlashSelfAttentionBlock(
                        emb_dim, emb_dim, max_seq_len, n_heads, p_dropout
                    )
                    for _ in range(attn_blocks)
                ]
            )
        self.mlp = nn.Linear(emb_dim, vocab_size)
        self.layer_norm = RMSNorm(emb_dim)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.p_dropout = p_dropout

        self.register_buffer("_device_tracker", torch.empty(0))
        pe = self._compute_pe(max_seq_len, emb_dim)
        self.register_buffer("pe", pe)

    @property
    def device(self):
        return self._device_tracker.device

    def forward(self, x):
        _, t = x.shape
        x = self.emb(x)
        x = x + self.pe[:, :t, :]
        x = F.dropout(x, p=self.p_dropout)
        x = self.attn(x)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

    @torch.no_grad
    def run_inference(self, text_input, tau=1.0, k=10):
        x = self.tokenizer.encode(text_input).ids[:-1]
        eos_id = self.tokenizer.token_to_id("<eos>")
        next_token = -1
        cur_iter = 0
        max_iter = 128

        while next_token != eos_id and cur_iter < max_iter:
            x_tensor = torch.tensor(x).to(self.device)
            logits = self.forward(x_tensor.unsqueeze(0))
            q = F.softmax(logits / tau, dim=-1)[:, -1]
            topk = q.topk(k=k)
            next_token_index = topk.values.multinomial(1).item()
            next_token = topk.indices[0, next_token_index]
            x += [next_token]
            cur_iter += 1

        return self.tokenizer.decode(x)

    def _compute_pe(self, max_seq_len, emb_dim):
        pe = torch.zeros(max_seq_len, emb_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-np.log(10000.0) / emb_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        return pe
