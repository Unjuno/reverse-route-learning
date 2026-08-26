import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# TinyStories-8M alternates global and local GPT-Neo attention. The local
# attention window is 256 tokens. This minimal runtime intentionally uses a
# single causal-attention implementation; for sequences <=256 tokens it is
# equivalent to the published local/global pattern, but it must not be used
# beyond that range without implementing the local mask.
MAX_EXACT_SEQUENCE = 256


class Attention(nn.Module):
    def __init__(self, h: int = 256, heads: int = 16):
        super().__init__()
        self.h = h
        self.heads = heads
        self.d = h // heads
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.out_proj = nn.Linear(h, h, bias=True)

    def split(self, x):
        *b, t, h = x.shape
        return x.view(*b, t, self.heads, self.d).transpose(-3, -2)

    def merge(self, x):
        x = x.transpose(-3, -2).contiguous()
        *b, t, heads, d = x.shape
        return x.view(*b, t, heads * d)

    def forward(self, x, past=None, use_cache=False):
        q = self.split(self.q_proj(x))
        k = self.split(self.k_proj(x))
        v = self.split(self.v_proj(x))
        past_len = 0
        if past is not None:
            pk, pv = past
            past_len = pk.size(-2)
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d)
        tq, tk = x.size(-2), k.size(-2)
        qi = torch.arange(past_len, past_len + tq, device=x.device)[:, None]
        kj = torch.arange(tk, device=x.device)[None, :]
        scores = scores.masked_fill(kj > qi, torch.finfo(scores.dtype).min)
        a = torch.softmax(scores, dim=-1)
        y = self.out_proj(self.merge(torch.matmul(a, v)))
        return y, ((k, v) if use_cache else None)


class Block(nn.Module):
    def __init__(self, h: int = 256, heads: int = 16, mlp: int = 1024):
        super().__init__()
        self.ln_1 = nn.LayerNorm(h, eps=1e-5)
        self.attn = Attention(h, heads)
        self.ln_2 = nn.LayerNorm(h, eps=1e-5)
        self.fc = nn.Linear(h, mlp)
        self.proj = nn.Linear(mlp, h)

    def forward(self, x, past=None, use_cache=False):
        a, p = self.attn(self.ln_1(x), past, use_cache)
        x = x + a
        y = self.fc(self.ln_2(x))
        y = 0.5 * y * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (y + 0.044715 * y.pow(3))))
        x = x + self.proj(y)
        return x, p


class TinyStoriesNeo(nn.Module):
    """Minimal GPT-Neo-compatible runtime for roneneldan/TinyStories-8M.

    It intentionally avoids a Transformers dependency so the controlled
    experiments operate directly on the published PyTorch state dict.

    Important: this implementation is exact for the repository experiments,
    whose total sequence lengths stay within TinyStories-8M's 256-token local
    attention window. Longer sequences are rejected explicitly.
    """

    def __init__(self, model_path: str | Path):
        super().__init__()
        self.model_path = Path(model_path)
        self.wte = nn.Embedding(50257, 256)
        self.wpe = nn.Embedding(2048, 256)
        self.h = nn.ModuleList([Block() for _ in range(8)])
        self.ln_f = nn.LayerNorm(256, eps=1e-5)
        self.load_checkpoint()

    def load_checkpoint(self):
        sd = torch.load(self.model_path, map_location="cpu")
        ns = {
            "wte.weight": sd["transformer.wte.weight"],
            "wpe.weight": sd["transformer.wpe.weight"],
            "ln_f.weight": sd["transformer.ln_f.weight"],
            "ln_f.bias": sd["transformer.ln_f.bias"],
        }
        for i in range(8):
            p = f"transformer.h.{i}."
            q = f"h.{i}."
            ns[q + "ln_1.weight"] = sd[p + "ln_1.weight"]
            ns[q + "ln_1.bias"] = sd[p + "ln_1.bias"]
            ns[q + "ln_2.weight"] = sd[p + "ln_2.weight"]
            ns[q + "ln_2.bias"] = sd[p + "ln_2.bias"]
            for n in ["q_proj", "k_proj", "v_proj"]:
                ns[q + "attn." + n + ".weight"] = sd[p + "attn.attention." + n + ".weight"]
            ns[q + "attn.out_proj.weight"] = sd[p + "attn.attention.out_proj.weight"]
            ns[q + "attn.out_proj.bias"] = sd[p + "attn.attention.out_proj.bias"]
            ns[q + "fc.weight"] = sd[p + "mlp.c_fc.weight"]
            ns[q + "fc.bias"] = sd[p + "mlp.c_fc.bias"]
            ns[q + "proj.weight"] = sd[p + "mlp.c_proj.weight"]
            ns[q + "proj.bias"] = sd[p + "mlp.c_proj.bias"]
        self.load_state_dict(ns, strict=True)

    def hidden(self, ids, past=None, use_cache=False, positions=None):
        if ids.dim() == 1:
            ids = ids[None, :]
        _, t = ids.shape
        past_len = 0 if past is None else past[0][0].size(-2)
        total_len = past_len + t
        if total_len > MAX_EXACT_SEQUENCE:
            raise ValueError(
                f"TinyStoriesNeo minimal runtime is only exact up to {MAX_EXACT_SEQUENCE} tokens; "
                f"requested sequence length {total_len}."
            )
        if positions is None:
            positions = torch.arange(past_len, past_len + t, device=ids.device)
        x = self.wte(ids) + self.wpe(positions)[None, :, :]
        new = []
        for i, blk in enumerate(self.h):
            x, p = blk(x, None if past is None else past[i], use_cache)
            new.append(p)
        x = self.ln_f(x)
        return x, (new if use_cache else None)

    def forward(self, ids, past=None, use_cache=False, return_hidden=False):
        h, p = self.hidden(ids, past, use_cache)
        z = F.linear(h, self.wte.weight)
        if use_cache:
            return (z, p, h) if return_hidden else (z, p)
        return (z, h) if return_hidden else z

    @torch.no_grad()
    def last_logits(self, ids):
        return self.forward(ids)[:, -1, :]

    @torch.no_grad()
    def greedy(self, seq, n: int):
        out = list(seq)
        for _ in range(n):
            out.append(int(self.last_logits(torch.tensor(out, dtype=torch.long)).argmax()))
        return out


@torch.no_grad()
def kv_prefix_cache(model, prefix):
    ids = torch.tensor(prefix, dtype=torch.long)[None, :]
    z, c, h = model(ids, use_cache=True, return_hidden=True)
    return {"past": c, "logits": z[:, -1, :], "hidden": h[:, -1, :]}


@torch.no_grad()
def kv_step(model, cache, tokens, return_hidden=False):
    tok = tokens if torch.is_tensor(tokens) else torch.tensor(tokens, dtype=torch.long)
    if tok.dim() == 0:
        tok = tok[None]
    tok = tok.long().view(-1, 1)
    past = []
    for k, v in cache["past"]:
        if k.size(0) == 1 and tok.size(0) > 1:
            k = k.expand(tok.size(0), -1, -1, -1)
            v = v.expand(tok.size(0), -1, -1, -1)
        past.append((k, v))
    z, c, h = model(tok, past=past, use_cache=True, return_hidden=True)
    nc = {"past": c, "logits": z[:, -1, :], "hidden": h[:, -1, :]}
    return (z[:, -1, :], nc, h[:, -1, :]) if return_hidden else (z[:, -1, :], nc)
