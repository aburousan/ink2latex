"""Relative-2D encoder and parallel causal attention-refinement decoder."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    d: int = 256
    heads: int = 8
    ff: int = 1024
    encoder_layers: int = 6
    decoder_layers: int = 4
    dropout: float = .1
    max_points: int = 1600
    max_tokens: int = 160

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.width = config.heads, config.d // config.heads
        self.q, self.k, self.v, self.out = [nn.Linear(config.d, config.d) for _ in range(4)]
        self.dropout = nn.Dropout(config.dropout)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.width).transpose(1, 2)

    def project_kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def attend(self, query, kv, mask=None, bias=None, causal=False):
        q = self.split(self.q(query))
        k, v = kv
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.width)
        if bias is not None:
            score = score + bias
        if mask is not None:
            score = score.masked_fill(~mask[:, None, None, :], -10000.)
        if causal:
            future = torch.arange(k.shape[2], device=q.device)[None, :] > torch.arange(q.shape[2], device=q.device)[:, None]
            score = score.masked_fill(future[None, None], -10000.)
        weights = score.float().softmax(dim=-1).to(v.dtype)
        value = (self.dropout(weights) @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out(value), weights

    def forward(self, query, context, **kwargs):
        return self.attend(query, self.project_kv(context), **kwargs)


def feedforward(config):
    return nn.Sequential(nn.Linear(config.d, config.ff), nn.GELU(), nn.Dropout(config.dropout),
                         nn.Linear(config.ff, config.d))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.norm1, self.norm2 = nn.LayerNorm(config.d), nn.LayerNorm(config.d)
        self.ff, self.drop = feedforward(config), nn.Dropout(config.dropout)

    def forward(self, x, mask, bias):
        n = self.norm1(x)
        x = x + self.drop(self.attn(n, n, mask=mask, bias=bias)[0])
        x = x + self.drop(self.ff(self.norm2(x)))
        return x * mask.unsqueeze(-1)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(8, config.d)
        # Mixing across boundaries is intentional; the input retains stroke flags.
        self.convs = nn.ModuleList([nn.Conv1d(config.d, config.d, 5, stride=2, padding=2) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(config.d) for _ in range(2)])
        self.position = nn.Embedding((config.max_points + 3) // 4, config.d)
        self.relative = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, config.heads))
        grid = torch.arange(-32, 33, dtype=torch.float32) / 4
        self.register_buffer("offset_grid", torch.stack(torch.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2))
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.final_norm = nn.LayerNorm(config.d)

    def forward(self, points, point_mask):
        mask, xy = point_mask.bool(), points[:, :, :2]
        x = self.input(points) * mask.unsqueeze(-1)
        lengths = mask.long().sum(1)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(F.gelu(conv(x.transpose(1, 2)).transpose(1, 2)))
            weights = mask.to(points.dtype).unsqueeze(1)
            numerator = F.avg_pool1d((xy * mask.unsqueeze(-1)).transpose(1, 2), 5, 2, 2)
            denominator = F.avg_pool1d(weights, 5, 2, 2).clamp_min(1e-6)
            xy = (numerator / denominator).transpose(1, 2)
            lengths = (lengths + 1) // 2
            mask = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
            x = x * mask.unsqueeze(-1)
        x = x + self.position(torch.arange(x.shape[1], device=x.device))[None]
        offsets = torch.round((xy[:, :, None] - xy[:, None, :]).clamp(-8, 8) * 4).long() + 32
        indices = offsets[..., 0] * 65 + offsets[..., 1]
        # Evaluate the shared MLP on the small grid, rather than on B*S*S pairs.
        bias = self.relative(self.offset_grid)[indices].permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x, mask, bias)
        return self.final_norm(x) * mask.unsqueeze(-1), mask


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(config), Attention(config)
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(config.d) for _ in range(3)]
        self.ff, self.drop = feedforward(config), nn.Dropout(config.dropout)
        self.coverage_weight = nn.Parameter(torch.full((config.heads,), math.log(math.expm1(1.))))
        self.coverage_conv = nn.Conv1d(config.heads, config.heads, 5, padding=2,
                                       groups=config.heads, bias=False)
        # A positive initial local average makes the refinement active immediately.
        nn.init.constant_(self.coverage_conv.weight, 1 / 5)

    def forward(self, x, memory, mask, active, coverage=None, cache=None,
                incremental=False, memory_kv=None):
        n = self.norm1(x)
        k, v = self.self_attn.project_kv(n)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
            key_mask = torch.cat([cache[2], active], 1)
        else:
            key_mask = active
        x = x + self.drop(self.self_attn.attend(n, (k, v), mask=key_mask, causal=not incremental)[0])
        attn = self.cross_attn
        mk, mv = memory_kv if memory_kv is not None else attn.project_kv(memory)
        scores = (attn.split(attn.q(self.norm2(x))) @ mk.transpose(-1, -2)) / math.sqrt(attn.width)
        scores = scores.float().masked_fill(~mask[:, None, None, :], -10000.)
        first = scores.softmax(-1) * active[:, None, :, None] * mask[:, None, None, :]
        if incremental:
            history = coverage[:, :, None, :]
        else:
            # Exclusive sum: the current or any future target cannot affect this step.
            history = first.cumsum(2) - first
        b, h, t, s = history.shape
        local = self.coverage_conv(history.transpose(1, 2).reshape(b * t, h, s))
        local = local.reshape(b, t, h, s).transpose(1, 2)
        bias = -F.softplus(self.coverage_weight)[None, :, None, None] * F.relu(local.float())
        refined = (scores + bias).softmax(-1) * mask[:, None, None, :]
        value = (attn.dropout(refined.to(mv.dtype)) @ mv).transpose(1, 2).reshape(x.shape)
        x = x + self.drop(attn.out(value))
        x = x + self.drop(self.ff(self.norm3(x)))
        # Each layer caches its own first attention, matching the parallel cumsum.
        updated = coverage + first[:, :, 0] if incremental else None
        return x, refined * active[:, None, :, None], (k, v, key_mask), updated


class Ink2Latex(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        if c.d % c.heads:
            raise ValueError("d must be divisible by heads")
        self.encoder = Encoder(c)
        self.embedding = nn.Embedding(vocab_size, c.d, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.position = nn.Embedding(c.max_tokens + 1, c.d)
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.decoder_layers)])
        self.norm = nn.LayerNorm(c.d)

    def project_memory(self, memory):
        return [layer.cross_attn.project_kv(memory) for layer in self.layers]

    def initial_coverage(self, memory):
        return memory.new_zeros((len(self.layers), memory.shape[0], self.config.heads,
                                 memory.shape[1]), dtype=torch.float32)

    def step(self, token, memory, mask, coverage, position, caches=None, memory_kv=None,
             active=None):
        x = self.embedding(token[:, None]) + self.position.weight[position][None, None]
        active = token.ne(0) if active is None else active
        new_caches, updated = [], []
        for i, layer in enumerate(self.layers):
            x, _, cache, cov = layer(x, memory, mask, active[:, None], coverage[i],
                                     None if caches is None else caches[i], True,
                                     None if memory_kv is None else memory_kv[i])
            new_caches.append(cache)
            updated.append(cov)
        logits = F.linear(self.norm(x[:, 0]), self.embedding.weight)
        return logits, torch.stack(updated), new_caches

    def decode(self, tokens, memory, memory_mask, active=None):
        active = tokens.ne(0) if active is None else active
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))[None]
        for layer in self.layers:
            x, attention, _, _ = layer(x, memory, memory_mask, active)
        logits = F.linear(self.norm(x), self.embedding.weight)
        return logits, attention.float().mean(1).sum(1)

    def prefix_step(self, tokens, memory, memory_mask):
        """Recompute the full prefix, returning only its last logits for ONNX."""
        return self.decode(tokens, memory, memory_mask)[0][:, -1]

    def forward(self, points, point_mask, decoder_tokens, targets=None):
        memory, mask = self.encoder(points, point_mask)
        active = targets.ne(0) if targets is not None else decoder_tokens.ne(0)
        logits, coverage = self.decode(decoder_tokens, memory, mask, active)
        return logits, coverage, mask


def relative_coverage(coverage, ink_mask, target_lengths=None):
    # The sum also recovers T for callers that only have the final attention.
    lengths = ((coverage * ink_mask).sum(-1) if target_lengths is None else target_lengths)
    return coverage * ink_mask.sum(-1, keepdim=True) / lengths[:, None].clamp_min(1e-6)


def coverage_loss(coverage, ink_mask, target_lengths=None):
    chat = relative_coverage(coverage, ink_mask, target_lengths)
    denominator = ink_mask.sum().clamp_min(1)
    twice = (F.relu(chat - 4.) * ink_mask).sum() / denominator
    unread = (F.relu(.25 - chat) * ink_mask).sum() / denominator
    return twice, unread


def training_loss(logits, targets, coverage, ink_mask):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                         ignore_index=0, label_smoothing=.1)
    lengths = targets.ne(0).sum(-1)
    twice, unread = coverage_loss(coverage, ink_mask, lengths)
    chat = relative_coverage(coverage, ink_mask, lengths)
    fraction = ((chat < .25) * ink_mask).sum() / ink_mask.sum().clamp_min(1)
    return ce + .05 * twice + .05 * unread, {
        "cross_entropy": ce, "coverage_twice": twice, "coverage_unread": unread,
        "coverage_unread_fraction": fraction}
