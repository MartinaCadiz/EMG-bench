"""Channel-count-agnostic EMG encoder, predictor and the JEPA objective."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_1d(n, dim, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    emb = torch.zeros(n, dim, device=device)
    emb[:, 0::2] = torch.sin(pos * freq)
    emb[:, 1::2] = torch.cos(pos * freq)
    return emb


def gather_tokens(x, idx):
    """x (B, L, D), idx (B, K) -> (B, K, D)"""
    return x.gather(1, idx[:, :, None].expand(-1, -1, x.shape[-1]))


def transformer(dim, depth, heads):
    layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class GridPos(nn.Module):
    """Position of token (channel c, time patch n): sinusoidal in time + learned per channel.

    Tokens are flattened channel-major: index = c * N + n.
    """

    def __init__(self, dim, max_channels):
        super().__init__()
        self.dim = dim
        self.channel = nn.Parameter(torch.zeros(max_channels, dim))
        nn.init.trunc_normal_(self.channel, std=0.02)

    def forward(self, C, N):
        assert C <= self.channel.shape[0], f"{C} channels > max_channels={self.channel.shape[0]}"
        time = sincos_1d(N, self.dim, self.channel.device)
        return (self.channel[:C, None, :] + time[None, :, :]).reshape(C * N, self.dim)


class EMGEncoder(nn.Module):
    """Raw EMG (B, C, T) -> tokens (B, C*N, dim), one token per channel per time patch.

    The same 1-D patch embedding is applied to every channel, so the encoder
    accepts any channel count up to max_channels.
    """

    def __init__(self, patch_len, dim=384, depth=8, heads=6, max_channels=64):
        super().__init__()
        self.patch_len = patch_len
        self.dim = dim
        self.patch_embed = nn.Conv1d(1, dim, kernel_size=patch_len, stride=patch_len)
        self.pos = GridPos(dim, max_channels)
        self.blocks = transformer(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, idx=None):
        B, C, T = x.shape
        N = T // self.patch_len
        tok = self.patch_embed(x.reshape(B * C, 1, T))            # (B*C, dim, N)
        tok = tok.transpose(1, 2).reshape(B, C * N, self.dim)
        tok = tok + self.pos(C, N)
        if idx is not None:
            tok = gather_tokens(tok, idx)
        return self.norm(self.blocks(tok))


class Predictor(nn.Module):
    """Predicts target-encoder latents at masked positions from context tokens."""

    def __init__(self, enc_dim, dim=192, depth=4, heads=6, max_channels=64):
        super().__init__()
        self.proj = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos = GridPos(dim, max_channels)
        self.blocks = transformer(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, enc_dim)

    def forward(self, ctx, ctx_idx, tgt_idx, C, N):
        pos = self.pos(C, N)
        x = self.proj(ctx) + pos[ctx_idx]
        m = self.mask_token + pos[tgt_idx]
        h = self.blocks(torch.cat([x, m], dim=1))
        return self.out(self.norm(h[:, x.shape[1]:]))


def sample_masks(B, C, N, n_blocks, block_len, drop_channels, device):
    """Mask n_blocks non-overlapping time spans across ALL channels.

    Channels are strongly correlated, so masking a span in only some channels
    would let the model copy from its neighbours. Optionally the context also
    loses whole channels (drop_channels).

    Returns ctx_idx (B, (C-drop)*(N-M)), tgt_idx (B, C*M), t_masked (B, M),
    with M = n_blocks * block_len and targets ordered channel-major.
    """
    M = n_blocks * block_len
    slots = N - n_blocks * (block_len - 1)
    assert slots >= n_blocks and M < N, f"cannot place {n_blocks} blocks of {block_len} in {N} patches"
    assert 0 <= drop_channels < C

    s = torch.rand(B, slots, device=device).argsort(1)[:, :n_blocks].sort(1).values
    starts = s + torch.arange(n_blocks, device=device) * (block_len - 1)
    t_masked = (starts[:, :, None] + torch.arange(block_len, device=device)).reshape(B, M)

    hidden = torch.zeros(B, N, dtype=torch.bool, device=device).scatter_(1, t_masked, True)
    t_visible = (~hidden).nonzero()[:, 1].reshape(B, N - M)
    ch_keep = torch.rand(B, C, device=device).argsort(1)[:, :C - drop_channels].sort(1).values

    ctx_idx = (ch_keep[:, :, None] * N + t_visible[:, None, :]).reshape(B, -1)
    tgt_idx = (torch.arange(C, device=device)[None, :, None] * N + t_masked[:, None, :]).reshape(B, -1)
    return ctx_idx, tgt_idx, t_masked


class EMGJEPA(nn.Module):
    """JEPA on raw EMG, with an optional video-latent target.

    self loss : predictor output vs EMA target-encoder latents at masked tokens.
    video loss: predictor output, averaged over channels per masked time patch
                and passed through a linear head, vs the frozen video latent of
                that moment (cosine distance). Set self_weight=0 for a pure
                video-target model.
    """

    def __init__(self, patch_len, dim=384, depth=8, heads=6, pred_dim=192, pred_depth=4,
                 pred_heads=6, max_channels=64, n_blocks=4, block_len=4, drop_channels=0,
                 self_weight=1.0, video_weight=0.0, video_dim=None):
        super().__init__()
        self.encoder = EMGEncoder(patch_len, dim, depth, heads, max_channels)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(dim, pred_dim, pred_depth, pred_heads, max_channels)
        self.mask_cfg = (n_blocks, block_len, drop_channels)
        self.self_weight = self_weight
        self.video_weight = video_weight
        self.video_head = nn.Linear(dim, video_dim) if video_weight > 0 else None

    def forward(self, x, video=None):
        B, C, T = x.shape
        N = T // self.encoder.patch_len
        ctx_idx, tgt_idx, t_masked = sample_masks(B, C, N, *self.mask_cfg, device=x.device)

        ctx = self.encoder(x, ctx_idx)
        with torch.no_grad():
            tgt = self.target_encoder(x)
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))
            tgt = gather_tokens(tgt, tgt_idx)
        pred = self.predictor(ctx, ctx_idx, tgt_idx, C, N)

        loss_self = F.smooth_l1_loss(pred, tgt)
        out = {"loss_self": loss_self.detach(),
               # collapse monitor: goes to 0 if every token maps to the same latent
               "target_std": tgt.float().reshape(-1, tgt.shape[-1]).std(0).mean()}
        loss = self.self_weight * loss_self

        if self.video_head is not None:
            assert video is not None, "video_weight > 0 needs video latents in the batch"
            per_time = pred.reshape(B, C, -1, pred.shape[-1]).mean(1)       # (B, M, dim)
            v = gather_tokens(video, t_masked)                              # (B, M, Dv)
            loss_video = 1 - F.cosine_similarity(self.video_head(per_time).float(), v.float(), dim=-1).mean()
            out["loss_video"] = loss_video.detach()
            loss = loss + self.video_weight * loss_video

        out["loss"] = loss
        return out

    @torch.no_grad()
    def update_target(self, momentum):
        for pt, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(momentum).add_(p.detach(), alpha=1 - momentum)
