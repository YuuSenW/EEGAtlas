import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embedding import get_3d_coordinates
from .utils import apply_rotary_emb, rotate_half


CHANNEL_DICT = {k.upper(): v for v, k in enumerate(
    ['FP1', 'FPZ', 'FP2',
     'AF3', 'AF4',
     'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
     'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
     'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
     'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
     'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
     'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
     'O1', 'OZ', 'O2', ])}


class GroupAttention(nn.Module):
    """
    Shared cross-patch attention with channel-conditioned FiLM adapters.

    All channels share one qkv/proj temporal core. A zero-initialized per-channel
    FiLM table modulates q/k/v, preserving channel identity without duplicating
    the full attention matrices. Different samples may therefore use different
    visible channel sets while retaining one fully vectorized attention call.

    Temporal RoPE uses the original patch ids before masking. Summary tokens do
    not participate in this branch and receive a zero residual update.
    """

    def __init__(self, dim, num_heads=8, num_patches=16, num_channels=58,
                 group_dim=512, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.group_dim = group_dim
        self.head_dim = group_dim // num_heads
        self.dim = dim
        assert group_dim % num_heads == 0
        assert self.head_dim % 2 == 0, "RoPE requires an even attention head dimension"

        self.num_patches = num_patches
        self.num_channels = num_channels

        self.qkv = nn.Linear(dim, group_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(group_dim, dim)

        # Per-channel gamma/beta for the concatenated qkv vector. Zero init makes
        # the initial model exactly the shared temporal core.
        self.channel_film = nn.Parameter(
            torch.zeros(num_channels, group_dim * 6)
        )
        self.register_buffer(
            'rope_inv_freq',
            1.0 / (rope_theta ** (
                torch.arange(0, self.head_dim, 2).float() / self.head_dim
            )),
            persistent=False,
        )
        self.attn_drop = attn_drop  # scalar (forward 里手动 dropout, 跟原版一致)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0. else nn.Identity()

    def _temporal_rope(self, patch_ids, batch_size, num_visible, device, dtype):
        if patch_ids is None:
            patch_ids = torch.arange(num_visible, device=device).unsqueeze(0)
            patch_ids = patch_ids.expand(batch_size, -1)
        elif patch_ids.ndim == 1:
            patch_ids = patch_ids.to(device).unsqueeze(0).expand(batch_size, -1)
        else:
            patch_ids = patch_ids.to(device)

        assert patch_ids.shape == (batch_size, num_visible), \
            f"patch_ids shape {tuple(patch_ids.shape)} != {(batch_size, num_visible)}"
        assert patch_ids.min() >= 0, "patch_ids must be non-negative"

        angles = patch_ids.float().unsqueeze(-1) * self.rope_inv_freq.to(device)
        angles = angles.repeat_interleave(2, dim=-1)
        cos = angles.cos().to(dtype=dtype).unsqueeze(1).unsqueeze(1)
        sin = angles.sin().to(dtype=dtype).unsqueeze(1).unsqueeze(1)
        return cos, sin

    def forward(self, x, batch_size=None, summary_size=0, visible_chan_ids=None,
                patch_ids=None):
        BmN, T, D = x.shape
        B = batch_size if batch_size is not None else int(BmN // self.num_patches)
        mC = T - summary_size

        x_channels = x[:, :mC, :]                                # (B*N, mC, D)
        x_summary  = x[:, mC:, :] if summary_size > 0 else None  # (B*N, summary_size, D)
        N = BmN // B
        x_channels = x_channels.reshape(B, N, mC, D).permute(0, 2, 1, 3)
        rope_cos, rope_sin = self._temporal_rope(
            patch_ids, B, N, x.device, x.dtype
        )

        if visible_chan_ids is None:
            assert mC == self.num_channels, \
                f"unmasked path requires mC ({mC}) == num_channels ({self.num_channels})"
            visible_chan_ids = torch.arange(self.num_channels, device=x.device).long()
            visible_chan_ids = visible_chan_ids.unsqueeze(0).expand(B, -1)  # (B, mC)
        else:
            visible_chan_ids = visible_chan_ids.to(x.device).long()
            assert visible_chan_ids.shape == (B, mC)

        qkv = self.qkv(x_channels)                                  # (B, mC, N, 3G)
        gamma, beta = self.channel_film[visible_chan_ids].chunk(2, dim=-1)
        qkv = qkv * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        qkv = qkv.reshape(B, mC, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(dim=0)

        q = q * rope_cos + rotate_half(q) * rope_sin
        k = k * rope_cos + rotate_half(k) * rope_sin
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.permute(0, 3, 1, 2, 4).reshape(B, N, mC, self.group_dim)
        out_ch = self.proj_drop(self.proj(out)).reshape(BmN, mC, D)

        if x_summary is not None:
            return torch.cat([out_ch, torch.zeros_like(x_summary)], dim=1)
        return out_ch


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False,
                 return_attention=False, use_gate=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope
        self.return_attention = return_attention
        self.is_causal = is_causal
        self.use_gate = use_gate

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gate_norm = nn.LayerNorm(dim) if self.use_gate else None
        self.gate_weight = nn.Linear(dim, num_heads, bias=False) if self.use_gate else None

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(diagonal=0)
                attn_mask = torch.full((T, T), -float('inf'), device=x.device).masked_fill(attn_mask, 0.)
            else:
                attn_mask = None
            attn_weight = torch.softmax(
                (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) + (attn_mask if self.is_causal else 0),
                dim=-1
            )
            attn_weight = torch.nn.functional.dropout(attn_weight, p=self.attn_drop, training=self.training)
            return attn_weight

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal
        )

        if self.use_gate:
            gate_input = self.gate_norm(x)
            gate_score = self.gate_weight(gate_input)
            gate_score = torch.sigmoid(gate_score)

            gate_score = gate_score.permute(0, 2, 1).unsqueeze(-1)

            y = y * gate_score

        x = y.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RelationAttention(Attention):
    """
    Patch-wise channel attention with dynamically routed relation experts.

    QKV and output projections are shared exactly as in standard attention.
    Relation experts only generate additive attention-logit biases from pairs
    of electrode coordinates. A patch-wise router mixes the top-k expert
    topologies, while an always-on shared relation provides a stable common
    path. Summary-token rows and columns receive zero relation bias.
    """

    def __init__(self, dim, num_heads=4, num_channels=58,
                 num_relation_experts=4, relation_top_k=2,
                 relation_hidden_dim=64, qkv_bias=False,
                 attn_drop=0., proj_drop=0., use_gate=True):
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop,
            use_rope=False, return_attention=False, use_gate=use_gate,
        )
        assert 0 < relation_top_k <= num_relation_experts
        self.num_channels = num_channels
        self.num_relation_experts = num_relation_experts
        self.relation_top_k = relation_top_k

        coords = get_3d_coordinates()
        assert coords.shape == (num_channels, 3)
        coords = coords / coords.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer('electrode_coords', coords, persistent=True)

        pair_dim = 5  # directed xyz delta, Euclidean distance, spherical dot product

        def make_relation_kernel():
            return nn.Sequential(
                nn.Linear(pair_dim, relation_hidden_dim),
                nn.GELU(),
                nn.Linear(relation_hidden_dim, num_heads),
            )

        self.shared_relation = make_relation_kernel()
        self.relation_experts = nn.ModuleList([
            make_relation_kernel() for _ in range(num_relation_experts)
        ])
        self.router = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 32), num_relation_experts),
        )
        # Keep the initial geometric contribution small without making the
        # experts identical or blocking their first-step gradients.
        self.relation_scale = nn.Parameter(torch.tensor(0.1))

        self.last_relation_importance = None
        self.last_relation_load = None

    def _pair_features(self, visible_chan_ids, batch_size, num_channels, device):
        if visible_chan_ids is None:
            assert num_channels == self.num_channels
            visible_chan_ids = torch.arange(
                self.num_channels, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            visible_chan_ids = visible_chan_ids.to(device=device, dtype=torch.long)
            assert visible_chan_ids.shape == (batch_size, num_channels)

        coords = self.electrode_coords.to(device)[visible_chan_ids]  # (B, C, 3)
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)             # (B, C, C, 3)
        distance = delta.square().sum(dim=-1, keepdim=True).sqrt()
        dot = (coords.unsqueeze(2) * coords.unsqueeze(1)).sum(
            dim=-1, keepdim=True
        )
        return torch.cat([delta, distance, dot], dim=-1)

    def _relation_bias(self, x_channels, batch_size, num_patches,
                       visible_chan_ids):
        BmN, mC, _ = x_channels.shape
        router_logits = self.router(x_channels.mean(dim=1))
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(
            router_probs, self.relation_top_k, dim=-1
        )
        top_weights = top_weights / top_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        sparse_gates = torch.zeros_like(router_probs)
        sparse_gates.scatter_(1, top_indices, top_weights)

        importance = router_probs.mean(dim=0)
        selected = torch.zeros_like(router_probs)
        selected.scatter_(1, top_indices, 1.0)
        load = selected.mean(dim=0)
        aux_loss = self.num_relation_experts * torch.sum(importance * load)

        # Kept detached for logging/inspection without retaining the graph.
        self.last_relation_importance = importance.detach()
        self.last_relation_load = load.detach()

        pair_features = self._pair_features(
            visible_chan_ids, batch_size, mC, x_channels.device
        )
        shared_bias = self.shared_relation(pair_features)  # (B, C, C, H)
        expert_bias = torch.stack([
            expert(pair_features) for expert in self.relation_experts
        ], dim=1)                                          # (B, K, C, C, H)

        gates = sparse_gates.reshape(
            batch_size, num_patches, self.num_relation_experts
        )
        mixed_bias = torch.einsum(
            'bnk,bkijh->bnhij', gates, expert_bias
        )
        shared_bias = shared_bias.permute(0, 3, 1, 2).unsqueeze(1)
        mixed_bias = mixed_bias + shared_bias
        mixed_bias = mixed_bias.reshape(
            BmN, self.num_heads, mC, mC
        )
        return self.relation_scale * mixed_bias, aux_loss

    def forward(self, x, freqs=None, batch_size=None, summary_size=0,
                visible_chan_ids=None):
        BmN, T, D = x.shape
        assert batch_size is not None
        B = batch_size
        assert BmN % B == 0
        N = BmN // B
        mC = T - summary_size
        assert mC > 0

        qkv = self.qkv(x).reshape(
            BmN, T, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        channel_bias, aux_loss = self._relation_bias(
            x[:, :mC, :], B, N, visible_chan_ids
        )
        if summary_size > 0:
            relation_bias = F.pad(
                channel_bias, (0, summary_size, 0, summary_size)
            )
        else:
            relation_bias = channel_bias
        relation_bias = relation_bias.to(dtype=q.dtype)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=relation_bias,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        if self.use_gate:
            gate_score = torch.sigmoid(self.gate_weight(self.gate_norm(x)))
            gate_score = gate_score.permute(0, 2, 1).unsqueeze(-1)
            y = y * gate_score

        y = y.transpose(1, 2).contiguous().view(BmN, T, D)
        y = self.proj_drop(self.proj(y))
        return y, aux_loss
