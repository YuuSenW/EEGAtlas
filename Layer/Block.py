import torch
import torch.nn as nn
from .MoE import ProgressiveMoE
from .AttentionLayer import Attention, GroupAttention, RelationAttention

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Standard Transformer block: TimeAttn + [GroupAttn] + MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., num_patches=16, act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rope=False,
                 return_attention=False, use_gate=True,
                 use_group_attn=False, summary_size=0):
        super().__init__()
        self.return_attention = return_attention
        self.use_group_attn = use_group_attn
        self.num_patches = num_patches
        self.summary_size = summary_size
        self.norm1 = norm_layer(dim, eps=1e-6)

        self.time_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_rope=use_rope, return_attention=return_attention, use_gate=use_gate)

        if self.use_group_attn:
            self.norm2g = norm_layer(dim, eps=1e-6)
            self.group_attn = GroupAttention(
                dim, num_heads=num_heads, num_patches=num_patches,
                num_channels=58,
                group_dim=dim,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

            self.grp_gate = nn.Parameter(torch.ones(1))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None, batch_size=None, visible_chan_ids=None,
                patch_ids=None):
        B = batch_size
        residual = x
        x_norm = self.norm1(x)

        timeattn_out = self.time_attn(x_norm, freqs)
        x = x + self.drop_path(timeattn_out)

        # GroupAttn — from raw x (parallel path)
        if self.use_group_attn:
            y_grp = self.group_attn(self.norm2g(residual), batch_size=B,
                                    summary_size=self.summary_size,
                                    visible_chan_ids=visible_chan_ids,
                                    patch_ids=patch_ids)
            y_grp = y_grp * self.grp_gate
            x = x + self.drop_path(y_grp)

        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class PMoEBlock(nn.Module):
    """Alternating-operator block with RelationAttn, GroupAttn and PMoE.

    The three operators are applied sequentially:
      Spatial Relation-MoE -> Temporal GroupAttention -> PMoE.

    Each operator reads the state updated by the preceding operator. This is
    intentionally different from the legacy parallel Relation/Group branches.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., num_experts=0, shared_expert=False, top_k=2,
                 num_patches=16,
                 use_relation_attn=False, relation_num_experts=4,
                 relation_top_k=2, relation_aux_scale=0.25,
                 use_group_attn=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rope=False,
                 return_attention=False,
                 summary_size=4):
        super().__init__()
        self.return_attention = return_attention
        self.num_patches = num_patches
        self.use_group_attn = use_group_attn
        self.use_relation_attn = use_relation_attn
        self.relation_aux_scale = relation_aux_scale
        self.summary_size = summary_size

        # ── TimeAttn ──
        self.norm1 = norm_layer(dim, eps=1e-6)
        if self.use_relation_attn:
            self.time_attn = RelationAttention(
                dim, num_heads=num_heads, num_channels=58,
                num_relation_experts=relation_num_experts,
                relation_top_k=relation_top_k,
                qkv_bias=qkv_bias, attn_drop=attn_drop,
                proj_drop=drop, use_gate=True)
        else:
            self.time_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop,
                use_rope=use_rope, return_attention=return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ── GroupAttn ──
        if self.use_group_attn:
            self.norm2g = norm_layer(dim, eps=1e-6)
            self.group_attn = GroupAttention(
                dim, num_heads=num_heads, num_patches=num_patches,
                num_channels=58,
                group_dim=dim,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # ── PMoE ──
        self.norm3 = norm_layer(dim, eps=1e-6)
        self.pmoe = ProgressiveMoE(
                dim=dim, num_experts=num_experts, shared_expert=shared_expert,
                top_k=top_k, mlp_ratio=mlp_ratio, drop=drop)


    def forward(self, x, freqs=None, batch_size=None, visible_chan_ids=None,
                patch_ids=None):
        B = batch_size

        # S: patch-wise spatial relation operator.
        x_norm = self.norm1(x)
        if self.use_relation_attn:
            timeattn_out, relation_aux_loss = self.time_attn(
                x_norm, batch_size=B, summary_size=self.summary_size,
                visible_chan_ids=visible_chan_ids)
        else:
            timeattn_out = self.time_attn(x_norm, freqs)
            relation_aux_loss = x.new_zeros(())
        x = x + self.drop_path(timeattn_out)

        # T: cross-patch temporal operator. It consumes the spatially updated
        # state above rather than the raw block residual.
        if self.use_group_attn:
            y_grp = self.group_attn(self.norm2g(x), batch_size=B,
                                    summary_size=self.summary_size,
                                    visible_chan_ids=visible_chan_ids,
                                    patch_ids=patch_ids)
            x = x + self.drop_path(y_grp)

        # F: token-wise conditional feature transformation.
        moe_out, aux_loss = self.pmoe(self.norm3(x))
        x = x + self.drop_path(moe_out)
        return x, aux_loss + self.relation_aux_scale * relation_aux_loss
