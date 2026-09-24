import torch.nn as nn
from .Layer.Block import PMoEBlock, Block
from .Layer.Embedding import PatchEmbed, RotaryEmbedding
from .Layer.AttentionLayer import CHANNEL_DICT
from .utils import *

# ── shared weight init helpers ──────────────────────────────────────────

def _rescale(param, layer_id):
    param.div_(math.sqrt(2.0 * layer_id))


def _init_weights(m, init_std=0.02):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        trunc_normal_(m.weight, std=init_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)


def _init_pmoe_weights(pmoe, layer_id):
    if hasattr(pmoe, 'gate'):
        nn.init.xavier_uniform_(pmoe.gate.weight)
        if pmoe.gate.bias is not None:
            nn.init.zeros_(pmoe.gate.bias)
    for attr in ('experts', 'shared'):
        if hasattr(pmoe, attr):
            items = getattr(pmoe, attr)
            if not isinstance(items, (list, nn.ModuleList)):
                items = [items]
            for expert in items:
                nn.init.kaiming_normal_(expert.fc1.weight, mode='fan_out', nonlinearity='relu')
                if expert.fc1.bias is not None:
                    nn.init.zeros_(expert.fc1.bias)
                _rescale(expert.fc2.weight.data, layer_id)
                if expert.fc2.bias is not None:
                    nn.init.zeros_(expert.fc2.bias)


# ── model components ────────────────────────────────────────────────────

class EEGTransformerReconstructor(nn.Module):

    def __init__(self, num_patches, patch_size=64, embed_num=1,
                 use_pos_embed=False, use_inp_embed=True,
                 embed_dim=768, reconstructor_embed_dim=384,
                 depth=6, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_std=0.02, interpolate_factor=2.,
                 return_attention_layer=-1, **kwargs):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches
        self.init_std = init_std

        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(embed_dim, reconstructor_embed_dim, bias=True)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        trunc_normal_(self.mask_token, std=init_std)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.time_embed_dim = (reconstructor_embed_dim // num_heads) // 2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), reconstructor_embed_dim)

        self.reconstructor_blocks = nn.ModuleList([
            Block(dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, use_rope=True,use_gate=False,
                  return_attention=(i + 1) == return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(reconstructor_embed_dim, patch_size, bias=True)

        self.apply(lambda m: _init_weights(m, init_std))
        self._fix_init_weight()

    def _fix_init_weight(self):
        for i, blk in enumerate(self.reconstructor_blocks):
            lid = i + 1
            _rescale(blk.time_attn.proj.weight.data, lid)
            _rescale(blk.mlp.fc2.weight.data, lid)

    def forward(self, x, chan_ids=None, mask_x=None, mask_y=None):
        chan_ids = chan_ids.to(x).long()

        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N = self.num_patches
        B, mN, eN, D = x.shape
        chan_embed = self.chan_embed(chan_ids).unsqueeze(0)  # (1,1,C,D)

        # ── RoPE freqs for context ──
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            if mask_x.ndim == 3:
                ctx_patch_ids = torch.div(mask_x[:, :, 0], C, rounding_mode='floor').long()
            else:
                ctx_patch_ids = torch.div(mask_x[:, 0], C, rounding_mode='floor').long()
                ctx_patch_ids = ctx_patch_ids.unsqueeze(0).expand(B, -1)
            base_freqs = self.time_embed.prepare_freqs((1, N), x.device, x.dtype)
            base_freqs = base_freqs.view(1, N, self.time_embed_dim).expand(B, -1, -1)
            gather_idx = ctx_patch_ids.unsqueeze(-1).expand(-1, -1, self.time_embed_dim)
            freqs_x = torch.gather(base_freqs, dim=1, index=gather_idx)
            freqs_x = freqs_x.unsqueeze(2).expand(-1, -1, eN, -1).flatten(1, 2)
        else:
            freqs_x = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype)
            freqs_x = freqs_x.unsqueeze(0).expand(B, -1, -1)

        # ── mask_y: reconstruction targets ──
        if mask_y is not None:
            mask_y = mask_y.to(x.device)
            if mask_y.ndim == 1:
                mask_y = mask_y.unsqueeze(0).expand(B, -1)
            N_y = mask_y.shape[1]
            chan_grid = chan_embed.expand(B, N, -1, -1)
            chan_embed_y = apply_mask(mask_y, chan_grid, batched=True)

            freqs = self.time_embed.prepare_freqs((C, N), x.device, x.dtype)
            freqs = freqs.view(1, N, C, self.time_embed_dim).expand(B, -1, -1, -1)
            freqs_y = apply_mask(mask_y, freqs, batched=True)

            y = self.mask_token.repeat(B, N_y, 1) + chan_embed_y

            if self.use_pos_embed:
                x = x + self.pos_embed.repeat(B, x.shape[1], 1, 1).to(x.device)

            x = torch.cat([x.flatten(1, 2), y], dim=1)
            freqs_x = torch.cat([freqs_x, freqs_y], dim=1).to(x).unsqueeze(1)

            for blk in self.reconstructor_blocks:
                x = blk(x, freqs_x)
                if blk.return_attention:
                    return x

            x = x[:, -N_y:, :]
            x = self.reconstructor_norm(x)
            x = self.reconstructor_proj(x)
            return x


class EEGTransformerPredictor(nn.Module):

    def __init__(self, num_patches, embed_dim=768, embed_num=1,
                 use_pos_embed=False, use_inp_embed=True, use_part_pred=False,
                 predictor_embed_dim=384, depth=6, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_std=0.02, interpolate_factor=2.,
                 return_attention_layer=-1, **kwargs):
        super().__init__()
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num
        self.init_std = init_std

        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
        trunc_normal_(self.mask_token, std=init_std)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.time_embed_dim = (predictor_embed_dim // num_heads) // 2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)

        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, use_rope=True,use_gate=False,
                  return_attention=(i + 1) == return_attention_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.apply(lambda m: _init_weights(m, init_std))
        self._fix_init_weight()

    def _fix_init_weight(self):
        for i, blk in enumerate(self.predictor_blocks):
            lid = i + 1
            _rescale(blk.time_attn.proj.weight.data, lid)
            _rescale(blk.mlp.fc2.weight.data, lid)

    def forward(self, x, mask_x=None, mask_t=None):
        if self.use_part_pred:
            inp_x = x
        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N = self.num_patches
        B, mN, eN, D = x.shape
        freqs = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype)

        if mask_x is not None:
            if mask_x.ndim == 3:
                patch_ids = torch.div(mask_x[:, :, 0], C, rounding_mode='floor').long()
                assert (patch_ids == patch_ids[0]).all(), \
                    "predictor currently requires batch-shared context patch ids"
                mask_x = patch_ids[0]
            else:
                mask_x = torch.floor(mask_x[:, 0] / C).long()
            if mask_t is None:
                mask_t = torch.tensor(list(set(range(N)) - set(mask_x.tolist()))).long()
            N_y = mask_t.shape[0]
            y = self.mask_token.repeat(B, N_y, 1, 1)
            x = torch.cat([x, y], dim=1)
            mask_id = torch.concat([mask_x.to(x.device), mask_t.to(x.device)], dim=0)
            x = torch.index_select(x, dim=1, index=torch.argsort(mask_id))

        if self.use_pos_embed:
            x = x + self.pos_embed.repeat(B, x.shape[1], 1, 1).to(x.device)

        B, N, eN, D = x.shape
        x = x.flatten(1, 2)
        for blk in self.predictor_blocks:
            x = blk(x, freqs)

        x = x.reshape(B, N, eN, D)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        if self.use_part_pred and mask_x is not None:
            cmb_x = torch.index_select(x, dim=1, index=mask_t.to(x.device))
            cmb_x = torch.concat([inp_x, cmb_x], dim=1)
            cmb_x = torch.index_select(cmb_x, dim=1, index=torch.argsort(mask_id))
            return x, cmb_x
        return x


class EEGTransformer(nn.Module):

    def __init__(self, img_size=(58, 1024), patch_size=64, patch_stride=None,
                 embed_dim=512, embed_num=4, depth=8, num_heads=8,
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, patch_module=PatchEmbed,
                 init_std=0.02, interpolate_factor=2.,
                 return_attention_layer=-1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.patch_size = patch_size
        self.init_std = init_std

        self.patch_embed = patch_module(img_size=img_size, patch_size=patch_size,
                                        patch_stride=patch_stride, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # self.conv_embed = ConvEmbed(img_size=img_size, embed_dim=embed_dim, patch_t=64)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = self._build_blocks(depth, embed_dim, num_heads, mlp_ratio,
                                         qkv_bias, drop_rate, attn_drop_rate,
                                         dpr, norm_layer)
        self.norm = norm_layer(embed_dim)

        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
        trunc_normal_(self.summary_token, std=init_std)
        self.apply(lambda m: _init_weights(m, init_std))
        self._fix_init_weight()

    def _build_blocks(self, depth, dim, num_heads, mlp_ratio, qkv_bias,
                      drop, attn_drop, dpr, norm_layer):

        # EXPERT_SCHEDULE = [0, 4, 4, 5, 5, 6, 6, 0]
        EXPERT_SCHEDULE = [0, 3, 3, 4, 4, 6, 6, 0]
        SHARED_EXPERT_LAYERS = [False, True, True, True, True, True, True, False]

        # NewVersion: all routed encoder blocks use the complete alternating
        # operator chain: Spatial Relation-MoE -> GroupAttention -> PMoE.
        GroupAttnLayers = [1,2,3,4,5,6]
        RelationAttnLayers = [1,2,3,4,5,6]

        num_patch = self.num_patches[1]


        blocks = nn.ModuleList()
        for i in range(depth):
            use_group = i in GroupAttnLayers
            use_relation = i in RelationAttnLayers
            num_exp = EXPERT_SCHEDULE[i]
            shared = SHARED_EXPERT_LAYERS[i]
            kw = dict(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                      drop_path=dpr[i], norm_layer=norm_layer)

            if num_exp > 0:
                blocks.append(PMoEBlock(
                    num_experts=num_exp, shared_expert=shared,
                    num_patches=num_patch,
                    use_relation_attn=use_relation,
                    relation_num_experts=4,
                    relation_top_k=2,
                    relation_aux_scale=0.25,
                    use_group_attn=use_group,
                    summary_size=self.embed_num,
                    **kw))
            else:
                blocks.append(Block(
                    num_patches=num_patch,
                    use_group_attn=use_group,
                    summary_size=self.embed_num,
                    **kw))

        return blocks

    def prepare_chan_ids(self, channels):
        return torch.tensor([CHANNEL_DICT[ch.upper().strip('.')]
                            for ch in channels]).unsqueeze_(0).long()

    def _fix_init_weight(self):
        for i, blk in enumerate(self.blocks):
            lid = i + 1
            _rescale(blk.time_attn.proj.weight.data, lid)
            if isinstance(blk, PMoEBlock):
                if blk.pmoe is not None:
                    _init_pmoe_weights(blk.pmoe, lid)
            else:
                _rescale(blk.mlp.fc2.weight.data, lid)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        total_aux_loss = 0.0
        weight_aux_loss = 0.0025

        # ── patch embed ──
        time_patch = self.patch_embed(x)                        # (B, N, C, D)

        B, N, C, D = time_patch.shape

        assert N == self.num_patches[1] and C == self.num_patches[0]

        if chan_ids is None:
            chan_ids = torch.arange(C, device=x.device)
        chan_ids = chan_ids.to(device=x.device, dtype=torch.long)
        if chan_ids.ndim == 1:
            chan_ids = chan_ids.unsqueeze(0)
        if chan_ids.ndim != 2 or chan_ids.shape[1] != C:
            raise ValueError(
                f"chan_ids must have shape (C,), (1, C), or (B, C); "
                f"got {tuple(chan_ids.shape)} for B={B}, C={C}"
            )
        if chan_ids.shape[0] == 1:
            batch_chan_ids = chan_ids.expand(B, -1)
        elif chan_ids.shape[0] == B:
            batch_chan_ids = chan_ids
        else:
            raise ValueError(
                f"chan_ids batch dimension must be 1 or B={B}; "
                f"got {chan_ids.shape[0]}"
            )

        # Keep canonical 58-channel ids for subset-channel downstream data.
        # Missing electrodes are not zero-filled: only the supplied channels
        # select their matching pretrained embeddings/coordinates/FiLM rows.
        x = time_patch + self.chan_embed(batch_chan_ids).unsqueeze(1)

        # ── 2D mask ──
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            if mask_x.ndim == 3:
                visible_patch_ids = torch.div(
                    mask_x[:, :, 0], self.num_patches[0], rounding_mode='floor'
                ).long()
                x = apply_mask(mask_x, x, batched=True)         # (B, mN, mC, D)
                local_visible_chan_ids = (
                    mask_x % self.num_patches[0]
                ).long()
            else:
                visible_patch_ids = torch.div(
                    mask_x[:, 0], self.num_patches[0], rounding_mode='floor'
                ).long().unsqueeze(0).expand(B, -1)
                x = apply_mask(mask_x, x)                       # (B, mN, mC, D)
                local_visible_chan_ids = (
                    mask_x % self.num_patches[0]
                ).long()
                local_visible_chan_ids = local_visible_chan_ids.unsqueeze(0).expand(
                    B, -1, -1
                )
            B, N, C, D = x.shape
            channel_lookup = batch_chan_ids.unsqueeze(1).expand(
                -1, local_visible_chan_ids.shape[1], -1
            )
            visible_chan_ids = torch.gather(
                channel_lookup, dim=2, index=local_visible_chan_ids
            )
        else:
            visible_chan_ids = batch_chan_ids
            visible_patch_ids = torch.arange(N, device=x.device).long()
            visible_patch_ids = visible_patch_ids.unsqueeze(0).expand(B, -1)

        x = x.flatten(0, 1)  # BmN, mC, D

        if mask_x is not None:
            grp_visible_chan_ids = visible_chan_ids[:, 0, :]  # (B, mC)
        else:
            grp_visible_chan_ids = visible_chan_ids

        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)  # BmN, mC+embed_num, D

        for i, blk in enumerate(self.blocks):
            if isinstance(blk, PMoEBlock):
                x, aux_loss = blk(x, batch_size=B,
                                  visible_chan_ids=grp_visible_chan_ids,
                                  patch_ids=visible_patch_ids)
                total_aux_loss += weight_aux_loss * aux_loss
            else:
                x = blk(x, batch_size=B,
                        visible_chan_ids=grp_visible_chan_ids,
                        patch_ids=visible_patch_ids)

        x = self.norm(x)

        x = x[:, -summary_token.shape[1]:, :]

        x = x.flatten(-2)
        x = x.reshape((B, N, -1))

        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)

        x = x.reshape((B, N, self.embed_num, -1))

        return x, total_aux_loss
