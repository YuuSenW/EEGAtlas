import copy
from functools import partial
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch import nn

from configs import *
from EEGAtlas import EEGTransformer, EEGTransformerPredictor, EEGTransformerReconstructor
from utils import CosineWDSchedule, grad_logger, mask_2d_to_1d_ctx, mask_2d_to_1d_recon, apply_mask


use_channels_names = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2',
]



class LitEEGPT(pl.LightningModule):

    def __init__(self, models_configs):
        super().__init__()

        encoder = EEGTransformer(
            img_size=[58, 256*4],
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['encoder'])

        predictor = EEGTransformerPredictor(
            num_patches=encoder.num_patches,
            use_part_pred=True,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['predictor'])

        reconstructor = EEGTransformerReconstructor(
            num_patches=encoder.num_patches,
            patch_size=32*2,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **models_configs['reconstructor'])

        target_encoder = copy.deepcopy(encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False

        self.encoder        = encoder
        self.target_encoder = target_encoder
        self.predictor      = predictor
        self.reconstructor  = reconstructor
        self.chans_id       = encoder.prepare_chan_ids(use_channels_names)

        self.loss_fn = torch.nn.MSELoss()
        self.train_steps = 0


    def make_masks(self, num_patchs, batch_size, mC_x=29, p_n_y=0.5, p_c_y=0.2):
        """
        Batched 2D masks with per-sample channel sets.

        N 维: 当前 batch 共享 ctx / target-only patch ids，保持 predictor 的规则时序布局。
        C 维: 每个 sample 独立抽取 mC_x 个 channel；同一样本的所有 ctx patches
              共享该组 channel，保证每个通道拥有完整的跨 patch 时间序列。

        target-only patch: 整行进 mask_recon.
        ctx patch 的 (C - mC_x) 个 hidden channels 进 recon pool, 按 p_c_y 概率二次抽样后进 mask_recon.

        Args:
            num_patchs: (C, N) = (58, 16)
            mC_x: visible channels per ctx-patch (default 29)
            p_n_y: probability a patch is target-only (default 0.5)
            p_c_y: fraction of recon pool kept in mask_recon (default 0.2)

        Returns:
            mask_ctx:   (B, N, C) bool — True = context encoder can see
            mask_recon: (B, N, C) bool — True = needs reconstruction
        """
        C, N = num_patchs

        assert 0 < mC_x < C
        while True:
            is_ctx_n = torch.rand(N) >= p_n_y
            if is_ctx_n.any() and (~is_ctx_n).any():
                break

        ctx_pids = is_ctx_n.nonzero(as_tuple=False).squeeze(-1)
        tgt_pids = (~is_ctx_n).nonzero(as_tuple=False).squeeze(-1)
        mask_ctx = torch.zeros(batch_size, N, C, dtype=torch.bool)
        mask_recon = torch.zeros(batch_size, N, C, dtype=torch.bool)
        mask_recon[:, tgt_pids, :] = True

        hidden_pool_size = ctx_pids.numel() * (C - mC_x)
        num_hidden_targets = int(round(hidden_pool_size * p_c_y))

        for b in range(batch_size):
            perm = torch.randperm(C)
            visible_cids = perm[:mC_x]
            hidden_cids = perm[mC_x:]
            mask_ctx[b, ctx_pids.unsqueeze(1), visible_cids.unsqueeze(0)] = True

            if num_hidden_targets > 0:
                pool_pids = ctx_pids.unsqueeze(1).expand(-1, hidden_cids.numel()).reshape(-1)
                pool_cids = hidden_cids.unsqueeze(0).expand(ctx_pids.numel(), -1).reshape(-1)
                selected = torch.randperm(hidden_pool_size)[:num_hidden_targets]
                mask_recon[b, pool_pids[selected], pool_cids[selected]] = True

        return mask_ctx, mask_recon

    def forward_target(self, x, mask_recon):
        """Target encoder path (EMA, no_grad). mask_recon: (B, N, C) bool."""
        with torch.no_grad():
            h, aux_loss = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim

            C, N = self.encoder.num_patches
            assert x.shape[-1] % N == 0 and x.shape[-2] % C == 0
            block_size_c, block_size_n = x.shape[-2] // C, x.shape[-1] // N
            # 重建目标与编码器输入同域
            x_r = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            x_r = x_r.permute(0, 3, 1, 2, 4).contiguous()  # B, N, C, bc, bn
            x_r = x_r.view(x_r.shape[0], N, C, block_size_c * block_size_n)

            # Convert 2D to 1D for apply_mask (exact MoEEG equivalence)
            mask_y = mask_2d_to_1d_recon(mask_recon)
            y = apply_mask(mask_y.to(x.device), x_r, batched=True)
            y = F.layer_norm(y, (y.size(-1),))

            return h, y, aux_loss

    def forward_context(self, x, mask_ctx, mask_recon):
        """Context encoder path. Both masks are (B, N, C) bool."""
        mask_x = mask_2d_to_1d_ctx(mask_ctx)
        mask_y = mask_2d_to_1d_recon(mask_recon)

        z, aux_loss = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)

        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r, aux_loss

    def relation_router_stats(self):
        """Mean Relation-MoE importance/load across enabled encoder layers."""
        importance, load = [], []
        for block in self.encoder.blocks:
            attn = getattr(block, 'time_attn', None)
            if (attn is not None
                    and getattr(attn, 'last_relation_importance', None) is not None):
                importance.append(attn.last_relation_importance)
                load.append(attn.last_relation_load)
        if not importance:
            return None, None
        return torch.stack(importance).mean(dim=0), torch.stack(load).mean(dim=0)

    def log_relation_router_stats(self, prefix):
        importance, load = self.relation_router_stats()
        if importance is None:
            return
        for expert_idx in range(importance.numel()):
            self.log(f'{prefix}_relation_importance_{expert_idx}',
                     importance[expert_idx], on_epoch=True, on_step=False,
                     sync_dist=True)
            self.log(f'{prefix}_relation_load_{expert_idx}',
                     load[expert_idx], on_epoch=True, on_step=False,
                     sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        mask_ctx, mask_recon = self.make_masks(self.encoder.num_patches, x.shape[0])
        h, y, aux_loss = self.forward_target(x, mask_recon)
        z, r, aux_loss = self.forward_context(x, mask_ctx, mask_recon)

        loss1 = self.loss_fn(h, z)
        loss2 = 2 * self.loss_fn(y, r)

        loss = loss1 + loss2 + aux_loss

        self.log('valid_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_aux_loss', aux_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
        self.log_relation_router_stats('valid')

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        self.train_steps += 1

        mask_ctx, mask_recon = self.make_masks(self.encoder.num_patches, x.shape[0])
        h, y, aux_loss = self.forward_target(x, mask_recon)
        z, r, aux_loss = self.forward_context(x, mask_ctx, mask_recon)
        loss1 = self.loss_fn(h, z)
        loss2 = 2 * self.loss_fn(y, r)

        loss = loss1 + loss2 + aux_loss

        if self.train_steps % 100 == 0:
            with torch.no_grad():
                ctx_mask = mask_ctx[0].any(dim=1)
                ctx_pids = ctx_mask.nonzero(as_tuple=False).squeeze(-1)
                tgt_pids = (~ctx_mask).nonzero(as_tuple=False).squeeze(-1)
            print(f"Step {self.train_steps}: Loss={loss.item():.4f} | "
                  f"Align={loss1.item():.5f}  | "
                  f"Recon={loss2.item():.4f} | Aux={aux_loss.item():.6f} | "
                  f"N_ctx={len(ctx_pids)} N_tgt={len(tgt_pids)}")

        self.log('train_loss1', loss1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss2', loss2, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_loss' , loss , on_epoch=True, on_step=False, sync_dist=True)
        self.log_relation_router_stats('train')

        return loss


    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log('grad_stats.first_layer', grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.last_layer', grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.min', grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log('grad_stats.max', grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)

        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return super().on_train_batch_end(outputs, batch, batch_idx)


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)

        self.configure_optimizers()
        return res

    def configure_optimizers(self):

        param_groups = [
            {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.reconstructor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=6e-5)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs,
                                                           div_factor = 2,
                                                           final_div_factor = 8,
                                                           pct_start = 0.2,
                                                           )
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'valid_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
        self.wd_scheduler = CosineWDSchedule(
                            optimizer,
                            ref_wd=1e-6,
                            final_wd=1e-6,
                            T_max=int(max_epochs*steps_per_epoch))
        ema = [0.96,1.0]
        self.momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(steps_per_epoch*max_epochs)
                          for i in range(int(steps_per_epoch*max_epochs)+1))
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
