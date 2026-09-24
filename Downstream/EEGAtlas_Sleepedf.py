import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import pytorch_lightning as pl

import torchvision
from functools import partial
import numpy as np
import random
import os
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F

import os
import torch
from torchvision.datasets import DatasetFolder


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


from utils import temporal_interpolation
from utils_eval import get_metrics
from Modules.Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from Pretraining.EEGAtlas import *

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint

use_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="..\..\Pretraining\Result\Large\Model\EEGAtlas_large.ckpt"):
        super().__init__()
        self.chans_num = len(use_channels_names)
        self.num_class = 5
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 256 * 30],
            patch_size=32 * 2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.target_encoder = target_encoder
        self.chans_id = target_encoder.prepare_chan_ids(use_channels_names)

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)

        self.chan_conv = Conv1dWithConstraint(2, self.chans_num, 1, max_norm=1)

        self.linear_probe1 = LinearWithConstraint(2048, 64, max_norm=1)
        self.drop = torch.nn.Dropout(p=0.50)
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(64, 4, 64 * 4, activation=torch.nn.functional.gelu,
                                                           batch_first=False),
            num_layers=4
        )
        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, 64) * 0.001, requires_grad=True)
        self.linear_probe2 = LinearWithConstraint(64, self.num_class, max_norm=0.25)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):
        B, C, T = x.shape
        x = temporal_interpolation(x, 256 * 30)
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z, _ = self.target_encoder(x, self.chans_id.to(x))

        h = z.flatten(2)

        h = self.linear_probe1(self.drop(h))
        pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)

        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h], dim=1)
        h = h.transpose(0, 1)
        h = self.decoder(h, h)[0, :, :]

        h = self.linear_probe2(h)
        return x, h

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity = False
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x, y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)

        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)

        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()) +
            [self.cls_token] +
            list(self.decoder.parameters()),
            weight_decay=0.02)  #

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler,  # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1,  # The frequency of the scheduler
            'monitor': 'val_loss',  # Metric for `ReduceLROnPlateau` to monitor
            'strict': True,  # Whether to crash the training if `monitor` is not found
            'name': None,  # Custom name for `LearningRateMonitor` to use
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


class SubjectFilteredSleepEDF(DatasetFolder):
    def __init__(self, root, subject_ids, loader=torch.load, extensions=("pt",)):
        self.subject_ids = subject_ids
        super().__init__(root, loader, extensions)

    def _find_classes(self, dir):
        return super()._find_classes(dir)

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
        samples = super().make_dataset(
            directory, class_to_idx, extensions=extensions,
            is_valid_file=is_valid_file, allow_empty=allow_empty
        )
        filtered_samples = []
        for path, target in samples:
            file_name = os.path.basename(path)
            if any(file_name.startswith(f"s{sid}_") for sid in self.subject_ids):
                filtered_samples.append((path, target))

        if not filtered_samples:
            raise FileNotFoundError(f"未找到匹配被试ID {self.subject_ids} 的样本！")
        return filtered_samples


def pt_loader(path):
    return torch.load(path, weights_only=False)


# load configs
subjects = [40, 42, 45, 46, 47, 48, 49, 52]
n_folds = 4
val_per_fold = 2

if __name__ == "__main__":

    for seed in [7,42,718]:
        for fold in range(n_folds):
            start = fold * val_per_fold
            set_valid = set(subjects[start : start + val_per_fold])
            set_train = set(subjects) - set_valid

            print(f"\n===== LOSO - Test Fold:{fold} =====")
            print(f"Train Subject:{sorted(list(set_train))}")

            train_dataset = SubjectFilteredSleepEDF(
                root="SleepEDF\\sleep_edf\\",
                subject_ids=set_train,
                loader=pt_loader,
                extensions=("pt",)
            )

            valid_dataset = SubjectFilteredSleepEDF(
                root="SleepEDF\\sleep_edf\\",
                subject_ids=set_valid,
                loader=pt_loader,
                extensions=("pt",)
            )
            # -- begin Training ------------------------------

            import math

            torch.set_float32_matmul_precision('medium')

            global max_epochs
            global steps_per_epoch
            global max_lr

            batch_size = 12

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,persistent_workers=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=2, shuffle=False,persistent_workers=True)

            max_epochs = 20
            steps_per_epoch = math.ceil(len(train_loader))
            max_lr = 4e-4

            model = LitEEGPTCausal()
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

            checkpoint_callback = ModelCheckpoint(
                dirpath=f'./logs/EEGAtlas/Seed{seed}/Fold{fold} Model',
                filename=f'EEGAtlas_Fold{fold}',
                monitor='valid_balanced_accuracy',
                mode='max',
                save_top_k=1,
                every_n_epochs=1,
                save_weights_only=False,
                save_last=False
            )
            callbacks = [lr_monitor, checkpoint_callback]

            trainer = pl.Trainer(accelerator='cuda',
                                 precision="16-mixed",
                                 max_epochs=max_epochs,
                                 callbacks=callbacks,
                                 logger=[pl_loggers.TensorBoardLogger(f'./logs/EEGAtlas/Seed{seed}', name="EEGAtlas_SLEEPEDF_tb",
                                                                      version=f"Fold{fold}"),
                                         pl_loggers.CSVLogger(f'./logs/EEGAtlas/Seed{seed}', name="EEGAtlas_SLEEPEDF_csv",
                                                              version=f"Fold{fold}")])

            trainer.fit(model, train_loader, valid_loader)