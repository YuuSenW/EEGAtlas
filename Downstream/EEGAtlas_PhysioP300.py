import pytorch_lightning as pl
from functools import partial
import os
from pathlib import Path

import torch.nn
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *
from Pretraining.EEGAtlas import *
from Modules.Network.utils import LinearWithConstraint
from sklearn import metrics
from utils_eval import get_metrics

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.set_float32_matmul_precision('medium')

ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
            'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
            'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
            'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
ch_names = [x.upper() for x in ch_names]

use_channels_names1 = ['FP1', 'FPZ', 'FP2',
                       'AF3', 'AF4',
                       'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                       'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                       'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                       'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                       'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                       'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                       'O1', 'OZ', 'O2', ]
use_channels_names = []
channels_index = []
for x in use_channels_names1:
    if x in ch_names:
        channels_index.append(ch_names.index(x))
        use_channels_names.append(x)


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path=None):
        super().__init__()
        if load_path is None:
            load_path = os.environ.get(
                "EEGATLAS_PRETRAIN_CKPT",
                str(Path(__file__).resolve().parents[2] / "Pretraining" /
                    "Result" / "Large" / "Model" / "EEGAtlas_large.ckpt"),
            )
        self.chans_num = len(use_channels_names)

        self.embed_num = 4
        self.embed_dim = 512
        self.depth = 8
        self.linear1_out_dim = 16

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 512],
            patch_size=32 * 2,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        self.target_encoder = target_encoder

        self.chan_scale = torch.nn.Parameter(
            torch.ones(1, self.chans_num, 1) + 0.001 * torch.rand((1, self.chans_num, 1)),
            requires_grad=True)

        self.N = self.target_encoder.patch_embed.num_patches[1]

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path, weights_only=False)

        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v

        self.target_encoder.load_state_dict(target_encoder_stat)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.linear_probe1 = LinearWithConstraint(2048, self.linear1_out_dim, max_norm=1.0)
        self.linear_probe2 = LinearWithConstraint(self.linear1_out_dim * self.N, 2, max_norm=0.25)

        self.drop = torch.nn.Dropout(p=0.5)
        self.GeLU = torch.nn.GELU()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):
        B, C, T = x.shape

        x = x.to(torch.float)

        x = x - x.mean(dim=-2, keepdim=True)

        x = x[:, channels_index, :512]
        x = x * self.chan_scale

        z, _ = self.target_encoder(x)

        h = z.flatten(2)

        h = self.linear_probe1(self.GeLU(self.drop(h)))

        h = self.GeLU(h.flatten(1))

        h = self.linear_probe2(h)

        return h

    def on_train_epoch_start(self) -> None:
        self.running_scores["train"] = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x, y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_train_epoch_end()

    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()

        logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))

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

        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)

        for key, value in results.items():
            if key == "balanced_accuracy":
                print(f"\nBalanced Accuracy:{value}")
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        label = y.long()

        logit = self.forward(x)

        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds == label) * 1.0).mean()

        loss = self.loss_fn(logit, label)
        y_score = logit
        y_score = torch.softmax(y_score, dim=-1)[:, 1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            [self.chan_scale] +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.03)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                           epochs=max_epochs, pct_start=0.1)
        lr_dict = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
            'name': None,
        }

        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )


import torchvision
import math

global max_epochs
global steps_per_epoch
global max_lr

batch_size = 24
max_epochs = 100

all_subjects = [1, 2, 3, 4, 5, 6, 7, 9, 11,]

if __name__ == "__main__":
    for seed in [7,42,718]:
        seed_torch(seed)
        selected_seed = os.environ.get("EEGATLAS_SEED")
        seeds = [int(selected_seed)] if selected_seed is not None else [ 7, 718, 815]
        # seeds = [int(selected_seed)] if selected_seed is not None else [42, 7, 718, 815]
        for seed in seeds:
            seed_torch(seed)
            log_root = Path(__file__).resolve().parent / "logs" / "EEGAtlas" / f"Seed{seed}"
            run_id = os.environ.get("EEGATLAS_RUN_ID")
            if run_id:
                log_root = log_root / f"Run {run_id}"
            for i, sub in enumerate(all_subjects):

                sub_train = [f".sub{x}" for x in all_subjects if x != sub]
                sub_valid = [f".sub{sub}"]
                print(sub_train, sub_valid)
                train_dataset = torchvision.datasets.DatasetFolder(root="PhysioNetP300", loader=torch.load,
                                                                   extensions=sub_train)
                valid_dataset = torchvision.datasets.DatasetFolder(root="PhysioNetP300", loader=torch.load,
                                                                   extensions=sub_valid)

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=6,
                                                           shuffle=True, persistent_workers=True)
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=2,
                                                           shuffle=False, persistent_workers=True)

                steps_per_epoch = math.ceil(len(train_loader))

                model = LitEEGPTCausal()

                lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

                checkpoint_callback = ModelCheckpoint(
                    dirpath=f'./logs/EEGAtlas/Seed{seed}/Subject{sub} Model',
                    filename=f'EEGAtlas_Sub{sub}',
                    monitor='valid_loss',
                    mode='min',
                    save_top_k=1,
                    every_n_epochs=1,
                    save_weights_only=False,
                    save_last=False
                )

                callbacks = [lr_monitor, checkpoint_callback]
                max_lr = 8e-4
                trainer = pl.Trainer(accelerator='cuda',
                                     max_epochs=max_epochs,
                                     callbacks=callbacks,
                                     enable_checkpointing=True,
                                     logger=[pl_loggers.TensorBoardLogger(f'./logs/EEGAtlas/Seed{seed}', name=f"EEGAtlas_PhysioP300_tb",
                                                                          version=f"subject{sub}"),
                                             pl_loggers.CSVLogger(f'./logs/EEGAtlas/Seed{seed}', name=f"EEGAtlas_PhysioP300_csv",
                                                                  version=f"subject{sub}")])

                trainer.fit(model, train_loader, valid_loader)
