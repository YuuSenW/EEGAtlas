""""""
import math
import random
import os
import torch
from torch import nn
import pytorch_lightning as pl
from functools import partial
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def pad_last_dim(x):
    current_last_dim = x.shape[2]
    if current_last_dim >= 1024:
        return x

    pad_length = 1024 - current_last_dim
    x_padded = F.pad(x, (0, pad_length), mode='constant', value=0)
    return x_padded

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from Pretraining.EEGAtlas import *
from Modules.Network.utils import LinearWithConstraint
from utils_eval import get_metrics

use_channels_names = [
    'FZ', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'P1', 'PZ', 'P2', 'POZ',
]

class LitEEGPTCausal(pl.LightningModule):
    def __init__(self, load_path="..\..\Pretraining\Result\Large\Model\EEGAtlas_large.ckpt"):
        super().__init__()
        self.chans_num = len(use_channels_names)

        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 1024],
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

        pretrain_ckpt = torch.load(load_path,weights_only=False)
        target_encoder_stat = {}
        for k, v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]] = v
        self.target_encoder.load_state_dict(target_encoder_stat)

        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(16 * 16, 4, max_norm=0.25)
        self.drop = torch.nn.Dropout(p=0.50)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.running_scores = {"train": [], "valid": [], "test": []}
        self.is_sanity = True

    def forward(self, x):
        x = x.to(torch.float)
        x = pad_last_dim(x)

        if x.shape[1] != self.chans_num:
            raise ValueError(f"Expected {self.chans_num} BCIC-2A EEG channels, got {x.shape[1]}")
        self.target_encoder.eval()
        z,_ = self.target_encoder(x, self.chans_id.to(x.device))
        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        h = h.flatten(1)
        h = self.linear_probe2(h)
        return x, h

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = F.one_hot(y.long(), num_classes=4).float()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        accuracy = ((torch.argmax(logit, dim=-1) == torch.argmax(y, dim=-1)) * 1.0).mean()
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False)
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

        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        for key, value in results.items():
            self.log('valid_' + key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        accuracy = ((torch.argmax(logit, dim=-1) == label) * 1.0).mean()
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False)
        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            pct_start=0.2
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}



class EEGDataset(Dataset):
    def __init__(self, root_dir, subject_ids):
        self.data = []
        self.labels = []

        for sub_id in subject_ids:
            sub_dir = os.path.join(root_dir, f"sub{sub_id}")
            data = torch.load(os.path.join(sub_dir, "data.pt"))  # 形状: [样本数, 通道数, 时间点]
            labels = torch.load(os.path.join(sub_dir, "label.pt"))  # 形状: [样本数]
            self.data.append(data)
            self.labels.append(labels)
        # 合并所有被试的数据
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 主实验流程（LOSO）
if __name__ == "__main__":
    # 配置参数
    batch_size = 24
    max_epochs = 100
    max_lr = 8e-4
    data_root = "BCIC_2aT_0_38HZ"
    all_subjects = [1,2,3,4,5,6,7,8,9]

    for seed in [7,718,42]:
        seed_torch(seed)
        for test_sub in all_subjects:
            train_subjects = [sub for sub in all_subjects if sub != test_sub]
            print(f"\nTest: sub{test_sub}, Train: sub {train_subjects}")

            train_dataset = EEGDataset(data_root, train_subjects)
            test_dataset = EEGDataset(data_root, [test_sub])

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=6,
                persistent_workers=True

            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                persistent_workers=True
            )

            steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)

            model = LitEEGPTCausal()
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            logger = [
                pl_loggers.TensorBoardLogger(f'./logs/EEGAtlas/Seed{seed}', name="LOSO_BCIC2a_tb", version=f"test_sub{test_sub}"),
                pl_loggers.CSVLogger(f'./logs/EEGAtlas/Seed{seed}', name="LOSO_BCIC2a_csv", version=f"test_sub{test_sub}")
            ]

            trainer = pl.Trainer(
                accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                max_epochs=max_epochs,
                callbacks=[lr_monitor],
                logger=logger,
            )

            trainer.fit(model, train_loader, test_loader)
