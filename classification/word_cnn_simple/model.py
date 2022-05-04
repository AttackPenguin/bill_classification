import os

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl

# define the LightningModule
from classification.word_cnn_simple.dataloader import BillDataModule


class WordCNNSimple(pl.LightningModule):
    def __init__(
            self,
            embedding_size: int = 300,
            pooling_size: int = 4,
            input_length: int = 8192,  # 2^13
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.pooling_size = pooling_size
        self.input_length = input_length

        # Default input dimension is (batch_size, 300, 8192)
        self.conv_l1 = nn.Conv1d(
            in_channels=embedding_size, out_channels=256,
            kernel_size=1, stride=1, padding=0
        )
        self.pool_l1 = nn.MaxPool1d(
            kernel_size=8,
            stride=8,
            padding=0
        )

        # Input dimension is (batch_size, 16, 256)
        self.linear_l5 = nn.Linear(
            in_features=256*1024,
            out_features=256
        )
        self.linear_l6 = nn.Linear(
            in_features=256,
            out_features=256
        )

        self.output = nn.Linear(
            in_features=256,
            out_features=1
        )

    def forward(self, x):
        # x dimensions = (batch_size, input_length, input_size)
        x = x.permute(0, 2, 1)

        # Apply Convolutions
        l1_convolved = F.leaky_relu(self.conv_l1(x.float()))
        l1_pooled = self.pool_l1(l1_convolved)

        # Pass through dense layers
        l4_linearized = l1_pooled.reshape(-1, 256*1024)
        l5_out = F.leaky_relu(self.linear_l5(l4_linearized))
        l6_out = F.leaky_relu(self.linear_l6(l5_out))

        # Get single output value in range [0, 1]
        return torch.sigmoid(self.output(l6_out)).flatten()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log(
            "train_loss", loss,
            sync_dist=True, on_epoch=True,
            prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log(
            "val_loss", loss,
            sync_dist=True, on_epoch=True,
            prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log(
            "test_loss", loss,
            sync_dist=True,
            prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = WordCNNSimple()
print(model)

datamodule = BillDataModule(batch_size=24)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=-1,
    filename='epoch {epoch:02d} val_loss {val_loss:.3f}'
)


if torch.cuda.is_available():
    devices = -1
    accelerator = 'gpu'
    strategy = 'dp'
else:
    devices = 1
    accelerator = 'cpu'
    strategy = None

trainer = Trainer(
    devices=devices, accelerator=accelerator, strategy=strategy,
    callbacks=[checkpoint_callback]
)

trainer.fit(model, datamodule)
