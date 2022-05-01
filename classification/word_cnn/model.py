import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl

# define the LightningModule
from classification.word_cnn.dataloader import BillDataModule


class WordCNN(pl.LightningModule):
    def __init__(
            self,
            embedding_size: int = 300,
            pooling_size: int = 4,
            input_length: int = 16_384,  # 2^14
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.pooling_size = pooling_size
        self.input_length = input_length

        # Default input dimension is (batch_size, 300, 16_384)
        self.conv_l1 = nn.Conv1d(
            in_channels=embedding_size, out_channels=64,
            kernel_size=1, stride=1, padding=0
        )
        # Default input dimension is (batch_size, 64, 16_384)
        self.conv_l2 = nn.Conv1d(
            in_channels=64, out_channels=32,
            kernel_size=4, stride=4, padding=0
        )
        # Input dimension is (batch_size, 32, 4096)
        self.conv_l3 = nn.Conv1d(
            in_channels=32, out_channels=16,
            kernel_size=4, stride=4, padding=0
        )
        # Input dimension is (batch_size, 16, 1024)
        self.conv_l4 = nn.Conv1d(
            in_channels=16, out_channels=16,
            kernel_size=4, stride=4, padding=0
        )

        # Input dimension is (batch_size, 16, 256)
        self.linear_l5 = nn.Linear(
            in_features=16 * 256,
            out_features=512
        )
        self.linear_l6 = nn.Linear(
            in_features=512,
            out_features=512
        )

        self.output = nn.Linear(
            in_features=512,
            out_features=1
        )

    def forward(self, x):
        # x dimensions = (batch_size, input_length, input_size)
        x = x.permute(0, 2, 1)

        # Apply Convolutions
        l1_convolved = F.leaky_relu(self.conv_l1(x.float()))
        l2_convolved = F.leaky_relu(self.conv_l2(l1_convolved))
        l3_convolved = F.leaky_relu(self.conv_l3(l2_convolved))
        l4_convolved = F.leaky_relu(self.conv_l4(l3_convolved))

        # Pass through dense layers
        l4_linearized = l4_convolved.reshape(-1, 16*256)
        l5_out = F.leaky_relu(self.linear_l5(l4_linearized))
        l6_out = F.leaky_relu(self.linear_l6(l5_out))

        # Get single output value in range [0, 1]
        return torch.sigmoid(self.output(l6_out)).flatten()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = WordCNN()
print(model)

datamodule = BillDataModule(batch_size=128)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=3
)

if torch.cuda.is_available():
    trainer = Trainer(
        devices=-1, accelerator="gpu", strategy="dp",
        callbacks=[checkpoint_callback]
    )
else:
    trainer = Trainer(
        callbacks=[checkpoint_callback]
    )
trainer.fit(model, datamodule)
