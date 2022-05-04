import os

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl

# define the LightningModule
from classification.gru_cnn.dataloader import BillDataModule


class GRU_CNN(pl.LightningModule):
    def __init__(
            self,
            kernel_length: int = 8,
            embedding_size: int = 300,
            pooling_size: int = 4,
            input_length: int = 65_536,  # 2^16
    ):
        super().__init__()

        self.kernel_length = kernel_length
        self.embedding_size = embedding_size
        self.pooling_size = pooling_size
        self.input_length = input_length

        # Input dimension is (batch_size, 300, 65_536)
        self.conv_l1 = nn.ModuleList([nn.GRU(
            input_size=300,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 64)
        self.pool_l1 = nn.MaxPool1d(
            kernel_size=4,
            stride=4,
            padding=0
        )

        # Input dimension is (batch_size, 64, 16_384)
        self.conv_l2 = nn.ModuleList([nn.GRU(
            input_size=64,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 16)
        self.pool_l2 = nn.MaxPool1d(
            kernel_size=4,
            stride=4,
            padding=0
        )

        # Input dimension is (batch_size, 16, 4096)
        self.conv_l3 = nn.ModuleList([nn.GRU(
            input_size=16,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 16)
        self.pool_l3 = nn.MaxPool1d(
            kernel_size=4,
            stride=4,
            padding=0
        )

        # Input dimension is (batch_size, 16, 1024)
        self.linear_l4 = nn.Linear(
            in_features=16 * 1024,
            out_features=512
        )
        self.linear_l5 = nn.Linear(
            in_features=512,
            out_features=512
        )

        self.output = nn.Linear(
            in_features=512,
            out_features=1
        )

    def forward(self, x):
        # x dimensions = (batch_size, input_length, input_size)

        # Apply layer 1
        l1_convolved = \
            torch.empty((x.shape[0], self.input_length, 64)).type_as(x)
        for i, kernel in enumerate(self.conv_l1):
            for step in range(self.input_length):
                word_first = step - self.kernel_length // 2
                if word_first < 0:
                    word_first = 0
                word_last = step + self.kernel_length // 2
                if word_last > l1_convolved.shape[1]:
                    word_last = l1_convolved.shape[1]
                l1_convolved[:, step, i] = \
                    kernel(x[:, word_first:word_last, :])[1] \
                        .view(2, x.shape[0]).mean(dim=0)
        l1_activated = F.leaky_relu(l1_convolved)
        l1_activated = l1_activated.permute(0, 2, 1)
        l1_pooled = self.pool_l1(l1_activated)
        l1_pooled = l1_pooled.permute(0, 2, 1)

        # Apply layer 2
        l2_convolved = torch.empty(
            (x.shape[0], l1_pooled.shape[1], 16)
        ).type_as(x)
        for i, kernel in enumerate(self.conv_l2):
            for step in range(l1_pooled.shape[1]):
                word_first = step - self.kernel_length // 2
                if word_first < 0:
                    word_first = 0
                word_last = step + self.kernel_length // 2
                if word_last > l2_convolved.shape[1]:
                    word_last = l2_convolved.shape[1]
                l2_convolved[:, step, i] = \
                    kernel(l1_pooled[:, word_first:word_last, :])[1] \
                        .view(2, x.shape[0]).mean(dim=0)
        l2_activated = F.leaky_relu(l2_convolved)
        l2_activated = l2_activated.permute(0, 2, 1)
        l2_pooled = self.pool_l2(l2_activated)
        l2_pooled = l2_pooled.permute(0, 2, 1)

        # Apply layer 3
        l3_convolved = torch.empty(
            (x.shape[0], l2_pooled.shape[1], 16)
        ).type_as(x)
        for i, kernel in enumerate(self.conv_l3):
            for step in range(l2_pooled.shape[1]):
                word_first = step - self.kernel_length // 2
                if word_first < 0:
                    word_first = 0
                word_last = step + self.kernel_length // 2
                if word_last > l3_convolved.shape[1]:
                    word_last = l3_convolved.shape[1]
                l3_convolved[:, step, i] = \
                    kernel(l2_pooled[:, word_first:word_last, :])[1] \
                        .view(2, x.shape[0]).mean(dim=0)
        l3_activated = F.leaky_relu(l3_convolved)
        l3_activated = l3_activated.permute(0, 2, 1)
        l3_pooled = self.pool_l3(l3_activated)
        l3_pooled = l3_pooled.permute(0, 2, 1)

        # Pass through dense layers
        l3_linearized = l3_pooled.reshape(-1, 1024*16)
        l4_out = self.linear_l4(l3_linearized)
        l4_activated = F.leaky_relu(l4_out)
        l5_out = self.linear_l5(l4_activated)
        l5_activated = F.leaky_relu(l5_out)

        # Get single output value in range [0, 1]
        output = F.sigmoid(self.output(l5_activated))
        return torch.tensor(output.flatten(), dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(
            torch.tensor(y_pred, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(
            torch.tensor(y_pred, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(
            torch.tensor(y_pred, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = GRU_CNN()
print(model)

datamodule = BillDataModule(batch_size=32)

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
