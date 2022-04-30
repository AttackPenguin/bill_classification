import os

import torch
from pytorch_lightning import Trainer
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
        self.conv_l1 = [nn.GRU(
            input_size=300,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 64
        self.pool_l1 = nn.MaxPool2d(
            kernel_size=(4, 64),
            stride=(4, 0),
            padding=(0, 0)
        )

        # Input dimension is (batch_size, 64, 16_384)
        self.conv_l2 = [nn.GRU(
            input_size=64,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 16
        self.pool_l2 = nn.MaxPool2d(
            kernel_size=(4, 16),
            stride=(4, 0),
            padding=(0, 0)
        )

        # Input dimension is (batch_size, 16, 4096)
        self.conv_l3 = [nn.GRU(
            input_size=16,
            hidden_size=1,
            bidirectional=True,
            batch_first=True
        )] * 16
        self.pool_l3 = nn.MaxPool2d(
            kernel_size=(4, 16),
            stride=(4, 0),
            padding=(0, 0)
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
        l1_convolved = torch.empty((x.shape[0], self.input_length, 64)).type_as(x)
        for i, kernel in enumerate(self.conv_l1):
            for step in range(self.input_length):
                word_first = step - self.kernel_length // 2
                if word_first < 0:
                    word_first = 0
                word_last = step + self.kernel_length // 2
                if word_last > self.input_length:
                    word_first = self.input_length
                l1_convolved[:, step, i] = \
                    x[:, word_first:word_last, :]
        l1_activated = F.leaky_relu(l1_convolved)
        l1_pooled = self.pool_l1(l1_activated)

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
                if word_last > self.input_length:
                    word_first = self.input_length
                l2_convolved[:, step, i] = \
                    l1_pooled[:, word_first:word_last, :]
        l2_activated = F.leaky_relu(l2_convolved)
        l2_pooled = self.pool_l2(l2_activated)

        # Apply layer 3
        l3_convolved = torch.empty(
            (x.shape[0], l2_pooled.shape[1], 16)
        ).type_as(x)
        for i, kernel in enumerate(self.conv_l3):
            for step in range(l1_pooled.shape[1]):
                word_first = step - self.kernel_length // 2
                if word_first < 0:
                    word_first = 0
                word_last = step + self.kernel_length // 2
                if word_last > self.input_length:
                    word_first = self.input_length
                l3_convolved[:, step, i] = \
                    l2_pooled[:, word_first:word_last, :]
        l3_activated = F.leaky_relu(l3_convolved)
        l3_pooled = self.pool_l3(l3_activated)

        # Pass through dense layers
        l3_linearized = l3_pooled.view(x.shape[0], -1)
        l4_out = self.linear_l4(l3_linearized)
        l4_activated = F.leaky_relu(l4_out)
        l5_out = self.linear_l5(l4_activated)
        l5_activated = F.leaky_relu(l5_out)

        # Get single output value in range [0, 1]
        return F.sigmoid(self.output(l5_activated))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = GRU_CNN()
datamodule = BillDataModule(batch_size=32)

if torch.cuda.is_available():
    trainer = Trainer(devices=-1, accelerator="gpu", strategy="dp")
else:
    trainer = Trainer()
print(model.device)
trainer.fit(model, datamodule)
