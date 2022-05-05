import os

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl

from classification.random_forest.random_forest \
    import get_most_important_features as get_features
from classification.word_cnn_simple.dataloader import BillDataModule
from dataset import get_glove_embeddings


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
            kernel_size=1, stride=1, padding=0,
            bias=False
        )
        self.conv_l2 = nn.Conv1d(
            in_channels=256, out_channels=64,
            kernel_size=9, stride=1, padding=4
        )
        self.pool_l2 = nn.MaxPool1d(
            kernel_size=8, stride=8, padding=0
        )
        self.conv_l3 = nn.Conv1d(
            in_channels=64, out_channels=32,
            kernel_size=9, stride=1, padding=4
        )
        self.pool_l3 = nn.MaxPool1d(
            kernel_size=8, stride=8, padding=0
        )

        # Input dimension is (batch_size, 32, 128)
        self.linear_l5 = nn.Linear(
            in_features=32*128,
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
        l2_convolved = F.leaky_relu(self.conv_l2(l1_convolved))
        l2_pooled = self.pool_l2(l2_convolved)
        l3_convolved = F.leaky_relu(self.conv_l3(l2_pooled))
        l3_pooled = self.pool_l3(l3_convolved)

        # Pass through dense layers
        l4_linearized = l3_pooled.reshape(-1, 32*128)
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
parameters = model.state_dict()
features, importances = get_features(
    "../random_forest/clf1.pickle"
)
embeddings = get_glove_embeddings()
feature_coordinates = [
    embeddings[feature] for feature in features
]
parameters['conv_l1.weight'] = \
    torch.Tensor(np.array(feature_coordinates).reshape((256,300,1)))
model.load_state_dict(parameters)
datamodule = BillDataModule(batch_size=8)

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
