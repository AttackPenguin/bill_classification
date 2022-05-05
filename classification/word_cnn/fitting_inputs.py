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
from model import WordCNN

output_class = 1
input_length: int = 16_384


class WordFit(nn.Module):
    def __init__(
            self,
            wordcnn: WordCNN,
            output_class: int = 1,
            input_length: int = 16_384
    ):
        super(WordFit, self).__init__()

        self.wordcnn = WordCNN
        self.wordcnn.freeze()

        self.x = (torch.rand(
            input_length,
            requires_grad=True,
            dtype=torch.float32
        )) * 2 - 1
        self.y = torch.tensor([output_class], dtype=torch.float32)

    def forward(self):
        return self.wordcnn(self.input)


if __name__ == '__main__':
    file_path = os.path.join(
        ''
    )
    wordcnn = WordCNN.load_from_checkpoint(file_path)
    output_class = 1
    wordfit = WordFit(
        wordcnn,
        output_class
    )

    optimizer = optim.Adam(wordfit.parameters(), lr=1e-3)

    for _ in range(1000):
        y_pred = wordcnn()
        print(y_pred)
        loss = F.binary_cross_entropy(y_pred, wordfit.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wordcnn2 = WordCNN.load_from_checkpoint(file_path)
