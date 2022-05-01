from collections import defaultdict
import random
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset

from classification.word_cnn.dataset import BillDataset


def stratified_shuffle(
        texts: list[list[list[float]]],
        labels: list[int],
        split: float
):
    data = list(zip(texts, labels))
    random.shuffle(data)

    enacted = list()
    failed = list()
    for text, label in zip(texts, labels):
        if label == 1:
            enacted.append((text, label))
        if label == 0:
            failed.append((text, label))

    n_enacted = len(enacted)
    n_failed = len(failed)
    split_enacted = int(n_enacted * split)
    split_failed = int(n_failed * split)

    first = (
            [x for x in enacted[:split_enacted]] +
            [x for x in failed[:split_failed]]
    )
    second = (
            [x for x in enacted[split_enacted:]] +
            [x for x in failed[split_failed:]]
    )

    random.shuffle(first)
    random.shuffle(second)

    X_first = [x[0] for x in first]
    y_first = [x[1] for x in first]
    X_second = [x[0] for x in second]
    y_second = [x[1] for x in second]

    return (X_first, y_first), (X_second, y_second)


class BDMDataset(Dataset):
    def __init__(
            self,
            texts: np.ndarray,
            labels: list[int]
    ):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        while len(text) < 16_384:
            text += text
        return (
            np.array(text[0:16_384]),
            self.labels[idx]
        )


class BillDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 32
    ):
        super().__init__()

        self.batch_size = batch_size

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def setup(self, stage: Optional[str] = None):
        dataset = BillDataset()
        X = dataset.texts
        y = dataset.labels

        (self.X_train, self.y_train), (X, y) = stratified_shuffle(X, y, 0.8)
        (self.X_val, self.y_val), (self.X_test, self.y_test) = \
            stratified_shuffle(X, y, 0.5)

    def train_dataloader(self):
        train_dataset = BDMDataset(self.X_train, self.y_train)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        val_dataset = BDMDataset(self.X_val, self.y_val)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        test_dataset = BDMDataset(self.X_test, self.y_test)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
