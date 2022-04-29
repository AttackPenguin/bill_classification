from collections import defaultdict
import random
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset

from classification.gru_cnn.dataset import BillDataset


class BDMDataset(Dataset):
    def __init__(
            self,
            texts,
            labels
    ):
        self.texts = texts
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        while len(text) < 65_536:
            text += text
        return text[0:65_536], self.labels[idx]


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

        data = list(zip(dataset.texts, dataset.labels))
        random.shuffle(data)
        texts, labels = tuple(zip(*data))

        enacted = list()
        failed = list()
        for text, label in zip(texts, labels):
            if label == 1:
                enacted.append((text, label))
            if label == 0:
                failed.append((text, label))

        n_enacted = len(enacted)
        n_failed = len(failed)
        n_enacted_train = int(n_enacted*0.8)
        n_failed_train = int(n_failed*0.8)
        n_enacted_remainder = n_enacted - n_enacted_train
        n_failed_remainder = n_failed - n_failed_train
        n_enacted_val = int(n_enacted_remainder*0.5)
        n_failed_val = int(n_failed_remainder*0.5)
        n_enacted_test = n_enacted_remainder - n_enacted_val
        n_failed_test = n_failed_remainder - n_failed_val

        start_enacted = 0
        stop_enacted = n_enacted_train
        start_failed = 0
        stop_failed = n_failed_train
        self.X_train = (
            [x[0] for x in enacted[start_enacted:stop_enacted]] +
            [x[0] for x in failed[start_failed:stop_failed]]
        )
        self.y_train = (
            [x[1] for x in enacted[start_enacted:stop_enacted]] +
            [x[1] for x in failed[start_failed:stop_failed]]
        )
        start_enacted = n_enacted_train
        stop_enacted = n_enacted_train + n_enacted_val
        start_failed = n_failed_train
        stop_failed = n_failed_train + n_failed_val
        self.X_val = (
            [x[0] for x in enacted[start_enacted:stop_enacted]] +
            [x[0] for x in failed[start_failed:stop_failed]]
        )
        self.y_val = (
            [x[1] for x in enacted[start_enacted:stop_enacted]] +
            [x[1] for x in failed[start_failed:stop_failed]]
        )
        start_enacted = n_enacted_train + n_enacted_val
        stop_enacted = n_enacted_train + n_enacted_val + n_enacted_test
        start_failed = n_failed_train + n_failed_val
        stop_failed = n_failed_train + n_failed_val + n_failed_test
        self.X_test = (
            [x[0] for x in enacted[start_enacted:stop_enacted]] +
            [x[0] for x in failed[start_failed:stop_failed]]
        )
        self.y_test = (
            [x[1] for x in enacted[start_enacted:stop_enacted]] +
            [x[1] for x in failed[start_failed:stop_failed]]
        )
        pass

    def train_dataloader(self):
        train_dataset = BDMDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        val_dataset = BDMDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = BDMDataset(self.X_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size)
