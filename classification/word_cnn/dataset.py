import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from analysis import analysis
from data_generation import build_data
import parameters as p

PICKLED_DATA = p.PICKLED_DATA

congresses = tuple(range(107, 115))


class BillDataset(Dataset):
    def __init__(
            self,
            congresses: tuple[int] = congresses,
            use_pickled: bool = True
    ):

        pickled_file_path = os.path.join(
            PICKLED_DATA,
            f"bill_dataset.pickle"
        )
        if use_pickled and os.path.exists(pickled_file_path):
            with open(pickled_file_path, 'rb') as file:
                self.texts, self.labels = pickle.load(file)
        else:
            df = build_data.build_raw_dataframe(congresses)
            df = analysis.clean_votes_df(df)
            tokenized_texts = df['tokenized text'].to_list()

            glove_embeddings = get_glove_embeddings()
            self.texts = list()
            for tokenized_text in tokenized_texts:
                vectorized_text = list()
                for token in tokenized_text:
                    if '-' in token or '_' in token:
                        continue
                    try:
                        vectorized_text.append(glove_embeddings[token])
                    except KeyError:
                        vectorized_text.append([0.0]*300)
                self.texts.append(vectorized_text)
            self.labels = df['enacted'].astype(int).to_numpy()

            with open(pickled_file_path, 'wb') as file:
                pickle.dump((self.texts, self.labels), file)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def get_glove_embeddings(
        use_pickled: bool = True
) -> dict[str, np.ndarray]:

    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"glove_embeddings.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            glove_embeddings = pickle.load(file)
    else:
        glove_embeddings = dict()
        with open('./glove.6B.300d.txt', 'r') as file:
            for line in file:
                values = line.split(' ')
                word = values[0]  # The first entry is the word
                coefs = list(np.asarray(values[1:], dtype='float32'))
                glove_embeddings[word] = coefs

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(glove_embeddings, file)

    return glove_embeddings


if __name__ == '__main__':
    dataset = BillDataset(congresses)
    print('done')