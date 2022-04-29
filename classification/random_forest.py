from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split

from analysis import analysis
from data_generation import build_data

congresses = tuple(range(107, 115))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

random_state = 666


def main():
    (X_train, y_train), (X_test, y_test) = get_and_split_data(congresses)
    vectorizer = get_fitted_vectorizer(X_train, ngram_min=3, ngram_max=5)
    names = vectorizer.get_feature_names_out()
    for name in names:
        print(name)
    file_path = "./classifiers/clf1.pickle"
    clf2 = train_classifier(
        X_train, y_train, vectorizer
    )
    with open(file_path, 'wb') as file:
        pickle.dump((clf2, vectorizer), file)
    # with open(file_path, 'rb') as file:
    #     clf, vectorizer = pickle.load(file)
    test_classifier(clf2, X_test, y_test, vectorizer)


def get_and_split_data(
        congresses: tuple[int] = congresses
) -> tuple[tuple[list[str], int],
           tuple[list[str], int]]:

    df = build_data.build_raw_dataframe(congresses)
    df = analysis.clean_votes_df(df)
    tokenized_texts = df['tokenized text'].to_numpy()

    texts = [
        ' '.join([token for token in tokenized_text if
                    (
                          len(token) >= 3 and
                          '-' not in token and
                          '_' not in token
                    )
                  ])
        for tokenized_text in tokenized_texts
    ]
    labels = df['enacted'].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.25, random_state=random_state,
        shuffle=True, stratify=labels
    )

    return (X_train, y_train), (X_test, y_test)


def get_fitted_vectorizer(
        texts: list[str],
        max_features: int = 1000,
        ngram_min: int = 1,
        ngram_max: int = 1,
        stop_words: str | None = 'english'
) -> CountVectorizer:
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"(?u)\b\w\w+\b",
        # token_pattern=r"(?u)\b[\w]\][\w\-][\w]+\b",
        ngram_range=(ngram_min, ngram_max), max_features=max_features
    )
    vectorizer.fit(texts)

    return vectorizer


def train_classifier(
        X_train: list[str],
        y_train: list[int],
        vectorizer: CountVectorizer
) -> RandomForestClassifier:
    X_train = vectorizer.transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=1000, criterion='gini',
        max_depth=None, max_features='sqrt',
        bootstrap=True, n_jobs=4,
        random_state=random_state, verbose=2,
        class_weight='balanced_subsample'
    )

    clf.fit(X_train, y_train)

    return clf


def test_classifier(
        clf: RandomForestClassifier,
        X_test: list[str],
        y_test: list[int],
        vectorizer: CountVectorizer
):
    X_test = vectorizer.transform(X_test)

    display = PrecisionRecallDisplay.from_estimator(
        clf, X_test, y_test
    )
    display.plot()


if __name__ == '__main__':
    main()
