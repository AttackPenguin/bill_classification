from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from data_generation import build_data

congresses = tuple(range(107, 116))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

random_state = 666


def main():
    pass


def get_and_split_data(
        congresses: list[int] = congresses
) -> tuple[tuple[list[str], int],
           tuple[list[str], int]]:
    df = build_data.build_raw_dataframe(congresses)
    texts = df['cleaned_text']
    labels = [int(enacted) for enacted in df['enacted']]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.20, random_state=random_state,
        shuffle=True, stratify=True
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
        stop_words=stop_words, token_pattern=r"(?u)\b[\w]\][\w\-][\w]+\b",
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

    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy*100:.2f}%\n")

    # TODO: testing output

if __name__ == '__main__':
    main()
