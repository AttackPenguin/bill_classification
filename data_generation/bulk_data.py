from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'

congresses = tuple(range(107, 116))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')


def main():
    for bill_number in range(1, 500):
        enacted = get_bill_enacted(
            107, bill_number, 's'
        )
        print(f"bill:", enacted)


def get_bill_enacted(
        congress: int,
        bill_number: int,
        chamber: str
) -> bool | None:

    if chamber == 'senate':
        dir_label = 's'
    else:
        dir_label = 'hr'

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{dir_label}',
        f'{dir_label}{bill_number}', 'data.json'
    )
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        result = data['history']['enacted']
        return result
    except FileNotFoundError:
        return np.NaN


def get_bill_sponsor(
        congress: int,
        bill_number: int,
        chamber: str
) -> str:

    if chamber == 'senate':
        dir_label = 's'
    else:
        dir_label = 'hr'

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{dir_label}',
        f'{dir_label}{bill_number}', 'data.json'
    )
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data['sponsor']['name']
    except FileNotFoundError:
        return np.NaN


def get_bill_cosponsors(
        congress: int,
        bill_number: int,
        chamber: str
) -> list[str]:

    if chamber == 'senate':
        dir_label = 's'
    else:
        dir_label = 'hr'

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{dir_label}',
        f'{dir_label}{bill_number}', 'data.json'
    )

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        cosponsors = list()
        for cosponsor in data['cosponsors']:
            cosponsors.append(cosponsor['name'])
        return cosponsors
    except FileNotFoundError:
        return np.NaN


def get_introduced_date(
        congress: int,
        bill_number: int,
        chamber: str
) -> pd.Timestamp:

    if chamber == 'senate':
        dir_label = 's'
    else:
        dir_label = 'hr'

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{dir_label}',
        f'{dir_label}{bill_number}', 'data.json'
    )

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return pd.Timestamp(data['introduced_at'])
    except FileNotFoundError:
        return np.NaN


if __name__ == '__main__':
    main()
