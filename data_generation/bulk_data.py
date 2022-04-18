from __future__ import annotations

import json
import os

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
        version_short: str
) -> bool:

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{version_short}',
        f'{version_short}{bill_number}', 'data.json'
    )
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['history']['enacted']


def get_bill_sponsor(
        congress: int,
        bill_number: int,
        version_short: str
) -> str:

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{version_short}',
        f'{version_short}{bill_number}', 'data.json'
    )
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['sponsor']['name']


def get_bill_cosponsors(
        congress: int,
        bill_number: int,
        version_short: str
) -> list[str]:

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{version_short}',
        f'{version_short}{bill_number}', 'data.json'
    )
    with open(file_path, 'r') as file:
        data = json.load(file)
    cosponsors = list()
    for cosponsor in data['cosponsors']:
        cosponsors.append(cosponsor['name'])
    return cosponsors


def get_introduced_date(
        congress: int,
        bill_number: int,
        version_short: str
) -> pd.Timestamp:

    file_path = os.path.join(
        BULK_DATA_DIR,
        f'{congress}', 'bills', f'{version_short}',
        f'{version_short}{bill_number}', 'data.json'
    )
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.Timestamp(data['introduced_at'])


if __name__ == '__main__':
    main()
