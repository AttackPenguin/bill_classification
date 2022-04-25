from __future__ import annotations

from collections import Counter
import os
import pickle
import re
from string import digits

import nltk
import numpy as np
from nltk import RegexpTokenizer
import pandas as pd

from data_generation import build_data

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'

congresses = tuple(range(107, 116))
c_analysis = tuple(range(107, 115))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_info_rows', 25)


def main():
    df = build_data.build_raw_dataframe(congresses, use_pickled=True)
    run(df)


def run(df: pd.DataFrame):
    df = clean_votes_df(df)
    figs_chronologic(df)


def clean_votes_df(df: pd.DataFrame):
    # Remove that one bad row with mostly missing data, and the only row
    # without a value for 'enacted'...
    df.drop(df.index[df['enacted'].isna()], inplace=True)

    # For non-voting representatives, replace 0 values in vote data columns
    # with np.nan values.
    non_voting_representatives = (
            df['sponsor votes against party'].isna() |
            df['sponsor votes with party'].isna() |
            df['sponsor missed votes'].isna()
    )
    df.loc[non_voting_representatives, 'sponsor votes against party'] = np.nan
    df.loc[non_voting_representatives, 'sponsor votes with party'] = np.nan
    df.loc[non_voting_representatives, 'sponsor missed votes'] = np.nan

    # There are two congress members with no value for gender. Both are male.
    df.loc[df['sponsor gender'].isna(), 'sponsor gender'] = 'm'

    # Get only congresses 107 through 114
    df = df.loc[df['congress'].isin(range(107, 115))]

    return df


def remove_non_voting_members(df: pd.DataFrame):
    df = df.loc(
        ~df['sponsor votes with party'].isna()
    )
    return df


def congress_start_date(x: int):
    return pd.Timestamp(
        int((2001 + (x - 107)) + (x - 107) * 2), 1, 3
    )


def congress_end_date(x: int):
    return pd.Timestamp(
        int((2001 + (x - 107)) + (x - 107) * 2 + 2), 1, 3
    )


def figs_chronologic(df: pd.DataFrame):
    """
    :param df:
    :return:
    """
    congresses = list(range(107, 115))
    num_bills_all = {
        congress: len(df.loc[df['congress'] == congress])
        for congress in congresses
    }
    num_bills_house = {
        congress: len(df.loc[(
                (df['congress'] == congress) &
                (df['chamber'] == 'house')
        )])
        for congress in congresses
    }
    num_bills_senate = {
        congress: len(df.loc[(
                (df['congress'] == congress) &
                (df['chamber'] == 'senate')
        )])
        for congress in congresses
    }
    num_enacted_all = {
        congress: len(df.loc[(
                (df['congress'] == congress) &
                (df['enacted'])
        )])
        for congress in congresses
    }
    num_enacted_house = {
        congress: len(df.loc[(
                (df['congress'] == congress) &
                (df['enacted']) &
                (df['chamber'] == 'house')
        )])
        for congress in congresses
    }
    for key, value in num_enacted_house.items():
        print(key, value)


if __name__ == '__main__':
    main()
