from __future__ import annotations

import os

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from data_generation import build_data

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'
FIGURES_DIR = 'figures'

congresses = tuple(range(107, 115))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

pd.set_option('display.precision', 2)
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_info_rows', 25)

sns.set_theme(context='paper')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']

c_house = 'C1'
c_senate = 'C2'
c_republican = 'C3'
c_democrat = 'C0'
c_other = 'C4'
c_enacted = 'C1'
c_failed = 'C2'


def main():
    dttm_start = pd.Timestamp.now()
    print(f"Started at {dttm_start}")

    df = build_data.build_raw_dataframe(congresses, use_pickled=True)
    run(df)

    dttm_stop = pd.Timestamp.now()
    print(f"Ended at {dttm_stop}")
    print(f"Runtime: {dttm_stop - dttm_start}")


def run(df: pd.DataFrame):
    df = clean_votes_df(df)
    # basic_data(df)
    # chronologic(df)
    # cosponsors(df)
    print(
        df['tokenized text'].map(lambda x: len(x)).max()
    )


def clean_votes_df(df: pd.DataFrame):
    # Remove reserved, unused bill slots.
    df = df[df['text'].str.startswith("\n[Congressional Bills")]

    # Remove that one bad row with mostly missing data, and the only row
    # without a value for 'enacted'...
    df = df[~df['enacted'].isna()]

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

    # Convert number of cosponsors from float to int
    df['number of cosponsors'] = df['number of cosponsors'].astype(int)

    # Enacted column is currently of type object - convert to bool
    df['enacted'] = df['enacted'].astype(bool)

    return df


def remove_non_voting_members(df: pd.DataFrame):
    # Given that non-voting members don't vote... we can eliminate their
    # proposed bills by eliminating rows where 'sponsor votes with party' are
    # null.
    df = df[~df['sponsor votes with party'].isna()]
    return df


def congress_start_date(x: int):
    return pd.Timestamp(
        int((2001 + (x - 107)) + (x - 107) * 2), 1, 3
    )


def congress_end_date(x: int):
    return pd.Timestamp(
        int((2001 + (x - 107)) + (x - 107) * 2 + 2), 1, 3
    )


def basic_data(df: pd.DataFrame):
    df.info()
    print()
    print(df.describe())
    print()
    print('Number of null values per column:')
    for column in df.columns:
        print(column, '\t', df[column].isna().sum())
    print()


def chronologic(df: pd.DataFrame):
    """
    :param df:
    :return:
    """
    congresses = list(range(107, 115))
    num_bills_all = {
        congress: len(df[df['congress'] == congress])
        for congress in congresses
    }
    num_bills_house = {
        congress: len(df[(
                (df['congress'] == congress) &
                (df['chamber'] == 'house')
        )])
        for congress in congresses
    }
    num_bills_senate = {
        congress: len(df[(
                (df['congress'] == congress) &
                (df['chamber'] == 'senate')
        )])
        for congress in congresses
    }
    num_enacted_all = {
        congress: len(df[(
                (df['congress'] == congress) &
                (df['enacted'])
        )])
        for congress in congresses
    }
    num_enacted_house = {
        congress: len(df[(
                (df['congress'] == congress) &
                (df['enacted']) &
                (df['chamber'] == 'house')
        )])
        for congress in congresses
    }
    num_enacted_senate = {
        congress: len(df[(
                (df['congress'] == congress) &
                (df['enacted']) &
                (df['chamber'] == 'senate')
        )])
        for congress in congresses
    }
    percent_enacted_all = {
        congress: num_enacted_all[congress] / num_bills_all[congress]
        for congress in congresses
    }
    percent_enacted_all_period = (
            sum(num_enacted_all.values()) / sum(num_bills_all.values())
    )
    percent_enacted_house = {
        congress: num_enacted_house[congress] / num_bills_house[congress]
        for congress in congresses
    }
    percent_enacted_house_period = (
            sum(num_enacted_house.values()) / sum(num_bills_house.values())
    )

    percent_enacted_senate = {
        congress: num_enacted_senate[congress] / num_bills_senate[congress]
        for congress in congresses
    }
    percent_enacted_senate_period = (
            sum(num_enacted_senate.values()) / sum(num_bills_senate.values())
    )

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'enacted_001.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    x_pos = np.arange(len(congresses))
    width = 0.35
    y_house = list(num_bills_house.values())
    y_senate = list(num_bills_senate.values())

    rects_house = ax.bar(
        x_pos - width / 2, y_house, width, label='House', color=c_house
    )
    rects_senate = ax.bar(
        x_pos + width / 2, y_senate, width, label='Senate', color=c_senate
    )

    plt.subplots_adjust(top=0.92, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig enacted_001: Total Bills Introduced by Congress and Chamber"
        r"\end{large}\par\begin{small}"
        r""
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Congress')
    ax.set_ylabel('Bills Introduced')

    ax.set_xticks(x_pos, congresses)

    ax.set_ylim(top=8500)

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'enacted_002.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    x_pos = np.arange(len(congresses))
    width = 0.35
    y_house = list(num_enacted_house.values())
    y_senate = list(num_enacted_senate.values())

    rects_house = ax.bar(
        x_pos - width / 2, y_house, width, label='House', color=c_house
    )
    rects_senate = ax.bar(
        x_pos + width / 2, y_senate, width, label='Senate', color=c_senate
    )

    plt.subplots_adjust(top=0.92, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig enacted_002: Total Bills Enacted by Congress and Chamber"
        r"\end{large}\par\begin{small}"
        r""
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Congress')
    ax.set_ylabel('Bills Enacted')

    ax.set_xticks(x_pos, congresses)

    ax.set_ylim(top=400)

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'enacted_003.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    x_pos = np.arange(len(congresses))
    width = 0.35
    y_house = list(percent_enacted_house.values())
    y_senate = list(percent_enacted_senate.values())

    rects_house = ax.bar(
        x_pos - width / 2, y_house, width, label='House', color=c_house
    )
    rects_senate = ax.bar(
        x_pos + width / 2, y_senate, width, label='Senate', color=c_senate
    )

    plt.subplots_adjust(top=0.92, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig enacted_003: Percent of Bills Enacted by Congress and Chamber"
        r"\end{large}\par\begin{small}"
        r""
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Congress')
    ax.set_ylabel('Percent Enacted')

    ax.set_xticks(x_pos, congresses)

    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=1, symbol=None, is_latex=True)
    )

    ax.set_ylim(top=0.07)

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'enacted_004.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    groups = [
        'Democratic Congress,\nRepublican Presidency',
        'Mixed Congress,\nRepublican Presidency',
        'Republican Congress,\nRepublican Presidency',
        'Republican Congress,\nDemocratic Presidency',
        'Mixed Congress,\nDemocratic Presidency',
        'Democratic Congress,\nDemocratic Presidency'
    ]
    group_pos = np.arange(len(groups))

    values = [
        np.mean([num_enacted_all[110]]),
        np.mean([num_enacted_all[107]]),
        np.mean([num_enacted_all[108], num_enacted_all[109]]),
        np.mean([num_enacted_all[114]]),
        np.mean([num_enacted_all[112], num_enacted_all[113]]),
        np.mean([num_enacted_all[111]])
    ]

    rects = ax.barh(
        group_pos, values, height=0.35, color=c_other
    )

    plt.subplots_adjust(top=0.92, bottom=.15, left=0.3)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig enacted_004: Total Bills Enacted by Party Control of Chambers "
        r"and "
        r"Presidency"
        r"\end{large}\par\begin{small}"
        r""
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Bills Enacted')

    ax.set_yticks(group_pos, groups)

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'enacted_005.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    groups = [
        'Democratic Congress,\nRepublican Presidency',
        'Mixed Congress,\nRepublican Presidency',
        'Republican Congress,\nRepublican Presidency',
        'Republican Congress,\nDemocratic Presidency',
        'Mixed Congress,\nDemocratic Presidency',
        'Democratic Congress,\nDemocratic Presidency'
    ]
    group_pos = np.arange(len(groups))

    values = [
        num_enacted_all[110] / num_bills_all[110],
        num_enacted_all[107] / num_bills_all[107],
        (num_enacted_all[108] + num_enacted_all[109]) /
        (num_bills_all[108] + num_bills_all[109]),
        num_enacted_all[114] / num_bills_all[114],
        (num_enacted_all[112] + num_enacted_all[113]) /
        (num_bills_all[112] + num_bills_all[113]),
        num_enacted_all[111] / num_bills_all[111],
    ]

    rects = ax.barh(
        group_pos, values, height=0.35, color=c_other
    )

    plt.subplots_adjust(top=0.92, bottom=.15, left=0.3)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig enacted_004: Percent Bills Enacted by Party Control of Chambers "
        r"and "
        r"Presidency"
        r"\end{large}\par\begin{small}"
        r""
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Percent Enacted')
    ax.xaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=1, symbol=None, is_latex=True)
    )

    ax.set_yticks(group_pos, groups)

    fig.show()
    fig.savefig(file_path)

    ############################################################################


def cosponsors(df: pd.DataFrame):
    """
    :param df:
    :return:
    """

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_001.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    r_values = df[df['sponsor party'] == 'R']['number of cosponsors']
    d_values = df[df['sponsor party'] == 'D']['number of cosponsors']
    bin_max = int(max(max(r_values), max(d_values)))
    bins = list(range(0, bin_max, 10))
    ax.hist(
        r_values, bins=bins, density=True, log=False,
        color=c_republican, alpha=0.5, label='Republican'
    )
    ax.hist(
        d_values, bins=bins, density=True, log=False,
        color=c_democrat, alpha=0.5, label='Democrat'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_001: Number of Cosponsors by Party"
        r"\end{large}\par\begin{small}"
        r"All Values, Linear Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_002.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    r_values = df[df['sponsor party'] == 'R']['number of cosponsors']
    d_values = df[df['sponsor party'] == 'D']['number of cosponsors']
    bin_max = int(max(max(r_values), max(d_values)))
    bins = list(range(0, bin_max, 10))
    ax.hist(
        r_values, bins=bins, density=True, log=True,
        color=c_republican, alpha=0.5, label='Republican'
    )
    ax.hist(
        d_values, bins=bins, density=True, log=True,
        color=c_democrat, alpha=0.5, label='Democrat'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_002: Number of Cosponsors by Party"
        r"\end{large}\par\begin{small}"
        r"All Values, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_003.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    r_values = df[
        (df['sponsor party'] == 'R') & (df['number of cosponsors'] <= 50)
        ]['number of cosponsors']
    d_values = df[
        (df['sponsor party'] == 'D') & (df['number of cosponsors'] <= 50)
        ]['number of cosponsors']
    bin_max = max(max(r_values), max(d_values))
    bins = list(range(0, bin_max + 2))
    ax.hist(
        list(r_values), bins=bins, density=True, log=True, align='left',
        color=c_republican, alpha=0.5, label='Republican'
    )
    ax.hist(
        list(d_values), bins=bins, density=True, log=True, align='left',
        color=c_democrat, alpha=0.5, label='Democrat'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_003: Number of Cosponsors by Party"
        r"\end{large}\par\begin{small}"
        r"Number of Cosponsors Less Than or Equal to 50, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_004.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    enacted = df[(
        df['enacted']
    )]['number of cosponsors']
    failed = df[(
        ~df['enacted']
    )]['number of cosponsors']
    bin_max = max(max(enacted), max(failed))
    bins = list(range(0, bin_max, 10))
    ax.hist(
        enacted, bins=bins, density=True, log=True,
        color=c_enacted, alpha=0.5, label='Enacted'
    )
    ax.hist(
        failed, bins=bins, density=True, log=True,
        color=c_failed, alpha=0.5, label='Failed'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_004: Number of Cosponsors by Bill Outcome"
        r"\end{large}\par\begin{small}"
        r"All Values, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_005.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    enacted = df[(
        df['enacted'] & (df['number of cosponsors'] <= 50)
    )]['number of cosponsors']
    failed = df[(
        ~df['enacted'] & (df['number of cosponsors'] <= 50)
    )]['number of cosponsors']
    bin_max = max(max(enacted), max(failed))
    bins = list(range(0, bin_max + 2))
    ax.hist(
        enacted, bins=bins, density=True, log=True,
        color=c_enacted, alpha=0.5, label='Enacted'
    )
    ax.hist(
        failed, bins=bins, density=True, log=True,
        color=c_failed, alpha=0.5, label='Failed'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_005: Number of Cosponsors by Bill Outcome"
        r"\end{large}\par\begin{small}"
        r"Number of Cosponsors Less Than or Equal to 50, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_006.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    male = df[(
        df['sponsor gender'] == 'M'
    )]['number of cosponsors']
    female = df[(
        df['sponsor gender'] == 'F'
    )]['number of cosponsors']
    bin_max = max(max(male), max(female))
    bins = list(range(0, bin_max, 10))
    ax.hist(
        male, bins=bins, density=True, log=True,
        color=c_enacted, alpha=0.5, label='Male'
    )
    ax.hist(
        female, bins=bins, density=True, log=True,
        color=c_failed, alpha=0.5, label='Female'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_006: Number of Cosponsors by Sponsor Gender"
        r"\end{large}\par\begin{small}"
        r"All Values, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################

    file_path = os.path.join(
        FIGURES_DIR,
        'sponsors_007.png'
    )

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    male = df[(
            (df['sponsor gender'] == 'M') & (df['number of cosponsors'] <= 50)
    )]['number of cosponsors']
    female = df[(
            (df['sponsor gender'] == 'F') & (df['number of cosponsors'] <= 50)
    )]['number of cosponsors']
    bin_max = max(max(male), max(female))
    bins = list(range(0, bin_max + 2))
    ax.hist(
        male, bins=bins, density=True, log=True,
        color=c_enacted, alpha=0.5, label='Male'
    )
    ax.hist(
        female, bins=bins, density=True, log=True,
        color=c_failed, alpha=0.5, label='Female'
    )

    plt.subplots_adjust(top=0.89, bottom=.15)
    fig.suptitle(
        r"\begin{center}\begin{large}"
        r"Fig sponsors_007: Number of Cosponsors by Gender of Sponsor"
        r"\end{large}\par\begin{small}"
        r"Number of Cosponsors Less Than or Equal to 50, Log Scaling of Density"
        r"\end{small}\end{center}"
    )

    ax.set_xlabel('Number of Cosponsors')
    ax.set_ylabel('Density')

    ax.legend(loc='upper right')

    fig.show()
    fig.savefig(file_path)

    ############################################################################


if __name__ == '__main__':
    main()
