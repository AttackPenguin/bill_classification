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

from data_generation import bulk_data as bd
from data_generation import congress_gov_scraping as cgs
from data_generation import prorepublica_api as pra

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'

congresses = tuple(range(107, 116))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

# nltk.download('all')


def main():
    df = build_raw_dataframe(congresses, use_pickled=False)
    print('done')


def tokenize_text(
        text: str
) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', digits))
    tokenizer = RegexpTokenizer(r'[\w\-]+', flags=re.UNICODE)
    tokenized_text = tokenizer.tokenize(text)
    return tokenized_text


def get_token_counts(
        tokenized_text: list[str] | list[list[str]],
        return_count: int | None = None
) -> list[tuple[str, int]]:
    token_counts = Counter(tokenized_text)
    if return_count:
        return token_counts.most_common()
    else:
        return token_counts.most_common(return_count)


def get_n_gram_list(
        tokenized_text: list[str],
        n: int
) -> list[list[str]]:
    n_grams = list()
    for i in range(len(tokenized_text)+1-n):
        n_grams.append(tokenized_text[i:i+n])
    return n_grams


def build_raw_dataframe(
        congresses: tuple[int],
        use_pickled: bool = True
):
    # Define the file path to where the pickled data frame is or will be stored.
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"raw_df_{congresses}.pickle"
    )
    # If the pickled data frame is there, unpickle it and return it.
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            df = pickle.load(file)
    # Otherwise build a new dataframe from scratch.
    else:
        # Construct a dict to add items to. After adding all items of
        # interest, we'll use this dict to construct our data frame.
        df_data = dict()

        # Get all introduced bills original text for the congresses specified.
        bill_texts = dict()  # type: dict[tuple[int, str, int], str]
        for congress in congresses:
            bill_texts.update(
                cgs.get_bill_texts_by_congress(
                    congress, bodies, bodies_short,
                    versions, versions_short
                )
            )

        # Create a column containing the number of the congress
        df_data['congress'] = [key[0] for key in bill_texts.keys()]

        # Create a column containing the chamber in which the bill was proposed.
        df_data['chamber'] = [
            'senate' if key[1] == 's' else 'house'
            for key in bill_texts.keys()]

        # Create a column containing the bill number
        df_data['bill number'] = [key[2] for key in bill_texts.keys()]

        # Create a column containing the raw text of the bill.
        # TODO: Figure out how much boiler plate to chop off the front of
        #  this raw text.
        df_data['text'] = [value for value in bill_texts.values()]

        # Create a column containing the tokenized text of the bill. This
        # method removes all non-alphanumeric characters.
        df_data['tokenized text'] = [
            tokenize_text(text) for text in bill_texts.values()
        ]

        # Create a column containing cleaned text: the tokenized text joined
        # back together into a single document, minus undesirable tokens.
        df_data['cleaned text'] = [
            ' '.join(tokens) for tokens in df_data['tokenized text']
        ]

        # Create a column containing the two letter string that is used in
        # our bulk data directory structure to specify bills introduced to
        # the house ('ih') and introduced to the senate ('is'). This is used
        # in methods to create other columns, and this makes the code much
        # simpler.
        # df_data['version_short'] = [
        #     'is' if item == 'senate' else 'ih' for item in df_data['chamber']
        # ]

        # Create a column indicating whether a bill was enacted into law.
        df_data['enacted'] = [
            bd.get_bill_enacted(congress, bill_number, chamber)
            for (congress, bill_number, chamber)
            in zip(df_data['congress'], df_data['bill number'],
                   df_data['chamber'])
        ]

        # Create a column with the full name of the sponsor of the bill,
        # in the format:
        # [Last], [First] [middle initial]
        df_data['sponsor'] = [
            bd.get_bill_sponsor(congress, bill_number, chamber)
            for (congress, bill_number, chamber)
            in zip(df_data['congress'], df_data['bill number'],
                   df_data['chamber'])
        ]

        # Create a column holding the sponsor's party.
        sp_first_names, sp_last_names = (
            [''.join([c for c in name.split()[1] if c.isalpha()])
             if name is not np.NaN else np.NaN
             for name in df_data['sponsor']],
            [''.join([c for c in name.split()[0] if c.isalpha()])
             if name is not np.NaN else np.NaN
             for name in df_data['sponsor']]
        )
        df_data['sponsor party'] = [
            pra.get_member_party(congress, chamber, first, last)
            for (congress, chamber, first, last)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names)
        ]

        # Create a column containing a list of lists of sponsors of the bill,
        # with each sponsor's name in the format:
        # [Last], [First] [middle initial]
        df_data['cosponsors'] = [
            bd.get_bill_cosponsors(congress, bill_number, chamber)
            for (congress, bill_number, chamber)
            in zip(df_data['congress'], df_data['bill number'],
                   df_data['chamber'])
        ]

        # Create a column containing the number of cosponsors.
        df_data['number of cosponsors'] = [
            np.int64(len(item)) if item is not np.NaN else np.NaN
            for item in df_data['cosponsors']
        ]

        # Create a column containing a list of the parties to which cosponsors
        # belong.
        cosponsor_parties = list()
        for cosponsor_list in df_data['cosponsors']:
            if cosponsor_list is not np.NaN:
                cs_first_names, cs_last_names = (
                    [''.join([c for c in name.split()[1] if c.isalpha()])
                     if name is not np.NaN else np.NaN
                     for name in cosponsor_list],
                    [''.join([c for c in name.split()[0] if c.isalpha()])
                     if name is not np.NaN else np.NaN
                     for name in cosponsor_list]
                )
                cosponsor_parties.append(
                    [
                        pra.get_member_party(congress, chamber, first, last)
                        for (congress, chamber, first, last)
                        in zip(df_data['congress'], df_data['chamber'],
                               cs_first_names, cs_last_names)
                    ]
                )
            else:
                cosponsor_parties.append(np.NaN)
        df_data['cosponsor parties'] = cosponsor_parties

        # Create columns indicating whether or not there is a member of a
        # given party either sponsoring or cosponsoring a given bill.
        df_data['republican sponsor or cosponsor'] = [
            'R' in cosponsors + [sponsor]
            if (cosponsors is not np.NaN) and (sponsor is not np.NaN)
            else np.NaN
            for (cosponsors, sponsor)
            in zip(df_data['cosponsor parties'], df_data['sponsor party'])
        ]
        df_data['democrat sponsor or cosponsor'] = [
            'D' in cosponsors + [sponsor]
            if (cosponsors is not np.NaN) and (sponsor is not np.NaN)
            else np.NaN
            for (cosponsors, sponsor)
            in zip(df_data['cosponsor parties'], df_data['sponsor party'])
        ]
        df_data['other party sponsor or cosponsor'] = [
            [i for i in cosponsors + [sponsor] if i not in ['D', 'R']]
            if (cosponsors is not np.NaN) and (sponsor is not np.NaN)
            else np.NaN
            for (cosponsors, sponsor)
            in zip(df_data['cosponsor parties'], df_data['sponsor party'])
        ]

        # Create a column indicating if there is both a republican and a
        # democrat among the sponsor and cosponsors.
        df_data['cross party sponsorship'] = [
            democrat and republican
            if (democrat is not np.NaN) and (republican is not np.NaN)
            else np.NaN
            for (democrat, republican)
            in zip(df_data['democrat sponsor or cosponsor'],
                   df_data['republican sponsor or cosponsor'])
        ]

        # Create a column with the date the bill was introduced
        df_data['date introduced'] = [
            bd.get_introduced_date(congress, bill_number, chamber)
            for (congress, bill_number, chamber)
            in zip(df_data['congress'], df_data['bill number'],
                   df_data['chamber'])
        ]

        # Create a column with the age of the sponsor.
        df_data['sponsor age'] = [
            pra.get_member_age(congress, chamber, first, last, date_introduced)
            for (congress, chamber, first, last, date_introduced)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names, df_data['date introduced'])
        ]

        # Create a column with the gender of the sponsor.
        df_data['sponsor gender'] = [
            pra.get_member_gender(congress, chamber, first, last)
            for (congress, chamber, first, last)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names)
        ]

        # Create a column containing the percentage of votes the sponsor has
        # missed.
        df_data['sponsor missed votes'] = [
            pra.get_member_missed_votes_perc(congress, chamber, first, last)
            for (congress, chamber, first, last)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names)
        ]

        # Create a column with the percentage of time the sponsor votes with
        # their party.
        df_data['sponsor votes with party'] = [
            pra.get_member_votes_with_party_pct(congress, chamber, first, last)
            for (congress, chamber, first, last)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names)
        ]

        # Create a colun with the percentage of time the sponsor votes
        # against their party.
        df_data['sponsor votes against party'] = [
            pra.get_member_votes_against_party_pct(congress, chamber, first, last)
            for (congress, chamber, first, last)
            in zip(df_data['congress'], df_data['chamber'],
                   sp_first_names, sp_last_names)
        ]

        df = pd.DataFrame(df_data)

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(df, file)

    return df


if __name__ == '__main__':
    main()
