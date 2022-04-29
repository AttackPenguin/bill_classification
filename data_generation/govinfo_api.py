import json
import os
import pickle

from bs4 import BeautifulSoup as bs
import requests

import parameters as p

api_root = f"https://api.govinfo.gov/"
api_key = 'tUvUxo2PCWHk9TjCxC66kLDrekBV6C5iO2xRsaV3'
api_documentation = ""
hourly_request_limit = 36_000
headers = {
    'x-api-key': api_key
}

PICKLED_DATA = os.path.join(
    p.PICKLED_DATA, 'govinfo_api'
)
JSON_FILES = p.JSON_FILES
BULK_DATA = p.BULK_DATA

# https://www.govinfo.gov/help/bills
suffixes_of_interest = {
    'ih': 'Introduced (House)',
    'is': 'Introduced (Senate)'
}
congresses = tuple(range(107, 115))
chambers = ('senate', 'house')
chambers_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

# Bill name format:
# f"BILLS-{congress_number}{bill_category}{bill_number}{bill_version}


def main():
    for congress in congresses:
        print(f"Getting Congress {congress}...")
        get_bill_texts_by_congress(
            congress
        )
        print("\n\n\n")


def get_bill_text(
        congress: int,
        chamber: str,
        bill_number: int,
        data_type: str = 'htm',
        use_pickled: bool = True
) -> str:
    if chamber == 'senate':
        category = 's'
        suffix = 'is'
    elif chamber == 'house':
        category = 'hr'
        suffix = 'ih'
    else:
        raise ValueError(
            "Chamber must be one of 'senate' and 'house'.\n"
            f"Cannot be {chamber}."
        )
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        "get_bill_text",
        f"get_bill_text_{congress}_"
        f"{chamber}_{bill_number}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        url = (
                api_root +
                'packages/' +
                f'BILLS-{congress}{category}{bill_number}{suffix}/' +
                f'{data_type}'
        )
        response = requests.get(
            url,
            headers=headers
        )
        html = response.content.decode()
        soup = bs(html, 'html.parser')
        data = soup.text

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


def get_bill_texts_by_congress(
        congress: int,
        use_pickled: bool = True
) -> dict[tuple[int, str, int], str]:

    pickled_file_path = os.path.join(
        PICKLED_DATA,
        "get_bill_text_by_congress",
        f"get_bill_text_by_congress{congress}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = dict()

        for chamber, body_s, version, v_short in zip(
            chambers, chambers_short, versions, versions_short
        ):
            target_directory = os.path.join(
                BULK_DATA, f'{congress}', 'bills', f'{body_s}'
            )
            num_bills = len(os.listdir(target_directory))
            for bill_number in range(1, num_bills+1):
                bill_text = get_bill_text(
                    congress, chamber, bill_number
                )
                if bill_text:
                    data[(congress, body_s, bill_number)] = bill_text
                    print(
                        f"Acquired congress "
                        f"{congress} {chamber} bill {bill_number}:"
                    )
                else:
                    print(f"Failed to acquire {chamber} bill {bill_number}")

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


if __name__ == '__main__':
    main()
