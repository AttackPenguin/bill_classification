import json
import os
import pickle

import requests

api_root = f"https://api.govinfo.gov/"
api_key = 'tUvUxo2PCWHk9TjCxC66kLDrekBV6C5iO2xRsaV3'
api_documentation = ""
daily_request_limit = 0
headers = {
    'x-api-key': api_key
}

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'

# https://www.govinfo.gov/help/bills
suffixes_of_interest = {
    'enr': 'Enrolled',  # Final bill passed by both bodies
    'ih': 'Introduced (House)',
    'is': 'Introduced (Senate)'
}
congress_numbers = list(range(107, 116))
bill_categories = ('s', 'hr')
bill_versions = ('ih', 'is')

# Bill name format:
# f"BILLS-{congress_number}{bill_category}{bill_number}{bill_version}


def main():
    print(get_bill_text(107, 'hr', 1000, 'ih'))


def get_bill_text(
        congress: int,
        category: str,
        bill_number: int,
        suffix: str,
        data_type: str = 'htm',
        use_pickled: bool = True
) -> str:
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_bill_text_{congress}_"
        f"{category}_{bill_number}_"
        f"{suffix}_{data_type}.pickle"
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
        data = response.text

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


def get_bill_text_by_congress(
        congress: int,
        categories: tuple[str] = bill_categories,
        versions: tuple[str] = bill_versions,
        use_pickled: bool = True
) -> list[str]:
    data = list()

    for category in categories:
        target_directory = os.path.join(
            BULK_DATA_DIR, f'{congress}', 'bills', f'category'
        )
        num_bills = os.listdir(target_directory)
        for version in versions:
            pass


def get_bill_enacted_by_congress(
        congress: int,
        categories: tuple[str] = bill_categories,
        versions: tuple[str] = bill_versions,
        use_pickled: bool = True
) -> list[bool]:
    pass


if __name__ == '__main__':
    main()
