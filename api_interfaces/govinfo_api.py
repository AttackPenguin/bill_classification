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

# https://www.govinfo.gov/help/bills
suffixes_of_interest = {
    'enr': 'Enrolled',  # Final bill passed by both bodies
    'ih': 'Introduced (House)',
    'is': 'Introduced (Senate)'
}
congress_numbers = list(range(107, 116))
bill_categories = ['s', 'hr']

# Bill name format:
# f"BILLS-{congress_number}{bill_category}{bill_number}{suffix}


def main():
    get_bill_text(107, 'hr', 1000, 'ih')


def get_bill_text(
        congress: int,
        category: str,
        bill_number: int,
        suffix: str,
        data_type: str = 'htm',
        use_pickled: bool = True,
        save_txt: bool = False
):
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
    print(data)


if __name__ == '__main__':
    main()
