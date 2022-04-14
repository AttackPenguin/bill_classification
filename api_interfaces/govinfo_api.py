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

PICKLED_DATA = 'pickled_data'
JSON_FILES = 'json_files'

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


def get_members_as_json(
        congress: int,
        chamber: str,
        use_pickled: bool = True,
        save_json: bool = False
):
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_members_as_json_{congress}_{chamber}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        url = (
                api_root +
                f'{congress}/' +
                chamber + '/' +
                'members.json'
        )
        response = requests.get(url, headers=headers)
        data = response.json()

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    if save_json:
        json_file_path = os.path.join(
            JSON_FILES,
            f"get_members_as_json_{congress}_{chamber}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    return data


def get_specific_member_as_json(
        member_id: str,
        use_pickled: bool = True,
        save_json: bool = False
):
    file_path = os.path.join(
        PICKLED_DATA,
        f"get_specific_member_as_json_{member_id}.pickle"
    )
    if use_pickled and os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        url = (
                api_root +
                f'members/{member_id}.json'
        )
        response = requests.get(url, headers=headers)
        data = response.json()

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    if save_json:
        json_file_path = os.path.join(
            JSON_FILES,
            f"get_specific_member_as_json_{member_id}.json"
        )
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    return data


if __name__ == '__main__':
    main()
