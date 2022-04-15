import json
import os
import pickle

import requests

api_version = 'v1'
api_root = f"https://api.propublica.org/congress/{api_version}/"
api_key = 'k7VvhUnM82NLeARf7jk6uy0VPjLmEyfgFijVKrt0'
api_documentation = "https://projects.propublica.org/api-docs/congress-api/"
daily_request_limit = 5000
headers = {
    'x-api-key': api_key
}

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'


def main():
    # Get basic data on all members of congress and chamber
    data = get_members_as_json(110, 'house', save_json=True)
    # print(data)

    # Get detailed data on specific congress person
    data = get_specific_member_as_json('K000388', save_json=True)
    print(data)


def get_members_as_json(
        congress: int,
        chamber: str,
        use_pickled: bool = True,
        save_json: bool = False
) -> dict:
    """

    :param congress: The number of the congress. e.g. 110.
    :param chamber: One of 'senate' or 'house'
    :param use_pickled: Whether or not to use pickled output of method with
    specified arguments if it exists.
    :param save_json: Whether or not to save the data as a json file.
    :return: A
    """
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
