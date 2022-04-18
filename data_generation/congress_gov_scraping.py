from __future__ import annotations

import os
import pickle
import random
import time

from bs4 import BeautifulSoup as bs
import requests

PICKLED_DATA = '../data/pickled_data'
JSON_FILES = '../data/json_files'
BULK_DATA_DIR = '../data/propublica_bulk_bill_data'

# https://www.govinfo.gov/help/bills
suffixes_of_interest = {
    'enr': 'Enrolled',  # Final bill passed by both bodies
    'ih': 'Introduced (House)',
    'is': 'Introduced (Senate)'
}
congresses = tuple(range(107, 116))
bodies = ('senate', 'house')
bodies_short = ('s', 'hr')
versions = ('Introduced in Senate', 'Introduced in House')
versions_short = ('is', 'ih')

# Bill name format:
# f"BILLS-{congress_number}{bill_category}{bill_number}{bill_version}


def main():
    # print(
    #     get_bill_text(
    #         107, 'senate',
    #         1000, 'is', 'Introduced in Senate',
    #         use_pickled=False)
    # )

    for congress in congresses:
        get_bill_texts(congress)


def get_bill_text(
        congress: int,
        body: str,
        bill_number: int,
        version: str,
        version_short: str,
        use_pickled: bool = True
) -> str | bool:
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_bill_text_{congress}_{body}_{bill_number}_"
        f"{version}_{version_short}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        url = (
                f"https://www.congress.gov/bill/"
                f"{congress}th-congress/{body}-bill/"
                f"{bill_number}/text/is?format=txt"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print("HTTP Request Failed")
            return False
        html = response.content.decode()
        soup = bs(html, 'html.parser')
        try:
            bill_type = soup.find_all(class_='currentVersion')[0]
        except:
            return False
        bill_type = str(bill_type.contents[2].contents[0])
        if not bill_type.startswith(version):
            return False
        data = str(soup.find(id='billTextContainer').contents[0])

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


def get_bill_texts(
        congress: int,
        bodies: tuple[str, str] = bodies,
        bodies_short: tuple[str, str] = bodies_short,
        versions: tuple[str, str] = versions,
        versions_short: tuple[str, str] = versions_short,
        wait_time: int = 3,
        use_pickled: bool = True
) -> dict[tuple[int, str, int], str]:
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_bill_texts_"
        f"{congress}_"
        f"{bodies}_{bodies_short}_"
        f"{versions}_{versions_short}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = dict()

        for body, body_s, version, v_short in zip(
            bodies, bodies_short, versions, versions_short
        ):
            target_directory = os.path.join(
                BULK_DATA_DIR, f'{congress}', 'bills', f'{body_s}'
            )
            num_bills = len(os.listdir(target_directory))
            for bill_number in range(1, num_bills+1):
                bill_text = get_bill_text(
                    congress, body, bill_number,
                    version, v_short
                )
                if bill_text:
                    data[(congress, body_s, bill_number)] = bill_text
                    print(f"Acquired {body} bill {bill_number}:")
                    print(f"\t{bill_text[:200]}")
                else:
                    print(f"Failed to acquire {body} bill {bill_number}")
                time.sleep(random.uniform(1, wait_time))

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


if __name__ == '__main__':
    main()
