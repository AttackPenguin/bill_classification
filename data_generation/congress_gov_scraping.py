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
chambers = ('senate', 'house')
chambers_short = ('s', 'hr')
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
        get_bill_texts_by_congress(congress)


def get_bill_text(
        congress: int,
        chamber: str,
        bill_number: int,
        version: str,
        version_short: str,
        use_pickled: bool = True,
        wait_time: int | float | None = None
) -> str | bool:
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_bill_text_{congress}_{chamber}_{bill_number}_"
        f"{version}_{version_short}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        if (congress, chamber, bill_number) in [
            (111, 'house', 3619),
            (113, 'house', 3979)
        ]:
            return False
        url = (
                f"https://www.congress.gov/bill/"
                f"{congress}th-congress/{chamber}-bill/"
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

        if wait_time:
            time.sleep(random.uniform(1.0/wait_time, wait_time))

    return data


def get_bill_texts_by_congress(
        congress: int,
        chambers: tuple[str, str] = chambers,
        chambers_short: tuple[str, str] = chambers_short,
        versions: tuple[str, str] = versions,
        versions_short: tuple[str, str] = versions_short,
        wait_time: int | float = 1,
        use_pickled: bool = True
) -> dict[tuple[int, str, int], str]:
    pickled_file_path = os.path.join(
        PICKLED_DATA,
        f"get_bill_texts_"
        f"{congress}_"
        f"{chambers}_{chambers_short}_"
        f"{versions}_{versions_short}.pickle"
    )
    if use_pickled and os.path.exists(pickled_file_path):
        with open(pickled_file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        data = dict()

        for body, body_s, version, v_short in zip(
            chambers, chambers_short, versions, versions_short
        ):
            target_directory = os.path.join(
                BULK_DATA_DIR, f'{congress}', 'bills', f'{body_s}'
            )
            num_bills = len(os.listdir(target_directory))
            for bill_number in range(1, num_bills+1):
                bill_text = get_bill_text(
                    congress, body, bill_number,
                    version, v_short, wait_time=wait_time
                )
                if bill_text:
                    data[(congress, body_s, bill_number)] = bill_text
                    print(
                        f"Acquired congress "
                        f"{congress} {body} bill {bill_number}:"
                    )
                else:
                    print(f"Failed to acquire {body} bill {bill_number}")

        with open(pickled_file_path, 'wb') as file:
            pickle.dump(data, file)

    return data


if __name__ == '__main__':
    main()
