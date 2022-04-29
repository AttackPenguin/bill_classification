from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import requests

api_version = 'v1'
api_root = f"https://api.propublica.org/congress/{api_version}/"
api_key = 'k7VvhUnM82NLeARf7jk6uy0VPjLmEyfgFijVKrt0'
api_documentation = "https://projects.propublica.org/api-docs/congress-api/"
daily_request_limit = 5000
headers = {
    'x-api-key': api_key
}

PICKLED_DATA = '../data/pickled_data/prorepublica_api'
JSON_FILES = '../data/json_files/prorepublica_api'


name_map = {
    ('Mike', 'Crapo'): ('Michael', 'Crapo'),
    ('Chuck', 'Grassley'): ('Charles', 'Grassley'),
    ('Russell', 'Feingold'): ('Russ', 'Feingold'),
    ('Dianne', 'Feinstein'): ('Charles', 'Grassley'),
    ('Rick', 'Santorum'): ('Richard', 'Santorum'),
    ('Mike', 'DeWine'): ('Michael', 'DeWine'),
    ('E', 'Nelson'): ('Ben', 'Nelson'),
    ('Jose', 'Serrano'): ('José', 'Serrano'),
    ('Doug', 'Bereuter'): ('Douglas', 'Bereuter'),
    ('Sheila', 'Jackson-Lee'): ('Sheila', 'Jackson Lee'),
    ('Jo', 'Emerson'): ('Jo Ann', 'Emerson'),
    ('J', 'Hayworth'): ('John', 'Hayworth'),
    ('Bill,', 'Pascrell'): ('Bill', 'Pascrell'),
    ('Thomas', 'Petri'): ('Tom', 'Petri'),
    ('Joseph', 'Pitts'): ('Joe', 'Pitts'),
    ('Jim', 'Saxton'): ('H. James', 'Saxton'),
    ('Mac', 'Thornberry'): ('William', 'Thornberry'),
    ('Tom', 'Davis'): ('Thomas', 'Davis'),
    ('Charles', 'Gonzalez'): ('Charlie', 'Gonzalez'),
    ('Tom', 'Osborne'): ('Thomas', 'Osborne'),
    ('Dennis', 'Rehberg'): ('Denny', 'Rehberg'),
    ('Nick', 'Lampson'): ('Nicholas', 'Lampson'),
    ('James', 'McGovern'): ('Jim', 'McGovern'),
    ('Michael', 'Doyle'): ('Mike', 'Doyle'),
    ('Luis', 'Gutierrez'): ('Luis', 'Gutiérrez'),
    ('Jim', 'Kolbe'): ('James', 'Kolbe'),
    ('E', 'Shaw'): ('E. Clay', 'Shaw'),
    ('Loretta', 'Sanchez'): ('Loretta', 'Sánchez'),
    ('William', 'Delahunt'): ('Bill', 'Delahunt'),
    ('Jim', 'Gibbons'): ('James', 'Gibbons'),
    ('Sheila', 'JacksonLee'): ('Sheila', 'Jackson Lee'),
    ('Ileana', 'RosLehtinen'): ('Ileana', 'Ros-Lehtinen'),
    ('Juanita', 'MillenderMcDonald'): ('Juanita', 'Millender-McDonald'),
    ('Fortney', 'Stark'): ('Pete', 'Stark'),
    ('Amo', 'Houghton'): ('Amory', 'Houghton'),
    ('Michael', 'Simpson'): ('Mike', 'Simpson'),
    ('Dave', 'Weldon'): ('David', 'Weldon'),
    ('Ernie', 'Fletcher'): ('Ernest', 'Fletcher'),
    ('Steve', 'Chabot'): ('Steven', 'Chabot'),
    ('Mac', 'Collins'): ('Michael', 'Collins'),
    ('Rob', 'Simmons'): ('Robert', 'Simmons'),
    ('Ed', 'Whitfield'): ('Edward', 'Whitfield'),
    ('Joe', 'Knollenberg'): ('Joseph', 'Knollenberg'),
    ('Lincoln', 'DiazBalart'): ('Lincoln', 'Diaz-Balart'),
    ('Jim', 'Ramstad'): ('James', 'Ramstad'),
    ('Mack', 'Bono'): ('Mary', 'Bono Mack'),
    ('Janice', 'Schakowsky'): ('Jan', 'Schakowsky'),
    ('Jo', 'Davis'): ('Jo Ann', 'Davis'),
    ('Bob', 'Goodlatte'): ('Robert', 'Goodlatte'),
    ('J', 'Watts'): ('J.C.', 'Watts'),
    ('Jerry', 'Weller'): ('Gerald', 'Weller'),
    ('James', 'Langevin'): ('Jim', 'Langevin'),
    ('W', 'Akin'): ('Todd', 'Akin'),
    ('Ken', 'Lucas'): ('Kenneth', 'Lucas'),
    ('Lucille', 'RoybalAllard'): ('Lucille', 'Roybal-Allard'),
    ('F', 'Sensenbrenner'): ('F.', 'Sensenbrenner'),
    ('Jim', 'Nussle'): ('James', 'Nussle'),
    ('Nydia', 'Velazquez'): ('Nydia', 'Velázquez'),
    ('Mike', 'Ferguson'): ('Michael', 'Ferguson'),
    ('C', 'Otter'): ('C.L.', 'Otter'),
    ('John', 'Moakley'): ('Joe', 'Moakley'),
    ('Curt', 'Weldon'): ('W. Curtis', 'Weldon'),
    ('Chris', 'Cannon'): ('Christopher', 'Cannon'),
    ('Tom', 'DeLay'): ('Thomas', 'DeLay'),
    ('Patrick', 'Tiberi'): ('Pat', 'Tiberi'),
    ('Ruben', 'Hinojosa'): ('Rubén', 'Hinojosa'),
    ('W', 'Tauzin'): ('William', 'Tauzin'),
    ('J', 'Forbes'): ('J.', 'Forbes'),
    ('Anibal', 'AcevedoVila'): ('An', 'Acevedo-Vila'),
    ('CW', 'Young'): ('Don', 'Young'),
    ('Wm', 'Clay'): ('William', 'Clay'),
    ('Gil', 'Gutknecht'): ('Gilbert', 'Gutknecht'),
    ('Jim', 'Turner'): ('James', 'Turner'),
    ('Don', 'Sherwood'): ('Donald', 'Sherwood'),
    ('Jim', 'Talent'): ('James', 'Talent'),
    ('J', 'Hastert'): ('J. Dennis', 'Hastert'),
    ('Raul', 'Grijalva'): ('Raúl', 'Grijalva'),
    ('Timothy', 'Ryan'): ('Tim', 'Ryan'),
    ('J', 'Barrett'): ('J.', 'Barrett'),
    ('Ginny', 'BrownWaite'): ('Ginny', 'Brown-Waite'),
    ('Hollen', 'Van'): ('Chris', 'Van Hollen'),
    ('Linda', 'Sanchez'): ('Linda', 'Sánchez'),
    ('Stevan', 'Pearce'): ('Steve', 'Pearce'),
    ('Mario', 'DiazBalart'): ('Mario', 'Diaz-Balart'),
    ('C', 'Ruppersberger'): ('C.A. Dutch', 'Ruppersberger'),
    ('Jim', 'Davis'): ('James', 'Davis'),
    ('Stephanie', 'Herseth'): ('Stephanie', 'Herseth Sandlin'),
    ('G', 'Butterfield'): ('G.', 'Butterfield'),
    ('Rodgers', 'McMorris'): ('Cathy', 'McMorris Rodgers'),
    ('David', 'Reichert'): ('Dave', 'Reichert'),
    ('Daniel', 'Lungren'): ('Dan', 'Lungren'),
    ('Charles', 'Dent'): ('Charlie', 'Dent'),
    ('Schultz', 'Wasserman'): ('Debbie', 'Wasserman Schultz'),
    ('K', 'Conaway'): ('K.', 'Conaway'),
    ('Robert', 'Casey'): ('Bob', 'Casey'),
    ('Timothy', 'Walz'): ('Tim', 'Walz'),
    ('Henry', 'Johnson'): ('Hank', 'Johnson'),
    ('Charles', 'Wilson'): ('Charlie', 'Wilson'),
    ('Sandlin', 'Herseth'): ('Stephanie', 'Herseth Sandlin'),
    ('Timothy', 'Walberg'): ('Tim', 'Walberg'),
    ('Zachary', 'Space'): ('Zack', 'Space'),
    ('Carol', 'SheaPorter'): ('Carol', 'Shea-Porter'),
    ('David', 'Loebsack'): ('Dave', 'Loebsack'),
    ('Andre', 'Carson'): ('André', 'Carson'),
    ('James', 'Risch'): ('Jim', 'Risch'),
    ('Walter', 'Minnick'): ('Walt', 'Minnick'),
    ('Ben', 'Lujan'): ('Ben', 'Luján'),
    ('Thomas', 'Rooney'): ('Tom', 'Rooney'),
    ('Thomas', 'Perriello'): ('Tom', 'Perriello'),
    ('Deborah', 'Halvorson'): ('Debbie', 'Halvorson'),
    ('David', 'Roe'): ('Phil', 'Roe'),
    ('James', 'Himes'): ('Jim', 'Himes'),
    ('Daniel', 'Maffei'): ('Dan', 'Maffei'),
    ('Kathleen', 'Dahlkemper'): ('Kathy', 'Dahlkemper'),
    ('Mary', 'Kilroy'): ('Mary Jo', 'Kilroy'),
    ('William', 'Owens'): ('Bill', 'Owens'),
    ('Lee', 'Jackson'): ('Sheila', 'Jackson Lee'),
    ('Theodore', 'Deutch'): ('Ted', 'Deutch'),
    ('Pat', 'Toomey'): ('Patrick', 'Toomey'),
    ('Michael', 'Grimm'): ('Mike', 'Grimm'),
    ('H', 'Griffith'): ('Morgan', 'Griffith'),
    ('C', 'Young'): ('C. W. Bill', 'Young'),
    ('Benjamin', 'Quayle'): ('Ben', 'Quayle'),
    ('Raul', 'Labrador'): ('Raúl', 'Labrador'),
    ('Jeff', 'Denham'): ('Jeffrey', 'Denham'),
    ('Sandy', 'Adams'): ('Sandra', 'Adams'),
    ('Beutler', 'Herrera'): ('Jaime', 'Herrera Beutler'),
    ('William', 'Keating'): ('Bill', 'Keating'),
    ('Jeff', 'Duncan'): ('Jeffrey', 'Duncan'),
    ('E', 'Rigell'): ('Scott', 'Rigell'),
    ('Jeffrey', 'Landry'): ('Jeff', 'Landry'),
    ('Christopher', 'Gibson'): ('Chris', 'Gibson'),
    ('Joseph', 'Heck'): ('Joe', 'Heck'),
    ('James', 'Renacci'): ('Jim', 'Renacci'),
    ('Charles', 'Fleischmann'): ('Chuck', 'Fleischmann'),
    ('Ann', 'Buerkle'): ('Ann Marie', 'Buerkle'),
    ('Robert', 'Schilling'): ('Bobby', 'Schilling'),
    ('Eric', 'Crawford'): ('Rick', 'Crawford'),
    ('Patrick', 'Meehan'): ('Pat', 'Meehan'),
    ('Kathleen', 'Hochul'): ('Kathy', 'Hochul'),
    ('Robert', 'Turner'): ('Bob', 'Turner'),
    ('David', 'Joyce'): ('Dave', 'Joyce'),
    ('William', 'Enyart'): ('Bill', 'Enyart'),
    ('McLeod', 'Negrete'): ('Gloria', 'Negrete McLeod'),
    ('Beto', 'ORourke'): ('Beto', "O'Rourke"),
    ('Joaquin', 'Castro'): ('Joaquín', 'Castro'),
    ('Daniel', 'Kildee'): ('Dan', 'Kildee'),
    ('Richard', 'Nolan'): ('Rick', 'Nolan'),
    ('Grisham', 'Lujan'): ('Ben', 'Luján'),
    ('Tony', 'Cardenas'): ('Tony', 'Cárdenas'),
    ('Bradley', 'Schneider'): ('Brad', 'Schneider'),
    ('Thomas', 'MacArthur'): ('Tom', 'MacArthur'),
    ('Alexander', 'Mooney'): ('Alex', 'Mooney'),
    ('Stephen', 'Knight'): ('Steve', 'Knight'),
    ('Aumua', 'Radewagen'): ('Amata', 'Radewagen'),
    ('Earl', 'Carter'): ('Buddy', 'Carter'),
    ('Coleman', 'Watson'): ('Bonnie', 'Watson Coleman'),
    ('Dave', 'Brat'): ('David', 'Brat'),
    ('J', 'Hill'): ('French', 'Hill'),
    ('David', 'Trott'): ('Dave', 'Trott'),
    ('Masto', 'Cortez'): ('Catherine', 'Cortez Masto'),
    ('Cindy', 'HydeSmith'): ('Cindy', 'Hyde-Smith'),
    ('Jenniffer', 'GonzalezColon'): ('Jenniffer', 'González-Colón'),
    ('J', 'Correa'): ('J.', 'Correa'),
    ('Chris', 'Cannon'): ('Christopher', 'Cannon'),
    ('Tom', 'OHalleran'): ('Tom', "O'Halleran"),
    ('A', 'Ferguson'): ('A.', 'Ferguson'),
    ('Nanette', 'Barragan'): ('Nanette', 'Barragán'),
    ('Rochester', 'Blunt'): ('Lisa', 'Blunt Rochester'),
    ('A', 'McEachin'): ('A.', 'McEachin'),
    ('Gibbs', 'Sekula'): ('Shelley', 'Sekula-Gibbs'),
    ('Jeff', 'Chiesa'): ('Jeffrey', 'Chiesa'),
    ('Joseph', 'Morelle'): ('Joe', 'Morelle')

}


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
        "get_members_as_json",
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
) -> dict:
    file_path = os.path.join(
        PICKLED_DATA,
        "get_specific_member_as_json",
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


def get_member_party(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> str | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    # Sponsors are always in specified congress and chamber.
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    party = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            party = member['party']
            break
    if party is None:
        # Sometimes cosponsors are from previous or following congresses.
        other_members = list()
        for congress in [congress-1] + list(range(congress+1, 116)):
            data = get_members_as_json(congress, chamber, use_pickled)
            members_other = data['results'][0]['members']
            other_members.append(members_other)
            for member in members_other:
                if (member['first_name'].lower() == first_name.lower() and
                        member['last_name'].lower() == last_name.lower()):
                    party = member['party']
                    break
            if party is not None:
                break
    if party is None:
        # Apparently cosponsors can also be from other chamber?
        if chamber == 'house':
            chamber = 'senate'
        else:
            chamber = 'house'
        other_chambers = list()
        for congress in list(range(106, 116)):
            data = get_members_as_json(congress, chamber, use_pickled)
            members_other = data['results'][0]['members']
            other_chambers.append(members_other)
            for member in members_other:
                if (member['first_name'].lower() == first_name.lower() and
                        member['last_name'].lower() == last_name.lower()):
                    party = member['party']
                    break
            if party is not None:
                break
    if party is None:
        return np.NaN
    return party


def get_member_dob(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> pd.Timestamp | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    return pd.Timestamp(member['date_of_birth'])


def get_member_age(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        bill_date: pd.Timestamp,
        use_pickled: bool = True
) -> int | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    age = bill_date - pd.Timestamp(member['date_of_birth'])
    return int(age.total_seconds() / (365.25 * 24 * 60 * 60))


def get_member_body(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> str | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    if member['title'][0] == 'S':
        return 'senate'
    else:
        return 'house'


def get_member_gender(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> str | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    return member['gender']


def get_member_missed_votes_perc(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> float | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    try:
        return member['missed_votes_pct']
    except KeyError:
        return np.NaN


def get_member_votes_with_party_pct(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> float | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    try:
        return member['votes_with_party_pct']
    except KeyError:
        return np.NaN


def get_member_votes_against_party_pct(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> float | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    try:
        return member['votes_against_party_pct']
    except KeyError:
        return np.NaN


def get_member_pro_republica_id(
        congress: int,
        chamber: str,
        first_name: str,
        last_name: str,
        use_pickled: bool = True
) -> str | bool:
    if first_name is np.NaN or last_name is np.NaN:
        return np.NaN
    if (first_name, last_name) in name_map:
        (first_name, last_name) = \
            name_map[(first_name, last_name)]
    data = get_members_as_json(congress, chamber, use_pickled)
    members = data['results'][0]['members']
    member = None
    for member in members:
        if (member['first_name'].lower() == first_name.lower() and
                member['last_name'].lower() == last_name.lower()):
            break
    if member is None:
        return np.NaN
    return member['id']


if __name__ == '__main__':
    main()
