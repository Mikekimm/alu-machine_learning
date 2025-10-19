#!/usr/bin/env python3

""" Return list of ships"""

import requests


def availableShips(passengerCount):
    """ Return list of ships

    Args:
        passengerCount (int): number of ships
    """

    base_url = "https://swapi-api.alx-tools.com/api"

    res = requests.get(f'{base_url}/starships')

    output = []
    while res.status_code == 200:
        res = res.json()
        for ship in res['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    output.append(ship['name'])
            except ValueError:
                pass
        try:
            res = requests.get(res['next'])
        except Exception:
            break
    return output
