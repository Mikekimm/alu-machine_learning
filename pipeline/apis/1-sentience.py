#!/usr/bin/env python3
""" Return list of ships"""

import requests


def sentientPlanets():
    """Return list of names of the home planets of all sentient species."""
    base_url = "https://swapi-api.alx-tools.com/api"
    url = "{}/species/".format(base_url)

    output = []
    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break

        data = res.json()
        for species in data['results']:
            if species['designation'] == "sentient" or \
                    species['classification'] == "sentient":

                if not species['homeworld']:
                    continue

                planet = requests.get(species['homeworld'])
                if planet.status_code == 200:
                    output.append(planet.json().get('name'))

        url = data.get('next')

    return output
