#!/usr/bin/env python3
"""Pipeline API"""

import requests


def main():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching launches data")
        return

    launches = response.json()
    rocket_dict = {}

    # Count launches per rocket
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            rocket_dict[rocket_id] = rocket_dict.get(rocket_id, 0) + 1

    # Sort rockets by launch count descending
    sorted_rockets = sorted(rocket_dict.items(), key=lambda kv: kv[1], reverse=True)

    for rocket_id, count in sorted_rockets:
        rurl = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
        rocket_response = requests.get(rurl)
        if rocket_response.status_code == 200:
            rocket_name = rocket_response.json().get("name", "Unknown")
            print(f"{rocket_name}: {count}")
        else:
            print(f"Rocket ID {rocket_id}: {count} (Name not found)")


if __name__ == '__main__':
    main()
