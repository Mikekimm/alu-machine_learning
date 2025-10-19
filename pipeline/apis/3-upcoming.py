#!/usr/bin/env python3
"""Displays the next upcoming SpaceX launch."""

import requests
from datetime import datetime
import pytz


def get_upcoming_launch():
    base_url = "https://api.spacexdata.com/v4"

    # Get all upcoming launches
    launches_res = requests.get("{}/launches/upcoming".format(base_url))
    if launches_res.status_code != 200:
        return

    launches = launches_res.json()
    if not launches:
        return

    # Sort by date_unix (earliest first)
    launches.sort(key=lambda x: x['date_unix'])

    # Get the soonest launch
    next_launch = launches[0]

    launch_name = next_launch['name']
    launch_time_utc = datetime.fromtimestamp(next_launch['date_unix'], tz=pytz.utc)
    local_time = launch_time_utc.astimezone()  # Convert to local time

    rocket_id = next_launch['rocket']
    launchpad_id = next_launch['launchpad']

    # Get rocket name
    rocket_res = requests.get("{}/rockets/{}".format(base_url, rocket_id))
    rocket_name = rocket_res.json().get('name', 'Unknown')

    # Get launchpad info
    pad_res = requests.get("{}/launchpads/{}".format(base_url, launchpad_id))
    pad_info = pad_res.json()
    pad_name = pad_info.get('name', 'Unknown')
    pad_locality = pad_info.get('locality', 'Unknown')

    # Format output
    output = "{} ({}) {} - {} ({})".format(
        launch_name,
        local_time.isoformat(),
        rocket_name,
        pad_name,
        pad_locality
    )
    print(output)


if __name__ == "__main__":
    get_upcoming_launch()

