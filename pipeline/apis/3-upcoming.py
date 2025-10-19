#!/usr/bin/env python3
"""Display the next upcoming SpaceX launch.

Format:
<launch name> (<date in local time>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests
from datetime import datetime
import pytz


def get_upcoming_launch():
    """Fetch and print the soonest upcoming SpaceX launch."""
    base_url = "https://api.spacexdata.com/v4"

    # Fetch upcoming launches
    response = requests.get(f"{base_url}/launches/upcoming")
    response.raise_for_status()
    launches = response.json()

    if not launches:
        print("No upcoming launches found")
        return

    # Sort launches by date_unix ascending
    launches.sort(key=lambda launch: launch['date_unix'])

    # Pick the earliest launch
    launch = launches[0]

    # Get local datetime from UTC unix timestamp
    launch_time_utc = datetime.utcfromtimestamp(launch['date_unix']).replace(tzinfo=pytz.utc)
    launch_time_local = launch_time_utc.astimezone()

    # Get rocket info
    rocket_response = requests.get(f"{base_url}/rockets/{launch['rocket']}")
    rocket_response.raise_for_status()
    rocket = rocket_response.json()

    # Get launchpad info
    launchpad_response = requests.get(f"{base_url}/launchpads/{launch['launchpad']}")
    launchpad_response.raise_for_status()
    launchpad = launchpad_response.json()

    # Format output string
    output = (
        f"{launch['name']} "
        f"({launch_time_local.isoformat()}) "
        f"{rocket['name']} - "
        f"{launchpad['name']} ({launchpad['locality']})"
    )

    print(output)


if __name__ == "__main__":
    get_upcoming_launch()
