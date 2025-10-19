#!/usr/bin/env python3
"""Return user's location based on URL response."""

import requests
import sys
import time


def get_user_location(url):
    """Handles response and prints location or appropriate status."""
    try:
        res = requests.get(url)
    except requests.RequestException:
        print("Request failed")
        return

    if res.status_code == 403:
        rate_limit = int(res.headers.get('X-Ratelimit-Reset', 0))
        current_time = int(time.time())
        diff = (rate_limit - current_time) / 60
        print("Rate limit. Retry in {} min".format(int(round(diff))))
    elif res.status_code == 404:
        print("Not found")
    elif res.status_code == 200:
        data = res.json()
        print(data.get('location'))
    else:
        print("Unexpected status code:", res.status_code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./script.py <URL>")
        sys.exit(1)

    get_user_location(sys.argv[1])

