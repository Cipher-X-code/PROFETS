"""Download CSVs from football-data.co.uk for selected seasons and leagues.

This script downloads season CSVs and stores them in `predictor/data/`.
Example sources: https://www.football-data.co.uk/englandm.php and season CSV URLs.
"""
import os
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Example: download Premier League (E0) historical seasons from football-data.co.uk
# The filenames use two-digit seasons like 2223 for 2022/23. Adjust as needed.
SEASONS = ["2223", "2122", "2021", "2019"]
LEAGUE = "E0"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download(seasons=SEASONS, league=LEAGUE):
    for s in seasons:
        url = BASE_URL.format(season=s, league=league)
        out = DATA_DIR / f"{league}-{s}.csv"
        if out.exists():
            print(f"Skipping existing {out}")
            continue
        print(f"Downloading {url} -> {out}")
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            out.write_bytes(r.content)
        else:
            print(f"Failed to download {url}: {r.status_code}")


if __name__ == "__main__":
    download()
