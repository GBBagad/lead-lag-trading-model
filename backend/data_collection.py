import requests
import pandas as pd

API_KEY = "0UW6OMGc1q1XypZbSEcWtLOFv5KuSVqi"

def get_polygon_data(ticker):

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2025-01-01/2025-09-01?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        print("API error:", data)
        return None

    df = pd.DataFrame(data["results"])

    df["date"] = pd.to_datetime(df["t"], unit="ms")

    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })

    df = df[["date","open","high","low","close","volume"]]

    return df


print("Downloading leader data...")
leader = get_polygon_data("SPY")

print("Downloading lagger data...")
lagger = get_polygon_data("QQQ")

if leader is not None and lagger is not None:

    leader.to_csv("data/leader_data.csv", index=False)
    lagger.to_csv("data/lagger_data.csv", index=False)

    print("Data downloaded successfully")
    print("Leader rows:", len(leader))
    print("Lagger rows:", len(lagger))

   