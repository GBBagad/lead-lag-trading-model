import requests
import pandas as pd

API_KEY = "YQcAWfMh5jO1aizfwTgEO2iB0IXE4D49"

def get_polygon_data(ticker):

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2022-01-01/2024-01-01?apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    results = data["results"]

    df = pd.DataFrame(results)

    df["date"] = pd.to_datetime(df["t"], unit="ms")

    df = df.rename(columns={
        "c": "close",
        "o": "open",
        "h": "high",
        "l": "low",
        "v": "volume"
    })

    return df


leader = get_polygon_data("AAPL")
lagger = get_polygon_data("MSFT")

leader.to_csv("data/leader_data.csv", index=False)
lagger.to_csv("data/lagger_data.csv", index=False)

print("Data downloaded successfully")