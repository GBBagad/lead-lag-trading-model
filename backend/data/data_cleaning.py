
import numpy as np

def compute_log_returns(series):
    return np.log(series / series.shift(1))
import pandas as pd
import numpy as np

# ✅ log returns
def compute_log_returns(series):
    return np.log(series / series.shift(1))


# ✅ RESAMPLE FUNCTION (MISSING HOTA 🔥)
def resample_data(df, timeframe):

    df.columns = df.columns.str.lower()

    # detect time column
    if 'datetime' in df.columns:
        time_col = 'datetime'
    elif 'date' in df.columns:
        time_col = 'date'
    elif 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        raise Exception(f"No datetime column found. Columns: {df.columns}")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled.reset_index() 
