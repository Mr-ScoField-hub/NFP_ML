import yfinance as yf
import pandas as pd

# Load NFP Data
nfp_df = pd.read_csv("../data/nfp_data.csv")

# Clean NFP Data
nfp_df.columns = nfp_df.columns.str.strip()  # Remove extra spaces
nfp_df["Date"] = pd.to_datetime(nfp_df["Date"])

# Keep only necessary columns
nfp_df = nfp_df[["Date", "Actual", "Forecast", "Previous"]]

# Fetch EUR/USD hourly price data from 2019 to 2024
forex_df = yf.download('EURUSD=X', start='2019-01-01', end='2024-06-29', interval='1h')

# Reset index to have DateTime as a column
forex_df.reset_index(inplace=True)

# Clean price data
forex_df["Datetime"] = pd.to_datetime(forex_df["Datetime"])

# Extract NFP event dates (we'll match prices on NFP release times: 14:30 SAST / 12:30 UTC)
nfp_times = nfp_df["Date"] + pd.Timedelta(hours=12, minutes=30)

# Filter Forex prices within ±12 hours of each NFP event
forex_events = forex_df[forex_df["Datetime"].isin(nfp_times)]

# Merge data (later you can extend to add price movement before/after)
merged_df = pd.merge(nfp_df, forex_events, left_on="Date", right_on=forex_events["Datetime"].dt.date)

# Save cleaned merged data
merged_df.to_csv("data/merged_nfp_forex.csv", index=False)

print(" Cleaned NFP and Forex data saved to data/merged_nfp_forex.csv")
