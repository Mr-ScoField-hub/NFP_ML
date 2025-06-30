import pandas as pd
import yfinance as yf

def prepare_nfp_dataset():
    # Load employment data
    emp = pd.read_csv("../data/employment.csv", parse_dates=["observation_date"])
    emp.rename(columns={"observation_date": "Date", "PAYEMS": "Employment"}, inplace=True)

    # Fetch EUR/USD data
    start_date = emp['Date'].min()
    end_date = emp['Date'].max()

    eurusd = yf.download('EURUSD=X', start=start_date, end=end_date, interval='1mo')

    # Flatten MultiIndex columns if necessary
    if isinstance(eurusd.columns, pd.MultiIndex):
        eurusd.columns = eurusd.columns.get_level_values(0)

    print("\n✅ Downloaded EUR/USD data columns:", eurusd.columns)
    print(eurusd.head())

    # Select relevant price column
    if 'Adj Close' in eurusd.columns:
        eurusd_monthly = eurusd[['Adj Close']].reset_index()
        eurusd_monthly.rename(columns={'Adj Close': 'Close'}, inplace=True)
    elif 'Close' in eurusd.columns:
        eurusd_monthly = eurusd[['Close']].reset_index()
    else:
        print("❌ No 'Close' or 'Adj Close' column found.")
        return

    # Merge on Date
    df = pd.merge(emp, eurusd_monthly, on='Date', how='inner')

    # Calculate percentage changes
    df['Emp_Pct_Change'] = df['Employment'].pct_change()
    df['Price_Pct_Change'] = df['Close'].pct_change()

    # Create target: 1 if next month’s price goes up, else 0
    df['Target'] = (df['Price_Pct_Change'].shift(-1) > 0).astype(int)

    # Drop NaN rows from pct_change and shift
    df_clean = df.dropna().reset_index(drop=True)

    print("\n Prepared dataset preview:")
    print(df_clean.head())

    # Save cleaned dataset
    df_clean.to_csv("../data/prepared_nfp_dataset.csv", index=False)
    print("\n Dataset saved as '../data/prepared_nfp_dataset.csv'")

if __name__ == "__main__":
    prepare_nfp_dataset()
