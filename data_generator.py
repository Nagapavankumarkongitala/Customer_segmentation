import pandas as pd
import numpy as np
from datetime import datetime

# Single-file sample dataset (transaction level) consumed by
# data_preprocessing.process_single_dataset. One row per transaction,
# carrying customer attributes so no second customers.csv is needed.
SAMPLE_FILE = "sample_transactions.csv"
# Initial data file the dashboard (app.py) loads on startup.
PROCESSED_FILE = "processed_customer_data.csv"

GEO_PROFILES = [
    # (Country, Region, State, City, weight)
    ("USA", "West", "CA", "Los Angeles", 0.22),
    ("USA", "Northeast", "NY", "New York", 0.18),
    ("USA", "South", "TX", "Houston", 0.15),
    ("USA", "Southeast", "FL", "Miami", 0.10),
    ("USA", "Midwest", "IL", "Chicago", 0.10),
    ("USA", "West", "WA", "Seattle", 0.05),
    ("Canada", "Ontario", "ON", "Toronto", 0.05),
    ("UK", "England", "ENG", "London", 0.05),
    ("Germany", "Bavaria", "BY", "Munich", 0.04),
    ("Australia", "NSW", "NSW", "Sydney", 0.03),
    ("India", "Maharashtra", "MH", "Mumbai", 0.03),
]

# Seasonal demand shape (Jan..Dec): holiday peak + summer lift.
MONTH_WEIGHTS = np.array([0.07, 0.065, 0.075, 0.075, 0.08, 0.085,
                          0.09, 0.09, 0.08, 0.085, 0.10, 0.105])
MONTH_WEIGHTS = MONTH_WEIGHTS / MONTH_WEIGHTS.sum()


def _pick_geo(rng, n):
    idx = rng.choice(len(GEO_PROFILES), size=n,
                     p=np.array([g[4] for g in GEO_PROFILES]))
    return pd.DataFrame([GEO_PROFILES[i][:4] for i in idx],
                        columns=["Country", "Region", "State", "City"])


def generate_synthetic_data(num_customers=1000, num_transactions=15000,
                            years=(2022, 2023, 2024), seed=42):
    """Generate ONE transaction-level DataFrame spanning multiple years.

    Yearly volume grows year-over-year so YoY sales/segment charts show
    realistic movement. Returns the single sample DataFrame.
    """
    rng = np.random.default_rng(seed)

    customer_ids = np.arange(1, num_customers + 1)
    ages = rng.integers(18, 70, num_customers)
    genders = rng.choice(["Male", "Female", "Other"], num_customers,
                         p=[0.45, 0.50, 0.05])
    geo = _pick_geo(rng, num_customers)

    # Year weights grow YoY (0.25 / 0.33 / 0.42 normalized).
    year_raw = np.linspace(0.7, 1.3, len(years))
    year_p = year_raw / year_raw.sum()
    tx_years = rng.choice(years, size=num_transactions, p=year_p)
    tx_months = np.array([rng.choice(np.arange(1, 13), p=MONTH_WEIGHTS)
                          for _ in range(num_transactions)])
    # Day clamped to 28 to stay valid for every month.
    tx_days = rng.integers(1, 29, num_transactions)
    tx_dates = pd.to_datetime({"year": tx_years, "month": tx_months,
                               "day": tx_days}).dt.strftime("%Y-%m-%d")

    # Spend drifts up slightly each year (inflation + growth).
    year_factor = {y: 1.0 + 0.06 * (y - years[0]) for y in years}
    base = rng.uniform(10, 500, num_transactions)
    amounts = np.round(base * np.array([year_factor[y] for y in tx_years]), 2)

    tx_customers = rng.integers(1, num_customers + 1, num_transactions)
    df = pd.DataFrame({
        "TransactionID": np.arange(1, num_transactions + 1),
        "CustomerID": tx_customers,
        "TransactionDate": list(tx_dates),
        "Amount": amounts,
    })
    cust = pd.DataFrame({
        "CustomerID": customer_ids, "Age": ages, "Gender": genders,
    })
    cust = pd.concat([cust, geo], axis=1)
    df = df.merge(cust, on="CustomerID", how="left")

    # Churn correlates with recency: dormant customers churn more.
    last_seen = df.groupby("CustomerID")["TransactionDate"].max()
    snapshot = pd.to_datetime(df["TransactionDate"]).max()
    recency = (snapshot - pd.to_datetime(last_seen)).dt.days
    churn_p = recency.apply(lambda d: 0.45 if d > 180 else 0.12)
    churn_map = (rng.random(len(churn_p)) < churn_p.values).astype(int)
    churn_map = pd.Series(churn_map, index=churn_p.index)
    df["HasChurned"] = df["CustomerID"].map(churn_map).astype(int)

    return df[["TransactionID", "CustomerID", "TransactionDate", "Amount",
               "Age", "Gender", "State", "Region", "Country", "City",
               "HasChurned"]]


if __name__ == "__main__":
    sample_df = generate_synthetic_data()

    # 1. Single sample data file.
    sample_df.to_csv(SAMPLE_FILE, index=False)
    print(f"Single sample file saved to {SAMPLE_FILE} "
          f"({len(sample_df):,} transactions, "
          f"{sample_df['CustomerID'].nunique():,} customers, "
          f"years {sorted(pd.to_datetime(sample_df['TransactionDate']).dt.year.unique())})")

    # 2. Connect to the dashboard's initial data file via the pipeline.
    from data_preprocessing import (
        process_single_dataset, analyze_yearly_sales,
        analyze_yearly_segment_trends,
    )
    processed_df, _ = process_single_dataset(sample_df)
    processed_df.to_csv(PROCESSED_FILE, index=False)
    print(f"Initial dashboard data saved to {PROCESSED_FILE} "
          f"({len(processed_df):,} customer profiles)")

    yearly = analyze_yearly_sales(sample_df)
    yearly.to_csv("yearly_sales.csv", index=False)
    trends = analyze_yearly_segment_trends(sample_df)
    trends.to_csv("yearly_segment_trends.csv", index=False)
    print("Yearly aggregates saved to yearly_sales.csv and "
          "yearly_segment_trends.csv")
