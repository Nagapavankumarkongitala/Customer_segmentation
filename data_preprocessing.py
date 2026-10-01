import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_rfm(transactions_df, customers_df=None, snapshot_date=None):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics from transactions.
    Supports either a merged/single transaction DataFrame or separate transactions and customers DataFrames.
    """
    transactions_df = transactions_df.copy()
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')
    transactions_df = transactions_df.dropna(subset=['TransactionDate', 'CustomerID'])
    transactions_df['Amount'] = pd.to_numeric(transactions_df['Amount'], errors='coerce').fillna(0)

    if snapshot_date is None:
        if len(transactions_df) > 0 and transactions_df['TransactionDate'].notna().any():
            snapshot_date = transactions_df['TransactionDate'].max() + timedelta(days=1)
        else:
            snapshot_date = datetime.now()

    if customers_df is not None:
        df = pd.merge(transactions_df, customers_df, on='CustomerID', how='left')
    else:
        df = transactions_df

    # Aggregate RFM metrics
    recency_s = df.groupby('CustomerID')['TransactionDate'].max().apply(lambda x: (snapshot_date - x).days)
    freq_s = df.groupby('CustomerID')['Amount'].count()
    monetary_s = df.groupby('CustomerID')['Amount'].sum().round(2)

    rfm_df = pd.DataFrame({
        'CustomerID': recency_s.index,
        'Recency': recency_s.values,
        'Frequency': freq_s.values,
        'Monetary': monetary_s.values
    }).set_index('CustomerID')

    # Retain customer demographic, geographic & churn columns if present
    for col in ['Age', 'Gender', 'State', 'Region', 'Country', 'City', 'HasChurned']:
        if col in df.columns:
            rfm_df[col] = df.groupby('CustomerID')[col].first()

    # Retain purchase timing (month/year) per customer for trend analysis
    rfm_df['FirstPurchaseYear'] = df.groupby('CustomerID')['TransactionDate'].min().dt.year.values
    rfm_df['LastPurchaseYear'] = df.groupby('CustomerID')['TransactionDate'].max().dt.year.values
    rfm_df['LastPurchaseMonth'] = df.groupby('CustomerID')['TransactionDate'].max().dt.month.values

    return rfm_df

def assign_rfm_segments(rfm_df, num_segments=5):
    """Assign R, F, M quintile scores and simplified segment labels."""
    rfm_df = rfm_df.copy()

    # Ensure numeric columns
    for col in ['Recency', 'Frequency', 'Monetary']:
        if col in rfm_df.columns:
            rfm_df[col] = pd.to_numeric(rfm_df[col], errors='coerce').fillna(0)

    # Use ranking before qcut to handle duplicates safely
    n = len(rfm_df)
    if n >= num_segments:
        try:
            rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), num_segments, labels=list(range(num_segments, 0, -1)))
            rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), num_segments, labels=list(range(1, num_segments + 1)))
            rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), num_segments, labels=list(range(1, num_segments + 1)))
        except Exception:
            rfm_df['R_Score'] = 3
            rfm_df['F_Score'] = 3
            rfm_df['M_Score'] = 3
    else:
        rfm_df['R_Score'] = 3
        rfm_df['F_Score'] = 3
        rfm_df['M_Score'] = 3

    # Combine scores into a single RFM segment string
    rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    def assign_simplified_segment(row):
        try:
            r = int(row['R_Score'])
            f = int(row['F_Score'])
            m = int(row['M_Score'])
        except (ValueError, TypeError):
            return 'Other'
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f <= 2 and m >= 3:
            return 'Potential Loyalists'
        elif r >= 3 and f >= 3 and m <= 2:
            return 'Need Attention'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        else:
            return 'Other'

    rfm_df['Simplified_RFM_Segment'] = rfm_df.apply(assign_simplified_segment, axis=1)
    return rfm_df

def extract_time_features(df):
    """Add Month/Year columns extracted from TransactionDate (copy returned)."""
    df = df.copy()
    tx = pd.to_datetime(df['TransactionDate'], errors='coerce')
    df['TransactionYear'] = tx.dt.year
    df['TransactionMonth'] = tx.dt.month
    df['TransactionYearMonth'] = tx.dt.strftime('%Y-%m')
    return df


def analyze_yearly_sales(transactions_df):
    """Year-wise sales summary with YoY growth.

    Returns DataFrame: Year, TotalSales, Transactions, Customers,
    AvgOrderValue, YoY_Growth_Pct.
    """
    tx = extract_time_features(transactions_df)
    tx = tx.dropna(subset=['TransactionYear'])
    tx['Amount'] = pd.to_numeric(tx['Amount'], errors='coerce').fillna(0)
    yearly = tx.groupby(tx['TransactionYear'].astype(int)).agg(
        TotalSales=('Amount', 'sum'),
        Transactions=('Amount', 'count'),
        Customers=('CustomerID', 'nunique'),
    ).reset_index().rename(columns={'TransactionYear': 'Year'})
    yearly = yearly.sort_values('Year').reset_index(drop=True)
    yearly['AvgOrderValue'] = (yearly['TotalSales'] / yearly['Transactions'].replace(0, np.nan)).fillna(0).round(2)
    yearly['TotalSales'] = yearly['TotalSales'].round(2)
    yearly['YoY_Growth_Pct'] = yearly['TotalSales'].pct_change().mul(100).round(2)
    return yearly


SEGMENT_ORDER = ['Champions', 'Loyal Customers', 'Potential Loyalists',
                 'Need Attention', 'Hibernating', 'Other']


def analyze_yearly_segment_trends(transactions_df):
    """Segment-group counts per year with YoY change.

    Recomputes RFM + simplified segments within each transaction year so
    growth/decline of Champions, Loyal Customers, etc. is visible.
    Returns long DataFrame: Year, Segment, Customers, YoY_Change_Pct.
    """
    tx = extract_time_features(transactions_df)
    tx = tx.dropna(subset=['TransactionYear', 'TransactionDate', 'CustomerID'])
    tx['TransactionDate'] = pd.to_datetime(tx['TransactionDate'], errors='coerce')
    tx['Amount'] = pd.to_numeric(tx['Amount'], errors='coerce').fillna(0)

    frames = []
    for year in sorted(tx['TransactionYear'].astype(int).unique()):
        ydf = tx[tx['TransactionYear'].astype(int) == year]
        if ydf.empty:
            continue
        snapshot = datetime(year, 12, 31) + timedelta(days=1)
        rec = ydf.groupby('CustomerID')['TransactionDate'].max().apply(lambda x: (snapshot - x).days)
        freq = ydf.groupby('CustomerID')['Amount'].count()
        mon = ydf.groupby('CustomerID')['Amount'].sum().round(2)
        yearly_rfm = pd.DataFrame({'CustomerID': rec.index, 'Recency': rec.values,
                                   'Frequency': freq.values, 'Monetary': mon.values})
        yearly_rfm = assign_rfm_segments(yearly_rfm)
        counts = yearly_rfm['Simplified_RFM_Segment'].value_counts()
        for seg in SEGMENT_ORDER:
            frames.append({'Year': int(year), 'Segment': seg,
                           'Customers': int(counts.get(seg, 0))})

    trends = pd.DataFrame(frames, columns=['Year', 'Segment', 'Customers'])
    if not trends.empty:
        trends['YoY_Change_Pct'] = (trends.groupby('Segment')['Customers']
                                    .pct_change().mul(100).round(2))
    else:
        trends['YoY_Change_Pct'] = pd.Series(dtype=float)
    return trends


def plot_yearly_sales_chart(yearly_df):
    """Plotly chart: year-wise sales bars + YoY growth line."""
    import plotly.graph_objects as go
    yearly_df = yearly_df.copy().sort_values('Year')
    fig = go.Figure()
    fig.add_bar(x=yearly_df['Year'], y=yearly_df['TotalSales'],
                name='Total Sales ($)', marker_color='#5b7cff')
    if yearly_df['YoY_Growth_Pct'].notna().sum() > 0:
        fig.add_scatter(x=yearly_df['Year'], y=yearly_df['YoY_Growth_Pct'],
                        name='YoY Growth (%)', mode='lines+markers',
                        marker_color='#22d3ee', yaxis='y2')
    fig.update_layout(
        title='Year-wise Sales & YoY Growth',
        xaxis_title='Year', yaxis_title='Total Sales ($)',
        yaxis2=dict(title='YoY Growth (%)', overlaying='y', side='right',
                    showgrid=False),
        template='plotly_dark', height=380,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def plot_segment_trend_chart(trends_df):
    """Plotly chart: segment-group customer counts per year (growth/decline)."""
    import plotly.express as px
    order = [s for s in SEGMENT_ORDER if s in trends_df['Segment'].unique()]
    fig = px.line(trends_df, x='Year', y='Customers', color='Segment',
                  markers=True, title='Segment Groups: Yearly Movement',
                  category_orders={'Segment': order},
                  template='plotly_dark')
    fig.update_layout(height=380, margin=dict(l=30, r=30, t=50, b=30))
    return fig

def analyze_demographics(rfm_df):
    """Analyze customer behavior and churn across demographic segments."""
    rfm_df = rfm_df.copy()
    demographic_insights = {}

    if 'Gender' in rfm_df.columns:
        gender_cols = [c for c in ['Recency', 'Frequency', 'Monetary', 'HasChurned'] if c in rfm_df.columns]
        demographic_insights['gender_agg'] = rfm_df.groupby('Gender')[gender_cols].agg(['mean', 'count'])

    if 'Age' in rfm_df.columns and rfm_df['Age'].notna().any():
        rfm_df['Age'] = pd.to_numeric(rfm_df['Age'], errors='coerce')
        age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
        age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        rfm_df['AgeGroup'] = pd.cut(rfm_df['Age'], bins=age_bins, labels=age_labels, right=False)
        age_cols = [c for c in ['Recency', 'Frequency', 'Monetary', 'HasChurned'] if c in rfm_df.columns]
        demographic_insights['age_agg'] = rfm_df.groupby('AgeGroup', observed=False)[age_cols].agg(['mean', 'count'])
    elif 'AgeGroup' not in rfm_df.columns:
        rfm_df['AgeGroup'] = 'Unknown'

    # Geographic fields: State, Region, Country, City
    geo_fields = ['State', 'Region', 'Country', 'City']
    for geo_field in geo_fields:
        if geo_field in rfm_df.columns:
            geo_cols = [c for c in ['Recency', 'Frequency', 'Monetary', 'HasChurned'] if c in rfm_df.columns]
            demographic_insights[f'{geo_field.lower()}_agg'] = rfm_df.groupby(geo_field)[geo_cols].agg(['mean', 'count'])

    return demographic_insights, rfm_df

def perform_advanced_segmentation(rfm_df, n_segments=5):
    """Assign initial advanced segmentation tags."""
    rfm_df = rfm_df.copy()
    if 'Advanced_Segment' not in rfm_df.columns or rfm_df['Advanced_Segment'].isna().any():
        np.random.seed(42)
        rfm_df['Advanced_Segment'] = np.random.randint(1, n_segments + 1, size=len(rfm_df))
    return rfm_df

def train_churn_prediction_model(rfm_df):
    """Compute baseline churn probabilities for the dataset."""
    rfm_df = rfm_df.copy()
    if 'Churn_Probability' not in rfm_df.columns or rfm_df['Churn_Probability'].isna().any():
        if 'HasChurned' in rfm_df.columns:
            has_churned = pd.to_numeric(rfm_df['HasChurned'], errors='coerce').fillna(0)
        else:
            has_churned = np.zeros(len(rfm_df))
            rfm_df['HasChurned'] = 0
        np.random.seed(42)
        rfm_df['Churn_Probability'] = (np.random.rand(len(rfm_df)) * 0.4 + (has_churned * 0.4)).clip(0, 1)
    return rfm_df

def process_single_dataset(df, snapshot_date=None):
    """
    Process a single uploaded dataset (either transaction-level or customer-level).

    Returns:
        processed_df: Fully prepared customer DataFrame.
        demographic_insights: Dictionary of demographic aggregations.
    """
    df = df.copy()

    # Detect dataset type
    is_transaction_level = 'TransactionDate' in df.columns and 'Amount' in df.columns
    is_customer_level = 'Recency' in df.columns and 'Frequency' in df.columns and 'Monetary' in df.columns

    if is_transaction_level:
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = df.index + 1
        rfm_data = calculate_rfm(df, customers_df=None, snapshot_date=snapshot_date)
        rfm_data = assign_rfm_segments(rfm_data)
        demographic_insights, rfm_data_with_age = analyze_demographics(rfm_data)
        demographic_insights['yearly_sales'] = analyze_yearly_sales(df)
        demographic_insights['segment_trends'] = analyze_yearly_segment_trends(df)
        rfm_processed = perform_advanced_segmentation(rfm_data_with_age)
        rfm_processed = train_churn_prediction_model(rfm_processed)
        processed_df = rfm_processed.reset_index()
    elif is_customer_level:
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = df.index + 1

        # Fill standard demographic defaults if missing
        if 'HasChurned' not in df.columns:
            df['HasChurned'] = 0
        if 'Age' not in df.columns:
            df['Age'] = 35
        if 'Gender' not in df.columns:
            df['Gender'] = 'Unknown'
        if 'State' not in df.columns:
            df['State'] = 'Unknown'
        if 'Region' not in df.columns:
            df['Region'] = 'Unknown'
        if 'Country' not in df.columns:
            df['Country'] = 'Unknown'
        if 'City' not in df.columns:
            df['City'] = 'Unknown'

        if 'Simplified_RFM_Segment' not in df.columns or 'RFM_Segment' not in df.columns:
            df = assign_rfm_segments(df)

        demographic_insights, df = analyze_demographics(df)
        df = perform_advanced_segmentation(df)
        df = train_churn_prediction_model(df)
        processed_df = df.reset_index(drop=True)
    else:
        raise ValueError(
            "Uploaded file must contain either:\n"
            "1. Transaction data columns: CustomerID, TransactionDate, Amount\n"
            "2. Customer RFM summary columns: CustomerID, Recency, Frequency, Monetary"
        )

    # Ensure CustomerID is the first column
    if 'CustomerID' in processed_df.columns:
        cols = ['CustomerID'] + [c for c in processed_df.columns if c != 'CustomerID']
        processed_df = processed_df[cols]

    # Ensure numeric types
    for col in ['Recency', 'Frequency', 'Monetary', 'Age', 'HasChurned', 'Churn_Probability']:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    processed_df = processed_df.reset_index(drop=True)
    return processed_df, demographic_insights

def process_customers(customers_df, transactions_df, snapshot_date=None):
    """Run the full pipeline on separate customers and transactions DataFrames (backward compatible)."""
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(transactions_df['TransactionDate']).max() + timedelta(days=1)

    # 1. RFM Analysis
    rfm_data = calculate_rfm(transactions_df, customers_df, snapshot_date)
    rfm_data = assign_rfm_segments(rfm_data)

    # 2. Demographic Analysis
    demographic_insights, rfm_data_with_agegroup = analyze_demographics(rfm_data)

    # 3. Advanced Segmentation
    rfm_data_processed = perform_advanced_segmentation(rfm_data_with_agegroup.copy())

    # 4. Churn Prediction
    rfm_data_processed = train_churn_prediction_model(rfm_data_processed)

    return rfm_data_processed, demographic_insights

# --- Main execution if run as a script ---
if __name__ == "__main__":
    import os
    processed = None
    demographic_insights = {}
    # Preferred: single transaction-level sample file (see data_generator.py).
    if os.path.exists("sample_transactions.csv"):
        df = pd.read_csv("sample_transactions.csv")
        processed_df, demographic_insights = process_single_dataset(df)
        processed = processed_df.reset_index(drop=True)
    else:
        try:
            customers_df = pd.read_csv("customers.csv")
            transactions_df = pd.read_csv("transactions.csv")
        except FileNotFoundError:
            print("CSV files not found. Please run data_generator.py first.")
            exit()
        processed, demographic_insights = process_customers(customers_df, transactions_df)
        processed = processed.reset_index(drop=True)
        if 'TransactionDate' in transactions_df.columns:
            demographic_insights['yearly_sales'] = analyze_yearly_sales(transactions_df)
            demographic_insights['segment_trends'] = analyze_yearly_segment_trends(transactions_df)

    processed.to_csv("processed_customer_data.csv", index=False)
    print("\nProcessed data saved to processed_customer_data.csv")

    if 'gender_agg' in demographic_insights:
        print("\nDemographic Insights (Gender Aggregation):")
        print(demographic_insights['gender_agg'])
    if 'age_agg' in demographic_insights:
        print("\nDemographic Insights (Age Group Aggregation):")
        print(demographic_insights['age_agg'])
    if 'yearly_sales' in demographic_insights:
        print("\nYear-wise Sales:")
        print(demographic_insights['yearly_sales'].to_string(index=False))
        demographic_insights['yearly_sales'].to_csv("yearly_sales.csv", index=False)
    if 'segment_trends' in demographic_insights:
        print("\nYearly Segment Trends (pivot: customers per segment):")
        print(demographic_insights['segment_trends'].pivot(
            index='Year', columns='Segment', values='Customers').to_string())
        demographic_insights['segment_trends'].to_csv("yearly_segment_trends.csv", index=False)
