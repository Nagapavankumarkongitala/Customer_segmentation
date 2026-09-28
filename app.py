import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

@st.cache_resource(show_spinner=False)
def get_tensorflow():
    """Load TensorFlow only when a deep-learning action is requested."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        return tf, keras, layers
    except ImportError:
        return None

# --- Import processing pipeline from data_preprocessing.py ---
from data_preprocessing import (
    calculate_rfm,
    assign_rfm_segments,
    analyze_demographics,
    perform_advanced_segmentation,
    train_churn_prediction_model,
    process_single_dataset,
)

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="AI Customer Segmentation & Retention Dashboard",
    page_icon="📊"
)

# --- Custom Styling: Modern Dark Theme ---
st.markdown("""
<style>
:root {
    --bg-0: #070b14;
    --bg-1: #0b1220;
    --card: #111a2e;
    --card-2: #152039;
    --border: #243152;
    --text: #e8eefc;
    --muted: #94a3c8;
    --accent: #6d8dff;
    --accent-2: #22d3ee;
    --good: #10b981;
    --warn: #f59e0b;
    --bad: #ef4444;
}
/* Global Ground & Typography */
.stApp {
    background:
        radial-gradient(1100px 500px at 10% -10%, rgba(109,141,255,0.16), transparent 60%),
        radial-gradient(900px 480px at 90% 0%, rgba(34,211,238,0.12), transparent 60%),
        linear-gradient(180deg, #070b14 0%, #0a1122 100%);
    color: var(--text);
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
h1, h2, h3, h4, h5 { color: var(--text) !important; letter-spacing: -0.01em; }
p, li, span, div { color: var(--text); }
.stCaption, small { color: var(--muted) !important; }
a { color: var(--accent-2) !important; }

/* Header Banner */
.dashboard-header {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0) 40%),
        linear-gradient(135deg, #1a2456 0%, #27348b 45%, #0e7490 100%);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    color: white;
    box-shadow: 0 18px 50px -18px rgba(60,90,255,0.55);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.dashboard-header::after {
    content: "";
    position: absolute;
    inset: -60px -40px auto auto;
    width: 340px; height: 340px;
    background: radial-gradient(circle, rgba(34,211,238,0.35), transparent 70%);
    pointer-events: none;
}
.dashboard-header h1 {
    font-size: 2rem !important;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
    color: #ffffff !important;
}
.dashboard-header p { font-size: 1rem; opacity: 0.9; margin: 0; color: #dbe6ff !important; }
.header-stats { display: flex; gap: 0.6rem; margin-top: 0.9rem; flex-wrap: wrap; }
.header-pill {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 999px;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #eaf0ff !important;
}

/* View nav (st.radio horizontal) */
div[data-testid="stRadio"] > div { gap: 0.5rem; }
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    background: rgba(17,26,46,0.9);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.55rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 1.1rem;
}
div[data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 11px !important;
    padding: 0.5rem 0.9rem !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] label p { color: var(--muted) !important; font-weight: 700 !important; font-size: 0.9rem !important; }
div[data-testid="stRadio"] label:hover { background: rgba(109,141,255,0.12) !important; }
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #5b7cff, #22d3ee) !important;
    box-shadow: 0 8px 20px rgba(70,110,255,0.4);
}
div[data-testid="stRadio"] label:has(input:checked) p { color: #06101f !important; }

/* Legacy tab styles kept for st.tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; justify-content: flex-start; flex-wrap: wrap;
    background: var(--card);
    border-radius: 14px; padding: 0.5rem;
    border: 1px solid var(--border);
    margin-bottom: 1.1rem;
}
.stTabs [data-baseweb="tab"] {
    height: 40px; padding: 0 1rem; border-radius: 10px;
    font-weight: 700; font-size: 0.88rem; color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #5b7cff 0%, #22d3ee 100%) !important;
    color: #06101f !important;
}

/* Cards / layout */
.glass-card, .slicer-card, .chart-card, .kpi-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)), var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(0,0,0,0.35);
}
.slicer-card {
    padding: 1rem 1.2rem;
    border-top: 3px solid var(--accent);
    border-left: 1px solid var(--border);
    margin-bottom: 1rem;
}
.slicer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; gap: 0.6rem; flex-wrap: wrap; }
.slicer-title { font-size: 1.0rem; font-weight: 800; color: var(--text) !important; display: flex; align-items: center; gap: 0.45rem; }
.slicer-badge {
    background: rgba(109,141,255,0.14);
    color: #c9d6ff !important;
    font-size: 0.78rem; font-weight: 800;
    padding: 0.25rem 0.7rem; border-radius: 9999px;
    border: 1px solid rgba(109,141,255,0.4);
}
/* Dynamic chart grid: each plotly chart gets a card look */
div[data-testid="stPlotlyChart"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.008)), var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.55rem 0.55rem 0.2rem 0.55rem;
    box-shadow: 0 14px 34px rgba(0,0,0,0.35);
    margin-bottom: 0.2rem;
}
div[data-testid="stPlotlyChart"]:hover { border-color: rgba(109,141,255,0.55); }

/* Tiered Section Headers */
.section-tier-header {
    display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;
    font-size: 1.08rem; font-weight: 800; color: var(--text) !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem; margin-top: 1.4rem; margin-bottom: 0.9rem;
}
.tier-tag {
    font-size: 0.7rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.06em;
    padding: 0.22rem 0.6rem; border-radius: 999px;
    background: linear-gradient(135deg, #5b7cff, #22d3ee);
    color: #06101f !important;
}

/* KPI Metric Cards */
.stMetric {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01)), var(--card) !important;
    border-radius: 16px !important;
    padding: 1rem 1.1rem !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.35) !important;
    min-height: 118px;
}
.stMetric:hover { transform: translateY(-2px); border-color: rgba(109,141,255,0.55) !important; }
.stMetric [data-testid="stMetricLabel"] p, .stMetric > label {
    color: var(--muted) !important; font-weight: 700 !important;
    font-size: 0.76rem !important; text-transform: uppercase; letter-spacing: 0.06em;
}
.stMetric [data-testid="stMetricValue"] div, .stMetric > div {
    font-size: 1.55rem !important; font-weight: 800 !important; color: var(--text) !important;
}
.stMetric [data-testid="stMetricDelta"] { color: var(--muted) !important; }

/* Strategy & Action Cards */
.action-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)), var(--card);
    border-radius: 16px; padding: 1.1rem 1.2rem;
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent);
    margin-bottom: 0.9rem;
}
.action-card h4 { margin: 0 0 0.4rem 0; color: var(--text) !important; font-size: 1.02rem; font-weight: 800; }
.action-card p { margin: 0 0 0.4rem 0; color: var(--muted) !important; font-size: 0.9rem; line-height: 1.5; }
.action-card strong { color: var(--text) !important; }

/* Dataframe / tables */
div[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden;
    border: 1px solid var(--border);
    background: var(--card);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}

/* Inputs */
div[data-testid="stSelectbox"] > div, div[data-testid="stMultiSelect"] > div,
div[data-testid="stSlider"] > div, div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div, div[data-testid="stFileUploader"] > div {
    color: var(--text);
}
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(109,141,255,0.18) !important;
    border: 1px solid rgba(109,141,255,0.45) !important;
    border-radius: 999px !important;
}
.stMultiSelect [data-baseweb="tag"] span { color: #dbe6ff !important; }
div[data-baseweb="select"] > div, div[data-baseweb="popover"] {
    background-color: var(--card-2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
input, textarea { color: var(--text) !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* Expanders */
div[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}
div[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 700; }

/* Primary Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #5b7cff 0%, #22d3ee 100%) !important;
    color: #06101f !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.3rem !important;
    font-weight: 800 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 10px 26px rgba(70,110,255,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
}
.stButton > button p, .stDownloadButton > button p { color: #06101f !important; font-weight: 800 !important; }

/* Alerts / captions */
.stAlert { border-radius: 14px !important; background: rgba(17,26,46,0.9) !important; border: 1px solid var(--border) !important; }
.stAlert p { color: var(--text) !important; }
.stCaption, div[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
hr { border-color: var(--border) !important; }

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0a1122; }
::-webkit-scrollbar-thumb { background: #2b3b66; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #3d5290; }

/* Responsive: stack gracefully on narrow screens */
@media (max-width: 900px) {
    .dashboard-header { padding: 1.2rem 1.2rem; }
    .dashboard-header h1 { font-size: 1.5rem !important; }
}

/* Hide Default Streamlit Overhead */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions & Color Palettes ---

# Consistent, high-contrast categorical color palettes
SEGMENT_COLORS = {
    'Champions': '#10b981',        # Emerald Green
    'Loyal Customers': '#2563eb',  # Royal Blue
    'Potential Loyalists': '#8b5cf6', # Vibrant Purple
    'Need Attention': '#f59e0b',   # Amber
    'Hibernating': '#ef4444',      # Crimson Red
    'Other': '#64748b'             # Slate Gray
}

RISK_TIER_COLORS = {
    'Low Risk (<40%)': '#10b981',
    'Medium Risk (40-70%)': '#f59e0b',
    'High Risk (>70%)': '#ef4444',
    'Low Risk': '#10b981',
    'Medium Risk': '#f59e0b',
    'High Risk': '#ef4444',
    'Low': '#10b981',
    'Medium': '#f59e0b',
    'High': '#ef4444'
}

GENDER_COLORS = {
    'Female': '#ec4899',   # Vibrant Pink
    'Male': '#2563eb',     # Royal Blue
    'Other': '#8b5cf6',    # Violet
    'Unknown': '#64748b'   # Slate
}

AGE_GROUP_COLORS = {
    '<18': '#a78bfa',
    '18-24': '#38bdf8',
    '25-34': '#34d399',
    '35-44': '#fbbf24',
    '45-54': '#f87171',
    '55-64': '#fb923c',
    '65+': '#e879f9',
    'Unknown': '#94a3b8'
}

# Geographic color palettes for State, Region, Country, City
STATE_COLORS = {
    'CA': '#ef4444', 'NY': '#2563eb', 'TX': '#22c55e', 'FL': '#f97316',
    'IL': '#8b5cf6', 'PA': '#ec4899', 'OH': '#06b6d4', 'GA': '#84cc16',
    'NC': '#f59e0b', 'MI': '#6366f1', 'Unknown': '#64748b'
}

REGION_COLORS = {
    'Northeast': '#2563eb', 'Midwest': '#10b981', 'South': '#ef4444', 'West': '#f59e0b',
    'Northwest': '#8b5cf6', 'Southwest': '#f97316', 'Southeast': '#ec4899', 'Central': '#06b6d4',
    'Unknown': '#64748b'
}

COUNTRY_COLORS = {
    'USA': '#2563eb', 'Canada': '#ef4444', 'UK': '#f59e0b', 'Germany': '#10b981',
    'France': '#8b5cf6', 'Australia': '#ec4899', 'Japan': '#f97316', 'Brazil': '#22c55e',
    'India': '#06b6d4', 'China': '#84cc16', 'Unknown': '#64748b'
}

CITY_COLORS = {
    'New York': '#2563eb', 'Los Angeles': '#ef4444', 'Chicago': '#10b981', 'Houston': '#f59e0b',
    'Phoenix': '#8b5cf6', 'Philadelphia': '#ec4899', 'San Antonio': '#06b6d4', 'San Diego': '#f97316',
    'Dallas': '#84cc16', 'San Jose': '#6366f1', 'Unknown': '#64748b'
}

def style_plotly_fig(fig, height=360, show_legend=False, legend_title=None, legend_orientation="v"):
    """
    Standardize Plotly layout for the dark theme: glass cards, light text,
    subtle grids, readable legend.
    """
    layout_update = dict(
        height=height,
        margin=dict(l=30, r=30, t=52, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#e8eefc"),
        title=dict(
            font=dict(size=14, color="#e8eefc", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
            x=0.02,
            xanchor="left"
        ),
        showlegend=show_legend
    )

    if show_legend:
        if legend_orientation == "h":
            layout_update['legend'] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(11,18,32,0.9)",
                bordercolor="#243152",
                borderwidth=1,
                font=dict(size=11, color="#e8eefc")
            )
        else:
            layout_update['legend'] = dict(
                bgcolor="rgba(11,18,32,0.9)",
                bordercolor="#243152",
                borderwidth=1,
                font=dict(size=11, color="#e8eefc"),
                title=dict(text=legend_title if legend_title else "", font=dict(size=11, color="#9fb3ff", family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"))
            )

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        gridcolor="rgba(148,163,200,0.16)",
        linecolor="#243152",
        zerolinecolor="rgba(148,163,200,0.18)",
        tickfont=dict(color="#aebad9", size=11),
        title_font=dict(color="#cdd8f5", size=12)
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,200,0.16)",
        linecolor="#243152",
        zerolinecolor="rgba(148,163,200,0.18)",
        tickfont=dict(color="#aebad9", size=11),
        title_font=dict(color="#cdd8f5", size=12)
    )
    return fig


def build_geo_map(df, value_col='Monetary', agg='sum', title="Geographic Performance Map"):
    """Build a dynamic choropleth map from Country/State columns.

    Prefers Country-level world map, falls back to USA-states map when
    State codes look like US states. Returns (fig, level) or (None, None).
    """
    if df is None or df.empty or value_col not in df.columns:
        return None, None
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce').fillna(0)

    if 'Country' in work.columns and work['Country'].notna().any():
        g = work.groupby(work['Country'].astype(str))[value_col].agg(agg).reset_index()
        g = g[g['Country'].str.strip().ne('') & g['Country'].ne('Unknown')]
        if len(g) >= 1:
            fig = px.choropleth(
                g, locations='Country', locationmode='country names',
                color=value_col, hover_name='Country',
                color_continuous_scale='Blues',
                title=title, template='plotly_dark',
            )
            fig.update_geos(showframe=False, showcoastlines=True, projection_type='natural earth')
            return style_plotly_fig(fig, height=420, show_legend=True), 'Country'

    if 'State' in work.columns and work['State'].notna().any():
        g = work.groupby(work['State'].astype(str))[value_col].agg(agg).reset_index()
        g = g[g['State'].str.strip().ne('') & g['State'].ne('Unknown')]
        looks_usa = (g['State'].str.len() == 2).mean() > 0.5 if len(g) else False
        if len(g) >= 1 and looks_usa:
            fig = px.choropleth(
                g, locations='State', locationmode='USA-states',
                color=value_col, hover_name='State', scope='usa',
                color_continuous_scale='Blues',
                title=title, template='plotly_dark',
            )
            return style_plotly_fig(fig, height=420, show_legend=True), 'State'
        if len(g) >= 1:
            fig = px.bar(
                g.sort_values(value_col, ascending=False).head(20),
                x='State', y=value_col, color=value_col,
                color_continuous_scale='Blues',
                title=f"{title} (Top States)", template='plotly_dark',
            )
            return style_plotly_fig(fig, height=420, show_legend=False), 'State-bar'

    return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("processed_customer_data.csv")
        for col in ['Recency', 'Frequency', 'Monetary', 'Age', 'Churn_Probability', 'HasChurned']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def safe_range(series, default_min=0, default_max=100):
    s = series.dropna()
    if s.empty:
        return float(default_min), float(default_max)
    mn = float(s.min())
    mx = float(s.max())
    if mn == mx:
        return float(mn), float(mn + 1.0)
    return float(mn), float(mx)

# --- Autoencoder Implementation ---

def build_autoencoder(input_dim, encoding_dim=8, latent_dim=3):
    """Build an autoencoder model for dimensionality reduction."""
    tensorflow_modules = get_tensorflow()
    if tensorflow_modules is None:
        return None, None

    _, keras, layers = tensorflow_modules
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(inputs)
    encoded = layers.Dropout(0.1)(encoded)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(encoded)

    decoded = layers.Dense(encoding_dim, activation='relu')(latent)
    decoded = layers.Dropout(0.1)(decoded)
    outputs = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = keras.Model(inputs, outputs, name='Autoencoder')
    encoder = keras.Model(inputs, latent, name='Encoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def train_autoencoder(features, encoding_dim=8, latent_dim=3, epochs=100, batch_size=32):
    """Train an autoencoder on the given features and return latent representations."""
    tensorflow_modules = get_tensorflow()
    if tensorflow_modules is None:
        st.warning("TensorFlow is not installed. Autoencoder cannot be used. Install with: pip install tensorflow")
        return None, None

    _, keras, _ = tensorflow_modules
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_norm = (features_scaled - features_scaled.min(axis=0)) / (features_scaled.max(axis=0) - features_scaled.min(axis=0) + 1e-8)

    input_dim = features_norm.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim, latent_dim)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    with st.spinner("Training Autoencoder..."):
        autoencoder.fit(
            features_norm, features_norm,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0
        )

    latent_features = encoder.predict(features_norm, verbose=0)
    latent_df = pd.DataFrame(latent_features, columns=[f'Latent_{i}' for i in range(latent_dim)])
    latent_df.index = features.index
    return latent_df, autoencoder

# --- Ensemble Clustering Implementation ---

def ensemble_clustering(features, n_clusters=5, algorithms=None, dbscan_eps=0.5, dbscan_min=5):
    """Perform ensemble clustering using multiple algorithms and majority voting."""
    if algorithms is None:
        algorithms = ['kmeans', 'dbscan', 'agglomerative']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    cluster_results = {}

    if 'kmeans' in algorithms:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_results['KMeans'] = kmeans.fit_predict(features_scaled)

    if 'dbscan' in algorithms:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min)
        cluster_results['DBSCAN'] = dbscan.fit_predict(features_scaled)
        unique_labels = np.unique(cluster_results['DBSCAN'])
        unique_labels = unique_labels[unique_labels != -1]
        label_map = {old: new for new, old in enumerate(unique_labels)}
        cluster_results['DBSCAN'] = np.array([label_map.get(l, -1) for l in cluster_results['DBSCAN']])

    if 'agglomerative' in algorithms:
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_results['Agglomerative'] = agg.fit_predict(features_scaled)

    n_samples = features_scaled.shape[0]
    ensemble_labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        votes = {}
        for labels in cluster_results.values():
            vote = labels[i]
            votes[vote] = votes.get(vote, 0) + 1
        if votes:
            ensemble_labels[i] = max(votes, key=lambda k: (votes[k], -k))
        else:
            ensemble_labels[i] = 0

    ensemble_df = pd.DataFrame({
        'Ensemble_Segment': ensemble_labels,
        'KMeans_Segment': cluster_results.get('KMeans', [-1]*n_samples),
        'DBSCAN_Segment': cluster_results.get('DBSCAN', [-1]*n_samples),
        'Agglomerative_Segment': cluster_results.get('Agglomerative', [-1]*n_samples)
    }, index=features.index)

    return ensemble_df, cluster_results

def compute_silhouette_score(features, labels):
    """Compute silhouette score for clustering quality."""
    from sklearn.metrics import silhouette_score
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels:
        mask = labels != -1
        if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
            return None
        return silhouette_score(features[mask], labels[mask])
    return silhouette_score(features, labels)

# --- Deep Learning Churn Prediction ---

def build_churn_model(input_dim):
    """Build a neural network classifier for churn prediction."""
    tensorflow_modules = get_tensorflow()
    if tensorflow_modules is None:
        return None

    _, keras, layers = tensorflow_modules
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model

def train_churn_model(df, feature_cols, target_col='HasChurned', epochs=50, batch_size=32, validation_split=0.2):
    """Train a deep learning model for churn prediction."""
    tensorflow_modules = get_tensorflow()
    if tensorflow_modules is None:
        st.warning("TensorFlow is not installed. Cannot train deep learning churn model.")
        return None, None, None, None

    _, keras, _ = tensorflow_modules
    features = df[feature_cols].fillna(0).values
    target = df[target_col].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    stratify_opt = target if np.unique(target).size > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42, stratify=stratify_opt
    )

    model = build_churn_model(X_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    with st.spinner("Training Deep Learning Churn Model..."):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob) if len(np.unique(y_test)) > 1 else 0.5

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    return model, metrics, scaler, history

def predict_churn_deep_learning(df, model, scaler, feature_cols):
    """Predict churn probabilities using the trained deep learning model."""
    features = df[feature_cols].fillna(0).values
    features_scaled = scaler.transform(features)
    churn_prob = model.predict(features_scaled, verbose=0).flatten()
    return churn_prob

# --- Autoencoder Visualization ---

def plot_latent_space(latent_df, color_by, title="Latent Space Visualization"):
    """Visualize the 2D/3D latent space from autoencoder with clear styling and visible legend."""
    color_map = None
    if color_by == 'Simplified_RFM_Segment':
        color_map = SEGMENT_COLORS
    elif color_by == 'Risk_Level' or color_by == 'Temp_Risk_Tier':
        color_map = RISK_TIER_COLORS
    elif color_by == 'Gender':
        color_map = GENDER_COLORS
    elif color_by == 'AgeGroup':
        color_map = AGE_GROUP_COLORS
    elif color_by == 'State':
        color_map = STATE_COLORS
    elif color_by == 'Region':
        color_map = REGION_COLORS
    elif color_by == 'Country':
        color_map = COUNTRY_COLORS
    elif color_by == 'City':
        color_map = CITY_COLORS

    is_numeric = pd.api.types.is_numeric_dtype(latent_df[color_by]) if color_by in latent_df.columns else False

    if 'Latent_2' in latent_df.columns:
        fig = px.scatter_3d(
            latent_df, x='Latent_0', y='Latent_1', z='Latent_2',
            color=color_by, title=title, opacity=0.85,
            template="plotly_dark",
            color_discrete_map=color_map if not is_numeric else None,
            color_continuous_scale="Viridis" if is_numeric else None,
            color_discrete_sequence=px.colors.qualitative.Bold if not is_numeric and not color_map else None
        )
    elif 'Latent_1' in latent_df.columns:
        fig = px.scatter(
            latent_df, x='Latent_0', y='Latent_1',
            color=color_by, title=title, opacity=0.85,
            template="plotly_dark",
            color_discrete_map=color_map if not is_numeric else None,
            color_continuous_scale="Viridis" if is_numeric else None,
            color_discrete_sequence=px.colors.qualitative.Bold if not is_numeric and not color_map else None
        )
    else:
        fig = px.histogram(
            latent_df, x='Latent_0', color=color_by, title=title,
            template="plotly_dark",
            color_discrete_map=color_map if not is_numeric else None,
            color_discrete_sequence=px.colors.qualitative.Bold if not is_numeric and not color_map else None
        )
    style_plotly_fig(fig, height=360, show_legend=False, legend_title=str(color_by))
    return fig

COLUMN_DESCRIPTIONS = {
    "transactions": {
        "CustomerID": "Unique identifier for each customer (integer or string) — Required",
        "TransactionDate": "Date of transaction (YYYY-MM-DD or datetime format) — Required",
        "Amount": "Transaction spend amount (numeric) — Required",
        "Age": "Customer's age in years (optional, integer 18–100)",
        "Gender": "Customer gender (optional, e.g. Male / Female / Other)",
        "State": "State/Province (optional, e.g. CA, NY, TX)",
        "Region": "Region (optional, e.g. Northeast, Midwest, South, West)",
        "Country": "Country (optional, e.g. USA, Canada, UK)",
        "City": "City (optional, e.g. New York, Los Angeles, Chicago)",
        "HasChurned": "Customer churn flag (optional, 1 = Churned, 0 = Retained)"
    },
    "rfm_summary": {
        "CustomerID": "Unique identifier for each customer (integer or string) — Required",
        "Recency": "Days since the last purchase (integer) — Required",
        "Frequency": "Total number of purchases (integer) — Required",
        "Monetary": "Total spend across all purchases (numeric) — Required",
        "Age": "Customer age (optional, integer)",
        "Gender": "Customer gender (optional, string)",
        "State": "State/Province (optional, string)",
        "Region": "Region (optional, string)",
        "Country": "Country (optional, string)",
        "City": "City (optional, string)",
        "HasChurned": "Customer churn flag (optional, 1 = churned, 0 = retained)",
        "Simplified_RFM_Segment": "Segment label (optional, auto-assigned if missing)",
        "Churn_Probability": "Predicted churn probability (optional, auto-assigned if missing)"
    }
}

# --- Load Active Working Data ---
if 'uploaded_data' in st.session_state:
    customer_df = st.session_state['uploaded_data'].copy()
else:
    customer_df = load_data()

# Synchronize persisted session features if data exists
if customer_df is not None:
    if 'latent_features' in st.session_state:
        customer_df = customer_df.join(st.session_state['latent_features'], how='left')
    if 'ensemble_segments' in st.session_state:
        customer_df = customer_df.drop(columns=['Ensemble_Segment'], errors='ignore').join(st.session_state['ensemble_segments'], how='left')

    # Ensure numeric columns
    for col in ['Recency', 'Frequency', 'Monetary', 'Age', 'HasChurned', 'Churn_Probability']:
        if col in customer_df.columns:
            customer_df[col] = pd.to_numeric(customer_df[col], errors='coerce')

# Header Title Banner - Always shown
st.markdown("""
<div class="dashboard-header">
    <h1>📊 AI Customer Segmentation & Retention Platform</h1>
    <p>Interactive lifecycle analytics, dynamic cohort slicing, deep autoencoder representations, and predictive retention</p>
</div>
""", unsafe_allow_html=True)

# Show info message if no data
if customer_df is None:
    st.info("📂 No data file found. Please upload a dataset using the Upload Data tab below.")

# Render one dashboard view at a time to avoid building every chart on page load.
dashboard_views = [
    "📤 Upload Data",
    "🏠 Overview",
    "📈 RFM Analysis",
    "👥 Demographic Profiling",
    "🤖 Advanced Segmentation",
    "⚠️ Churn Prediction",
    "🎯 Retention Strategies"
]

# Force Upload Data view if no data
default_index = 0 if customer_df is None else 0
active_view = st.radio(
    "Dashboard view",
    options=dashboard_views,
    horizontal=True,
    label_visibility="collapsed",
    key="active_dashboard_view",
    index=default_index
)

# =========================================================================
# TAB 0: UPLOAD DATA - Always available
# =========================================================================
if active_view == "📤 Upload Data":
    st.header("📤 Single-File Data Ingestion & Preprocessing")
    st.markdown("Upload raw transaction logs or customer RFM summaries to automatically compute metrics and train models.")

    # --- DYNAMIC SLICERS FOR TAB 0 ---
    st.markdown("""
    <div class="slicer-card">
        <div class="slicer-header">
            <span class="slicer-title">🎛️ Ingestion & Preview Slicers</span>
            <span class="slicer-badge">Live Filter Controls</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t0_s1, t0_s2, t0_s3 = st.columns(3)
    with t0_s1:
        t0_sample_size = st.slider("Max Preview Rows:", 5, 50, 10, 5, key="t0_sample_size")
    with t0_s2:
        all_segments_t0 = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if customer_df is not None and 'Simplified_RFM_Segment' in customer_df.columns else []
        t0_seg_filter = st.multiselect("Filter Active Data by Segment:", options=all_segments_t0, default=all_segments_t0, key="t0_seg_filter")
    with t0_s3:
        if customer_df is not None:
            min_m0, max_m0 = safe_range(customer_df['Monetary'])
        else:
            min_m0, max_m0 = 0, 100
        t0_spend_range = st.slider("Filter by Customer Spend ($):", min_value=float(min_m0), max_value=float(max_m0), value=(float(min_m0), float(max_m0)), key="t0_spend_range")

    # Apply Slicer
    if customer_df is not None:
        df_tab0 = customer_df.copy()
        if t0_seg_filter:
            df_tab0 = df_tab0[df_tab0['Simplified_RFM_Segment'].isin(t0_seg_filter)]
        if 'Monetary' in df_tab0.columns:
            df_tab0 = df_tab0[(df_tab0['Monetary'] >= t0_spend_range[0]) & (df_tab0['Monetary'] <= t0_spend_range[1])]

        st.caption(f"🔍 **Active Slicer Scope:** Showing **{len(df_tab0):,}** of **{len(customer_df):,}** customers ({len(df_tab0)/max(len(customer_df),1):.1%})")
    else:
        df_tab0 = None
        st.caption("🔍 **Active Slicer Scope:** No data loaded yet")

    # --- TIER 1: KPI CARDS ---
    st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> Ingestion & Health KPIs</div>', unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        source_label = "Custom Upload" if 'uploaded_data' in st.session_state else "Default Synthetic"
        st.metric("Active Data Source", source_label)
    with kpi_col2:
        if df_tab0 is not None:
            st.metric("Filtered Customer Base", f"{len(df_tab0):,}")
        else:
            st.metric("Filtered Customer Base", "—")
    with kpi_col3:
        if df_tab0 is not None:
            total_rev = df_tab0['Monetary'].sum() if 'Monetary' in df_tab0.columns else 0
            st.metric("Total Revenue in Scope", f"${total_rev:,.2f}")
        else:
            st.metric("Total Revenue in Scope", "—")
    with kpi_col4:
        if df_tab0 is not None and not df_tab0.empty:
            churn_baseline = df_tab0['Churn_Probability'].mean() if 'Churn_Probability' in df_tab0.columns else 0
            st.metric("Baseline Churn Rate", f"{churn_baseline:.1%}")
        else:
            st.metric("Baseline Churn Rate", "—")

    # Active Dataset Notification / Reset
    if 'uploaded_data' in st.session_state:
        st.info(f"📌 **Active Upload:** Dashboard is running on custom uploaded data with **{len(st.session_state['uploaded_data']):,}** customer records.")
        if st.button("🔄 Reset to Default Sample Dataset", key="reset_default_data"):
            st.session_state.pop('uploaded_data', None)
            st.session_state.pop('latent_features', None)
            st.session_state.pop('ensemble_segments', None)
            st.session_state.pop('ensemble_trained', None)
            st.session_state.pop('dl_model', None)
            st.session_state.pop('dl_scaler', None)
            st.rerun()

    # File Uploader - Always visible
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Select a CSV or Excel dataset (.csv, .xlsx)",
        type=["csv", "xlsx"],
        key="single_dataset_uploader"
    )

    has_new_upload = False
    raw_uploaded_df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                raw_uploaded_df = pd.read_excel(uploaded_file)
            else:
                raw_uploaded_df = pd.read_csv(uploaded_file)
            has_new_upload = True

            is_tx = 'TransactionDate' in raw_uploaded_df.columns and 'Amount' in raw_uploaded_df.columns
            is_rfm = 'Recency' in raw_uploaded_df.columns and 'Frequency' in raw_uploaded_df.columns and 'Monetary' in raw_uploaded_df.columns

            if is_tx:
                st.success(f"✅ **Format Recognized: Transaction-Level Logs** (`TransactionDate`, `Amount`) — {len(raw_uploaded_df):,} transaction rows detected.")
            elif is_rfm:
                st.success(f"✅ **Format Recognized: Customer RFM Summary** (`Recency`, `Frequency`, `Monetary`) — {len(raw_uploaded_df):,} customer records detected.")
            else:
                st.warning("⚠️ File columns not automatically recognized. Please ensure required columns are present.")

            if is_tx or is_rfm:
                if st.button("▶️ Process & Apply Dataset to Dashboard", key="btn_process_single_file"):
                    with st.spinner("Executing RFM aggregation, demographic profiling, and churn modeling pipeline..."):
                        processed_df, demo_insights = process_single_dataset(raw_uploaded_df)
                        st.session_state['uploaded_data'] = processed_df
                        st.session_state.pop('latent_features', None)
                        st.session_state.pop('ensemble_segments', None)
                        st.session_state.pop('ensemble_trained', None)
                        st.session_state.pop('dl_model', None)
                        st.session_state.pop('dl_scaler', None)
                    st.success(f"🎉 Pipeline completed successfully! **{len(processed_df):,}** customer profiles are now active.")
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

        # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
        st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Dataset Distribution & Health Visuals</div>', unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            preview_target_df = raw_uploaded_df if has_new_upload and raw_uploaded_df is not None else df_tab0
            if preview_target_df is not None:
                completeness = preview_target_df.notna().mean() * 100
                fig_comp = px.bar(
                    x=completeness.index, y=completeness.values,
                    title="Data Completeness & Attribute Coverage (%)",
                    labels={'x': 'Column Attribute', 'y': 'Fill Rate (%)'},
                    color=completeness.values,
                    color_continuous_scale="Blues",
                    template="plotly_dark"
                )
                style_plotly_fig(fig_comp, height=350, show_legend=False)
                st.plotly_chart(fig_comp, use_container_width=True, theme=None)
            else:
                st.info("Upload a file to see data completeness analysis.")

        with chart_col2:
            if df_tab0 is not None and 'Monetary' in df_tab0.columns and not df_tab0.empty:
                fig_spend_dist = px.histogram(
                    df_tab0, x='Monetary', nbins=30,
                    title="Customer Spend (Monetary) Distribution in Sliced Scope",
                    labels={'Monetary': 'Total Spend ($)'},
                    color_discrete_sequence=['#2563eb'],
                    template="plotly_dark"
                )
                style_plotly_fig(fig_spend_dist, height=350, show_legend=False)
                st.plotly_chart(fig_spend_dist, use_container_width=True, theme=None)
            else:
                st.info("No data in current sliced scope to plot spend distribution.")

        # --- TIER 3: TABULAR REPRESENTATION ---
        st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Dataset Previews & Reference Specifications</div>', unsafe_allow_html=True)
        if has_new_upload and raw_uploaded_df is not None:
            st.markdown("##### 🔍 Uploaded File Raw Preview")
            st.dataframe(raw_uploaded_df.head(t0_sample_size), use_container_width=True)

        st.markdown("##### 🔍 Sliced Customer Dataset Preview")
        if df_tab0 is not None:
            st.dataframe(df_tab0.head(t0_sample_size), use_container_width=True)
        else:
            st.info("No data loaded yet. Upload a file to see preview.")

        with st.expander("📋 Supported File Schema Reference & Templates", expanded=False):
            col_ref1, col_ref2 = st.columns(2)
            with col_ref1:
                st.markdown("##### 1. Single Transaction Format")
                for col_name, col_desc in COLUMN_DESCRIPTIONS["transactions"].items():
                    st.markdown(f"- **`{col_name}`**: {col_desc}")
            with col_ref2:
                st.markdown("##### 2. Customer RFM Summary Format")
                for col_name, col_desc in COLUMN_DESCRIPTIONS["rfm_summary"].items():
                    st.markdown(f"- **`{col_name}`**: {col_desc}")

            st.divider()
            st.markdown("##### 📥 Download Sample Single-File CSV Templates")
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                sample_tx = pd.DataFrame([
                    {"CustomerID": 1, "TransactionDate": "2025-01-15", "Amount": 120.50, "Age": 34, "Gender": "Male", "State": "CA", "Region": "West", "Country": "USA", "City": "Los Angeles", "HasChurned": 0},
                    {"CustomerID": 1, "TransactionDate": "2025-02-10", "Amount": 85.00, "Age": 34, "Gender": "Male", "State": "CA", "Region": "West", "Country": "USA", "City": "Los Angeles", "HasChurned": 0},
                    {"CustomerID": 2, "TransactionDate": "2024-11-05", "Amount": 340.20, "Age": 48, "Gender": "Female", "State": "NY", "Region": "Northeast", "Country": "USA", "City": "New York", "HasChurned": 1},
                    {"CustomerID": 3, "TransactionDate": "2025-03-01", "Amount": 45.00, "Age": 22, "Gender": "Other", "State": "TX", "Region": "South", "Country": "USA", "City": "Houston", "HasChurned": 0},
                ])
                st.download_button(
                    "⬇️ Download Sample Transactions (.csv)",
                    data=sample_tx.to_csv(index=False),
                    file_name="sample_single_transactions.csv",
                    mime="text/csv",
                    key="dl_sample_tx_btn"
                )
            with dl_c2:
                sample_rfm = pd.DataFrame([
                    {"CustomerID": 1, "Recency": 15, "Frequency": 12, "Monetary": 1450.80, "Age": 34, "Gender": "Male", "State": "CA", "Region": "West", "Country": "USA", "City": "Los Angeles", "HasChurned": 0},
                    {"CustomerID": 2, "Recency": 140, "Frequency": 2, "Monetary": 180.00, "Age": 48, "Gender": "Female", "State": "NY", "Region": "Northeast", "Country": "USA", "City": "New York", "HasChurned": 1},
                    {"CustomerID": 3, "Recency": 5, "Frequency": 20, "Monetary": 3200.50, "Age": 29, "Gender": "Female", "State": "TX", "Region": "South", "Country": "USA", "City": "Houston", "HasChurned": 0},
                    {"CustomerID": 4, "Recency": 60, "Frequency": 5, "Monetary": 540.00, "Age": 52, "Gender": "Male", "State": "FL", "Region": "Southeast", "Country": "USA", "City": "Miami", "HasChurned": 0},
                ])
                st.download_button(
                    "⬇️ Download Sample Customer RFM (.csv)",
                    data=sample_rfm.to_csv(index=False),
                    file_name="sample_single_customer_rfm.csv",
                    mime="text/csv",
                    key="dl_sample_rfm_btn"
                )

# =========================================================================
# OTHER TABS - Only shown when data is loaded
# =========================================================================
if customer_df is not None:
    # =========================================================================
    # TAB 1: OVERVIEW
    # =========================================================================
    if active_view == "🏠 Overview":
        st.header("🏠 Executive Overview & Lifecycle Summary")
        st.markdown("High-level visibility into customer base size, aggregate revenue, retention health, and behavioral metrics.")

        # --- DYNAMIC SLICERS FOR TAB 1 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ Overview Cohort Dynamic Slicers</span>
                <span class="slicer-badge">Real-Time Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Basic Filters
        ov_s1, ov_s2, ov_s3, ov_s4 = st.columns(4)
        with ov_s1:
            all_rfm_segs = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if 'Simplified_RFM_Segment' in customer_df.columns else []
            ov_sel_segs = st.multiselect("RFM Segment:", options=all_rfm_segs, default=all_rfm_segs, key="ov_sel_segs")
        with ov_s2:
            ov_risk_tiers = st.multiselect("Churn Risk Tier:", options=['Low Risk (<40%)', 'Medium Risk (40-70%)', 'High Risk (>70%)'], default=['Low Risk (<40%)', 'Medium Risk (40-70%)', 'High Risk (>70%)'], key="ov_risk_tiers")
        with ov_s3:
            min_mov, max_mov = safe_range(customer_df['Monetary'])
            ov_spend_range = st.slider("Customer Spend ($):", min_value=float(min_mov), max_value=float(max_mov), value=(float(min_mov), float(max_mov)), key="ov_spend_range")
        with ov_s4:
            all_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            ov_sel_states = st.multiselect("State:", options=all_states, default=all_states, key="ov_sel_states")

        # Row 2: Geographic Filters
        ov_s5, ov_s6, ov_s7 = st.columns(3)
        with ov_s5:
            all_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            ov_sel_regions = st.multiselect("Region:", options=all_regions, default=all_regions, key="ov_sel_regions")
        with ov_s6:
            all_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            ov_sel_countries = st.multiselect("Country:", options=all_countries, default=all_countries, key="ov_sel_countries")
        with ov_s7:
            all_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            ov_sel_cities = st.multiselect("City:", options=all_cities, default=all_cities, key="ov_sel_cities")

        # Apply Overview Slicer
        df_ov = customer_df.copy()
        if ov_sel_segs:
            df_ov = df_ov[df_ov['Simplified_RFM_Segment'].isin(ov_sel_segs)]
        if 'State' in df_ov.columns and ov_sel_states:
            df_ov = df_ov[df_ov['State'].astype(str).isin(ov_sel_states)]
        if 'Region' in df_ov.columns and ov_sel_regions:
            df_ov = df_ov[df_ov['Region'].astype(str).isin(ov_sel_regions)]
        if 'Country' in df_ov.columns and ov_sel_countries:
            df_ov = df_ov[df_ov['Country'].astype(str).isin(ov_sel_countries)]
        if 'City' in df_ov.columns and ov_sel_cities:
            df_ov = df_ov[df_ov['City'].astype(str).isin(ov_sel_cities)]
        if 'Monetary' in df_ov.columns:
            df_ov = df_ov[(df_ov['Monetary'] >= ov_spend_range[0]) & (df_ov['Monetary'] <= ov_spend_range[1])]

        # Assign temp tier for filtering
        df_ov['Temp_Risk_Tier'] = pd.cut(
            df_ov['Churn_Probability'],
            bins=[-0.01, 0.4, 0.7, 1.0],
            labels=['Low Risk (<40%)', 'Medium Risk (40-70%)', 'High Risk (>70%)']
        )
        if ov_risk_tiers:
            df_ov = df_ov[df_ov['Temp_Risk_Tier'].isin(ov_risk_tiers)]

        st.caption(f"🔍 **Active Slicer Scope:** Displaying **{len(df_ov):,}** of **{len(customer_df):,}** customers ({len(df_ov)/max(len(customer_df),1):.1%})")

        if df_ov.empty:
            st.warning("⚠️ No customer records match the active filter criteria. Please broaden your slicers above.")
        else:
            # --- TIER 1: KPI CARDS ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> Executive Performance KPIs</div>', unsafe_allow_html=True)
            ov_kpi1, ov_kpi2, ov_kpi3, ov_kpi4, ov_kpi5 = st.columns(5)
            with ov_kpi1:
                st.metric("Total Customers", f"{len(df_ov):,}")
            with ov_kpi2:
                st.metric("Total Revenue", f"${df_ov['Monetary'].sum():,.2f}")
            with ov_kpi3:
                st.metric("Avg Customer Spend", f"${df_ov['Monetary'].mean():,.2f}")
            with ov_kpi4:
                st.metric("Avg Recency", f"{df_ov['Recency'].mean():.1f} days")
            with ov_kpi5:
                st.metric("Avg Churn Probability", f"{df_ov['Churn_Probability'].mean():.1%}")

            # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Macro Portfolio Visualizations</div>', unsafe_allow_html=True)
            ov_row1_c1, ov_row1_c2 = st.columns(2)
            with ov_row1_c1:
                rfm_counts = df_ov['Simplified_RFM_Segment'].value_counts().reset_index()
                rfm_counts.columns = ['Segment', 'Customer Count']
                fig_rfm_ov = px.bar(
                    rfm_counts, x='Segment', y='Customer Count',
                    color='Segment',
                    title="Customer Distribution by RFM Segment",
                    color_discrete_map=SEGMENT_COLORS,
                    template="plotly_dark"
                )
                style_plotly_fig(fig_rfm_ov, height=360, show_legend=False, legend_title="RFM Segment")
                st.plotly_chart(fig_rfm_ov, use_container_width=True, theme=None)

            with ov_row1_c2:
                risk_tier_counts = df_ov['Temp_Risk_Tier'].value_counts().reset_index()
                risk_tier_counts.columns = ['Risk Tier', 'Count']
                fig_risk_donut = px.pie(
                    risk_tier_counts, names='Risk Tier', values='Count',
                    title="Customer Portfolio Risk Tier Allocation",
                    hole=0.45,
                    color='Risk Tier',
                    color_discrete_map=RISK_TIER_COLORS,
                    template="plotly_dark"
                )
                fig_risk_donut.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(color="#e8eefc", size=12)
                )
                style_plotly_fig(fig_risk_donut, height=360, show_legend=False)
                st.plotly_chart(fig_risk_donut, use_container_width=True, theme=None)

            ov_row2_c1, ov_row2_c2 = st.columns(2)
            with ov_row2_c1:
                fig_rfm_scatter = px.scatter(
                    df_ov, x='Recency', y='Monetary',
                    color='Simplified_RFM_Segment',
                    size='Frequency' if (df_ov['Frequency'] > 0).any() else None,
                    hover_data=['CustomerID'],
                    title="Recency vs. Monetary Value (Bubble Size = Frequency)",
                    template="plotly_dark",
                    color_discrete_map=SEGMENT_COLORS
                )
                style_plotly_fig(fig_rfm_scatter, height=380, show_legend=False, legend_title="RFM Segment")
                st.plotly_chart(fig_rfm_scatter, use_container_width=True, theme=None)

            with ov_row2_c2:
                seg_rev = df_ov.groupby('Simplified_RFM_Segment')['Monetary'].sum().reset_index()
                seg_rev.columns = ['Segment', 'Total Revenue']
                fig_seg_rev = px.bar(
                    seg_rev, x='Segment', y='Total Revenue',
                    color='Segment',
                    title="Total Revenue Contribution by RFM Segment ($)",
                    template="plotly_dark",
                    color_discrete_map=SEGMENT_COLORS
                )
                style_plotly_fig(fig_seg_rev, height=380, show_legend=False, legend_title="RFM Segment")
                st.plotly_chart(fig_seg_rev, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Geographic Revenue Map (auto: Country → US States)")
            map_fig, map_level = build_geo_map(df_ov, value_col='Monetary', agg='sum', title="Total Revenue by Geography")
            if map_fig is not None:
                st.plotly_chart(map_fig, use_container_width=True, theme=None)
                st.caption(f"Map level: {map_level} • sums sliced Monetary in scope.")
            else:
                st.info("Add Country or 2-letter US State codes to enable the map.")

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Segment Summary & Customer Data Table</div>', unsafe_allow_html=True)
            st.markdown("##### 📊 RFM Segment Executive Aggregation Table")
            seg_summary = df_ov.groupby('Simplified_RFM_Segment').agg(
                Customer_Count=('CustomerID', 'count'),
                Avg_Recency=('Recency', 'mean'),
                Avg_Frequency=('Frequency', 'mean'),
                Avg_Monetary=('Monetary', 'mean'),
                Total_Monetary=('Monetary', 'sum'),
                Avg_Churn_Risk=('Churn_Probability', 'mean')
            ).round(2).reset_index()
            total_rev_ov = df_ov['Monetary'].sum()
            seg_summary['Revenue_Share_%'] = ((seg_summary['Total_Monetary'] / max(total_rev_ov, 1)) * 100).round(1)
            st.dataframe(seg_summary, use_container_width=True)

            st.markdown("##### 👥 Sliced Customer Level Detailed Records")
            overview_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Simplified_RFM_Segment', 'Churn_Probability']
            if 'Age' in df_ov.columns: overview_cols.append('Age')
            if 'Gender' in df_ov.columns: overview_cols.append('Gender')
            if 'State' in df_ov.columns: overview_cols.append('State')
            if 'Region' in df_ov.columns: overview_cols.append('Region')
            if 'Country' in df_ov.columns: overview_cols.append('Country')
            if 'City' in df_ov.columns: overview_cols.append('City')
            st.dataframe(df_ov[overview_cols].head(50), use_container_width=True)

    # =========================================================================
    # TAB 2: RFM ANALYSIS
    # =========================================================================
    if active_view == "📈 RFM Analysis":
        st.header("📈 RFM (Recency, Frequency, Monetary) Analysis")
        st.markdown("Quantify customer value and engagement through behavioral quintile scoring and segment classification.")

        # --- DYNAMIC SLICERS FOR TAB 2 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ RFM Metric Dynamic Slicers</span>
                <span class="slicer-badge">Interactive Quintile Slicing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: RFM Metric Slicers
        rfm_s1, rfm_s2, rfm_s3, rfm_s4 = st.columns(4)
        with rfm_s1:
            min_r, max_r = safe_range(customer_df['Recency'], 0, 365)
            rfm_r_range = st.slider("Recency Range (Days):", min_value=float(min_r), max_value=float(max_r), value=(float(min_r), float(max_r)), key="rfm_r_range")
        with rfm_s2:
            min_f, max_f = safe_range(customer_df['Frequency'], 1, 50)
            rfm_f_range = st.slider("Frequency Range (Orders):", min_value=float(min_f), max_value=float(max_f), value=(float(min_f), float(max_f)), key="rfm_f_range")
        with rfm_s3:
            min_m, max_m = safe_range(customer_df['Monetary'], 0, 5000)
            rfm_m_range = st.slider("Monetary Value ($):", min_value=float(min_m), max_value=float(max_m), value=(float(min_m), float(max_m)), key="rfm_m_range")
        with rfm_s4:
            all_rfm_labels = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if 'Simplified_RFM_Segment' in customer_df.columns else []
            rfm_seg_focus = st.multiselect("Segment Focus:", options=all_rfm_labels, default=all_rfm_labels, key="rfm_seg_focus")

        # Row 2: Geographic Slicers
        rfm_s5, rfm_s6, rfm_s7, rfm_s8 = st.columns(4)
        with rfm_s5:
            all_rfm_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            rfm_sel_state = st.multiselect("State:", options=all_rfm_states, default=all_rfm_states, key="rfm_sel_state")
        with rfm_s6:
            all_rfm_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            rfm_sel_region = st.multiselect("Region:", options=all_rfm_regions, default=all_rfm_regions, key="rfm_sel_region")
        with rfm_s7:
            all_rfm_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            rfm_sel_country = st.multiselect("Country:", options=all_rfm_countries, default=all_rfm_countries, key="rfm_sel_country")
        with rfm_s8:
            all_rfm_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            rfm_sel_city = st.multiselect("City:", options=all_rfm_cities, default=all_rfm_cities, key="rfm_sel_city")

        # Apply RFM Slicers
        df_rfm = customer_df[
            (customer_df['Recency'] >= rfm_r_range[0]) & (customer_df['Recency'] <= rfm_r_range[1]) &
            (customer_df['Frequency'] >= rfm_f_range[0]) & (customer_df['Frequency'] <= rfm_f_range[1]) &
            (customer_df['Monetary'] >= rfm_m_range[0]) & (customer_df['Monetary'] <= rfm_m_range[1])
        ]
        if rfm_seg_focus:
            df_rfm = df_rfm[df_rfm['Simplified_RFM_Segment'].isin(rfm_seg_focus)]
        if 'State' in df_rfm.columns and rfm_sel_state:
            df_rfm = df_rfm[df_rfm['State'].astype(str).isin(rfm_sel_state)]
        if 'Region' in df_rfm.columns and rfm_sel_region:
            df_rfm = df_rfm[df_rfm['Region'].astype(str).isin(rfm_sel_region)]
        if 'Country' in df_rfm.columns and rfm_sel_country:
            df_rfm = df_rfm[df_rfm['Country'].astype(str).isin(rfm_sel_country)]
        if 'City' in df_rfm.columns and rfm_sel_city:
            df_rfm = df_rfm[df_rfm['City'].astype(str).isin(rfm_sel_city)]

        st.caption(f"🔍 **Active Slicer Scope:** Displaying **{len(df_rfm):,}** of **{len(customer_df):,}** customers ({len(df_rfm)/max(len(customer_df),1):.1%})")

        if df_rfm.empty:
            st.warning("⚠️ No customer records match the active RFM criteria. Please broaden your filter sliders.")
        else:
            # --- TIER 1: KPI CARDS ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> RFM Behavioral Metrics</div>', unsafe_allow_html=True)
            champions_cnt = len(df_rfm[df_rfm['Simplified_RFM_Segment'] == 'Champions'])
            loyal_cnt = len(df_rfm[df_rfm['Simplified_RFM_Segment'] == 'Loyal Customers'])
            at_risk_cnt = len(df_rfm[df_rfm['Simplified_RFM_Segment'].isin(['Hibernating', 'Need Attention'])])

            rfm_kpi1, rfm_kpi2, rfm_kpi3, rfm_kpi4, rfm_kpi5 = st.columns(5)
            with rfm_kpi1:
                st.metric("Champions", f"{champions_cnt:,}", delta=f"{(champions_cnt/len(df_rfm)):.1%}")
            with rfm_kpi2:
                st.metric("Loyal Customers", f"{loyal_cnt:,}", delta=f"{(loyal_cnt/len(df_rfm)):.1%}")
            with rfm_kpi3:
                st.metric("At-Risk / Hibernating", f"{at_risk_cnt:,}", delta=f"{(at_risk_cnt/len(df_rfm)):.1%}", delta_color="inverse")
            with rfm_kpi4:
                st.metric("Median Recency", f"{df_rfm['Recency'].median():.0f} days")
            with rfm_kpi5:
                st.metric("Avg Order Frequency", f"{df_rfm['Frequency'].mean():.1f} orders")

            # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Segment & Metric Distribution Visualizations</div>', unsafe_allow_html=True)
            rfm_g_r1_c1, rfm_g_r1_c2 = st.columns(2)
            with rfm_g_r1_c1:
                rfm_seg_counts = df_rfm['Simplified_RFM_Segment'].value_counts().reset_index()
                rfm_seg_counts.columns = ['Segment', 'Customer Count']
                fig_rfm_dist = px.bar(
                    rfm_seg_counts, x='Segment', y='Customer Count',
                    color='Segment',
                    title="Customer Count by Simplified RFM Segment",
                    template="plotly_dark",
                    color_discrete_map=SEGMENT_COLORS
                )
                style_plotly_fig(fig_rfm_dist, height=340, show_legend=False, legend_title="Segment")
                st.plotly_chart(fig_rfm_dist, use_container_width=True, theme=None)

            with rfm_g_r1_c2:
                fig_rfm_pie = px.pie(
                    rfm_seg_counts, names='Segment', values='Customer Count',
                    title="RFM Segment Share of Filtered Base",
                    hole=0.4,
                    template="plotly_dark",
                    color='Segment',
                    color_discrete_map=SEGMENT_COLORS
                )
                fig_rfm_pie.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(color="#e8eefc", size=11)
                )
                style_plotly_fig(fig_rfm_pie, height=340, show_legend=False)
                st.plotly_chart(fig_rfm_pie, use_container_width=True, theme=None)

            st.markdown("##### 📊 Sliced RFM Metrics Spread")
            rfm_hist_c1, rfm_hist_c2, rfm_hist_c3 = st.columns(3)
            with rfm_hist_c1:
                fig_rec = px.histogram(
                    df_rfm, x='Recency', nbins=30,
                    title='Recency Spread (Days)',
                    color_discrete_sequence=['#2563eb'],
                    template="plotly_dark"
                )
                style_plotly_fig(fig_rec, height=300, show_legend=False)
                st.plotly_chart(fig_rec, use_container_width=True, theme=None)

            with rfm_hist_c2:
                fig_freq = px.histogram(
                    df_rfm, x='Frequency', nbins=30,
                    title='Frequency Spread (Orders)',
                    color_discrete_sequence=['#059669'],
                    template="plotly_dark"
                )
                style_plotly_fig(fig_freq, height=300, show_legend=False)
                st.plotly_chart(fig_freq, use_container_width=True, theme=None)

            with rfm_hist_c3:
                fig_mon = px.histogram(
                    df_rfm, x='Monetary', nbins=30,
                    title='Monetary Spread ($ Spend)',
                    color_discrete_sequence=['#7c3aed'],
                    template="plotly_dark"
                )
                style_plotly_fig(fig_mon, height=300, show_legend=False)
                st.plotly_chart(fig_mon, use_container_width=True, theme=None)

            # Average Metrics Comparison Chart
            avg_rfm_segment = df_rfm.groupby('Simplified_RFM_Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
            fig_avg_rfm = go.Figure(data=[
                go.Bar(name='Recency (Days)', x=avg_rfm_segment['Simplified_RFM_Segment'], y=avg_rfm_segment['Recency'], marker_color='#2563eb'),
                go.Bar(name='Frequency (Orders)', x=avg_rfm_segment['Simplified_RFM_Segment'], y=avg_rfm_segment['Frequency'], marker_color='#059669'),
                go.Bar(name='Monetary ($ Spend / 10)', x=avg_rfm_segment['Simplified_RFM_Segment'], y=avg_rfm_segment['Monetary'] / 10, marker_color='#d97706')
            ])
            fig_avg_rfm.update_layout(
                title="Comparison of Normalized RFM Profile by Simplified Segment",
                barmode='group',
                template="plotly_dark"
            )
            style_plotly_fig(fig_avg_rfm, height=360, show_legend=False, legend_title="Metric", legend_orientation="h")
            st.plotly_chart(fig_avg_rfm, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Customer Footprint Map")
            rfm_map, rfm_level = build_geo_map(df_rfm, value_col='Monetary', agg='sum', title="Revenue Footprint by Geography")
            if rfm_map is not None:
                st.plotly_chart(rfm_map, use_container_width=True, theme=None)
            else:
                st.info("Add Country or US State codes to render the footprint map.")

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> RFM Segment Summary & Detailed Customer Scores</div>', unsafe_allow_html=True)
            st.markdown("##### 📋 Segment Profile Matrix")
            rfm_matrix = df_rfm.groupby('Simplified_RFM_Segment').agg(
                Customers=('CustomerID', 'count'),
                Mean_Recency=('Recency', 'mean'),
                Mean_Frequency=('Frequency', 'mean'),
                Mean_Monetary=('Monetary', 'mean'),
                Min_Monetary=('Monetary', 'min'),
                Max_Monetary=('Monetary', 'max')
            ).round(2).reset_index()
            st.dataframe(rfm_matrix, use_container_width=True)

            st.markdown("##### 🔍 Sortable Customer RFM Score Table")
            rfm_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Simplified_RFM_Segment']
            if 'R_Score' in df_rfm.columns: rfm_cols.extend(['R_Score', 'F_Score', 'M_Score', 'RFM_Segment'])
            for geo_col in ['State', 'Region', 'Country', 'City']:
                if geo_col in df_rfm.columns:
                    rfm_cols.append(geo_col)
            st.dataframe(df_rfm[rfm_cols].sort_values('Recency'), use_container_width=True)

    # =========================================================================
    # TAB 3: DEMOGRAPHIC PROFILING
    # =========================================================================
    if active_view == "👥 Demographic Profiling":
        st.header("👥 Demographic Profiling & Cohort Analysis")
        st.markdown("Analyze customer lifetime value and churn vulnerability across age cohorts, genders, and geographic locations.")

        # --- DYNAMIC SLICERS FOR TAB 3 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ Demographic Dynamic Slicers</span>
                <span class="slicer-badge">Cohort Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Basic Demographics
        demo_s1, demo_s2, demo_s3, demo_s4 = st.columns(4)
        with demo_s1:
            all_genders = sorted(customer_df['Gender'].dropna().astype(str).unique()) if 'Gender' in customer_df.columns else []
            demo_sel_gender = st.multiselect("Gender:", options=all_genders, default=all_genders, key="demo_sel_gender")
        with demo_s2:
            min_age, max_age = safe_range(customer_df['Age'], 18, 80)
            demo_age_range = st.slider("Age Range (Years):", min_value=float(min_age), max_value=float(max_age), value=(float(min_age), float(max_age)), key="demo_age_range")
        with demo_s3:
            all_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            demo_sel_state = st.multiselect("State:", options=all_states, default=all_states, key="demo_sel_state")
        with demo_s4:
            demo_churn_filter = st.selectbox("Churn Cohort:", options=["All Customers", "Churned Only (Flag=1)", "Retained Only (Flag=0)"], key="demo_churn_filter")

        # Row 2: Geographic Fields (State, Region, Country, City)
        demo_s5, demo_s6, demo_s7, demo_s8 = st.columns(4)
        with demo_s5:
            all_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            demo_sel_region = st.multiselect("Region:", options=all_regions, default=all_regions, key="demo_sel_region")
        with demo_s6:
            all_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            demo_sel_country = st.multiselect("Country:", options=all_countries, default=all_countries, key="demo_sel_country")
        with demo_s7:
            all_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            demo_sel_city = st.multiselect("City:", options=all_cities, default=all_cities, key="demo_sel_city")
        with demo_s8:
            st.write("")  # Placeholder for alignment

        # Apply Demographic Slicers
        df_demo = customer_df.copy()
        if 'Gender' in df_demo.columns and demo_sel_gender:
            df_demo = df_demo[df_demo['Gender'].astype(str).isin(demo_sel_gender)]
        if 'Age' in df_demo.columns and df_demo['Age'].notna().any():
            df_demo = df_demo[(df_demo['Age'] >= demo_age_range[0]) & (df_demo['Age'] <= demo_age_range[1])]
        if 'State' in df_demo.columns and demo_sel_state:
            df_demo = df_demo[df_demo['State'].astype(str).isin(demo_sel_state)]
        if 'Region' in df_demo.columns and demo_sel_region:
            df_demo = df_demo[df_demo['Region'].astype(str).isin(demo_sel_region)]
        if 'Country' in df_demo.columns and demo_sel_country:
            df_demo = df_demo[df_demo['Country'].astype(str).isin(demo_sel_country)]
        if 'City' in df_demo.columns and demo_sel_city:
            df_demo = df_demo[df_demo['City'].astype(str).isin(demo_sel_city)]
        if 'HasChurned' in df_demo.columns:
            if demo_churn_filter == "Churned Only (Flag=1)":
                df_demo = df_demo[df_demo['HasChurned'] == 1]
            elif demo_churn_filter == "Retained Only (Flag=0)":
                df_demo = df_demo[df_demo['HasChurned'] == 0]

        st.caption(f"🔍 **Active Slicer Scope:** Displaying **{len(df_demo):,}** of **{len(customer_df):,}** customers ({len(df_demo)/max(len(customer_df),1):.1%})")

        if df_demo.empty:
            st.warning("⚠️ No customer records match the active demographic criteria. Please broaden your slicers.")
        else:
            # --- TIER 1: KPI CARDS ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> Demographic Health KPIs</div>', unsafe_allow_html=True)
            demo_kpi1, demo_kpi2, demo_kpi3, demo_kpi4 = st.columns(4)
            with demo_kpi1:
                avg_age = f"{df_demo['Age'].mean():.1f} yrs" if 'Age' in df_demo.columns and df_demo['Age'].notna().any() else "N/A"
                st.metric("Avg Customer Age", avg_age)
            with demo_kpi2:
                top_gender = df_demo['Gender'].mode()[0] if 'Gender' in df_demo.columns and not df_demo['Gender'].empty else "N/A"
                st.metric("Dominant Gender", str(top_gender))
            with demo_kpi3:
                top_state = df_demo['State'].mode()[0] if 'State' in df_demo.columns and not df_demo['State'].empty else "N/A"
                st.metric("Top State", str(top_state))
            with demo_kpi4:
                if 'AgeGroup' in df_demo.columns and not df_demo['AgeGroup'].empty:
                    top_age_grp = df_demo['AgeGroup'].mode()[0]
                    st.metric("Largest Age Cohort", str(top_age_grp))
                else:
                    st.metric("Largest Age Cohort", "N/A")

            # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Demographic Churn & Revenue Visuals</div>', unsafe_allow_html=True)

            # Row 1: Gender Analysis
            demo_r1_c1, demo_r1_c2 = st.columns(2)
            with demo_r1_c1:
                if 'Gender' in df_demo.columns and not df_demo.empty:
                    churn_gender = df_demo.groupby('Gender')['Churn_Probability'].mean().reset_index()
                    fig_churn_gen = px.bar(
                        churn_gender, x='Gender', y='Churn_Probability',
                        color='Gender',
                        title="Average Churn Probability by Gender",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=GENDER_COLORS
                    )
                    style_plotly_fig(fig_churn_gen, height=340, show_legend=False, legend_title="Gender")
                    st.plotly_chart(fig_churn_gen, use_container_width=True, theme=None)
                else:
                    st.info("Gender data not present in active scope.")

            with demo_r1_c2:
                if 'Gender' in df_demo.columns and not df_demo.empty:
                    spend_gender = df_demo.groupby('Gender')['Monetary'].mean().reset_index()
                    fig_spend_gen = px.bar(
                        spend_gender, x='Gender', y='Monetary',
                        color='Gender',
                        title="Average Customer Spend by Gender ($)",
                        labels={'Monetary': 'Avg Spend ($)'},
                        template="plotly_dark",
                        color_discrete_map=GENDER_COLORS
                    )
                    style_plotly_fig(fig_spend_gen, height=340, show_legend=False, legend_title="Gender")
                    st.plotly_chart(fig_spend_gen, use_container_width=True, theme=None)

            # Row 2: Age Group Analysis
            demo_r2_c1, demo_r2_c2 = st.columns(2)
            with demo_r2_c1:
                if 'AgeGroup' in df_demo.columns and not df_demo.empty:
                    churn_age = df_demo.groupby('AgeGroup', observed=False)['Churn_Probability'].mean().reset_index()
                    fig_churn_age = px.bar(
                        churn_age, x='AgeGroup', y='Churn_Probability',
                        color='AgeGroup',
                        title="Churn Probability by Age Cohort",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=AGE_GROUP_COLORS
                    )
                    style_plotly_fig(fig_churn_age, height=340, show_legend=False, legend_title="Age Cohort")
                    st.plotly_chart(fig_churn_age, use_container_width=True, theme=None)
                else:
                    st.info("Age group metrics not calculated in active scope.")

            with demo_r2_c2:
                if 'AgeGroup' in df_demo.columns and not df_demo.empty:
                    count_age = df_demo.groupby('AgeGroup', observed=False)['CustomerID'].count().reset_index()
                    count_age.columns = ['AgeGroup', 'Customer Count']
                    fig_count_age = px.bar(
                        count_age, x='AgeGroup', y='Customer Count',
                        color='AgeGroup',
                        title="Customer Volume by Age Cohort",
                        template="plotly_dark",
                        color_discrete_map=AGE_GROUP_COLORS
                    )
                    style_plotly_fig(fig_count_age, height=340, show_legend=False, legend_title="Age Cohort")
                    st.plotly_chart(fig_count_age, use_container_width=True, theme=None)

            # Row 3: State Analysis
            demo_r3_c1, demo_r3_c2 = st.columns(2)
            with demo_r3_c1:
                if 'State' in df_demo.columns and not df_demo.empty:
                    churn_state = df_demo.groupby('State')['Churn_Probability'].mean().reset_index()
                    fig_churn_state = px.bar(
                        churn_state, x='State', y='Churn_Probability',
                        color='State',
                        title="Churn Probability by State",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=STATE_COLORS
                    )
                    style_plotly_fig(fig_churn_state, height=340, show_legend=False, legend_title="State")
                    st.plotly_chart(fig_churn_state, use_container_width=True, theme=None)
                else:
                    st.info("State data not present.")

            with demo_r3_c2:
                if 'State' in df_demo.columns and not df_demo.empty:
                    state_rev = df_demo.groupby('State')['Monetary'].sum().reset_index()
                    fig_state_rev = px.pie(
                        state_rev, names='State', values='Monetary',
                        title="Revenue Contribution Share by State",
                        hole=0.4,
                        template="plotly_dark",
                        color='State',
                        color_discrete_map=STATE_COLORS
                    )
                    fig_state_rev.update_traces(
                        textinfo="label+percent",
                        textposition="outside",
                        textfont=dict(color="#e8eefc", size=11)
                    )
                    style_plotly_fig(fig_state_rev, height=340, show_legend=False)
                    st.plotly_chart(fig_state_rev, use_container_width=True, theme=None)

            # Row 5: Region Analysis
            demo_r5_c1, demo_r5_c2 = st.columns(2)
            with demo_r5_c1:
                if 'Region' in df_demo.columns and not df_demo.empty:
                    churn_region = df_demo.groupby('Region')['Churn_Probability'].mean().reset_index()
                    fig_churn_region = px.bar(
                        churn_region, x='Region', y='Churn_Probability',
                        color='Region',
                        title="Churn Probability by Region",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=REGION_COLORS
                    )
                    style_plotly_fig(fig_churn_region, height=340, show_legend=False, legend_title="Region")
                    st.plotly_chart(fig_churn_region, use_container_width=True, theme=None)
                else:
                    st.info("Region data not present.")

            with demo_r5_c2:
                if 'Region' in df_demo.columns and not df_demo.empty:
                    region_rev = df_demo.groupby('Region')['Monetary'].sum().reset_index()
                    fig_region_rev = px.pie(
                        region_rev, names='Region', values='Monetary',
                        title="Revenue Contribution Share by Region",
                        hole=0.4,
                        template="plotly_dark",
                        color='Region',
                        color_discrete_map=REGION_COLORS
                    )
                    fig_region_rev.update_traces(
                        textinfo="label+percent",
                        textposition="outside",
                        textfont=dict(color="#e8eefc", size=11)
                    )
                    style_plotly_fig(fig_region_rev, height=340, show_legend=False)
                    st.plotly_chart(fig_region_rev, use_container_width=True, theme=None)

            # Row 6: Country Analysis
            demo_r6_c1, demo_r6_c2 = st.columns(2)
            with demo_r6_c1:
                if 'Country' in df_demo.columns and not df_demo.empty:
                    churn_country = df_demo.groupby('Country')['Churn_Probability'].mean().reset_index()
                    fig_churn_country = px.bar(
                        churn_country, x='Country', y='Churn_Probability',
                        color='Country',
                        title="Churn Probability by Country",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=COUNTRY_COLORS
                    )
                    style_plotly_fig(fig_churn_country, height=340, show_legend=False, legend_title="Country")
                    st.plotly_chart(fig_churn_country, use_container_width=True, theme=None)
                else:
                    st.info("Country data not present.")

            with demo_r6_c2:
                if 'Country' in df_demo.columns and not df_demo.empty:
                    country_rev = df_demo.groupby('Country')['Monetary'].sum().reset_index()
                    fig_country_rev = px.pie(
                        country_rev, names='Country', values='Monetary',
                        title="Revenue Contribution Share by Country",
                        hole=0.4,
                        template="plotly_dark",
                        color='Country',
                        color_discrete_map=COUNTRY_COLORS
                    )
                    fig_country_rev.update_traces(
                        textinfo="label+percent",
                        textposition="outside",
                        textfont=dict(color="#e8eefc", size=11)
                    )
                    style_plotly_fig(fig_country_rev, height=340, show_legend=False)
                    st.plotly_chart(fig_country_rev, use_container_width=True, theme=None)

            # Row 7: City Analysis
            demo_r7_c1, demo_r7_c2 = st.columns(2)
            with demo_r7_c1:
                if 'City' in df_demo.columns and not df_demo.empty:
                    churn_city = df_demo.groupby('City')['Churn_Probability'].mean().reset_index()
                    fig_churn_city = px.bar(
                        churn_city, x='City', y='Churn_Probability',
                        color='City',
                        title="Churn Probability by City",
                        labels={'Churn_Probability': 'Avg Churn Risk'},
                        template="plotly_dark",
                        color_discrete_map=CITY_COLORS
                    )
                    style_plotly_fig(fig_churn_city, height=340, show_legend=False, legend_title="City")
                    st.plotly_chart(fig_churn_city, use_container_width=True, theme=None)
                else:
                    st.info("City data not present.")

            with demo_r7_c2:
                if 'City' in df_demo.columns and not df_demo.empty:
                    city_rev = df_demo.groupby('City')['Monetary'].sum().reset_index()
                    fig_city_rev = px.pie(
                        city_rev, names='City', values='Monetary',
                        title="Revenue Contribution Share by City",
                        hole=0.4,
                        template="plotly_dark",
                        color='City',
                        color_discrete_map=CITY_COLORS
                    )
                    fig_city_rev.update_traces(
                        textinfo="label+percent",
                        textposition="outside",
                        textfont=dict(color="#e8eefc", size=11)
                    )
                    style_plotly_fig(fig_city_rev, height=340, show_legend=False)
                    st.plotly_chart(fig_city_rev, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Demographic Geography Map")
            demo_map_metric = st.selectbox(
                "Map metric:", ["Monetary", "Churn_Probability", "Frequency"],
                index=0, key="demo_map_metric",
            )
            demo_agg = "mean" if demo_map_metric == "Churn_Probability" else "sum"
            demo_map, demo_level = build_geo_map(
                df_demo, value_col=demo_map_metric, agg=demo_agg,
                title=f"{'Avg Churn Risk' if demo_map_metric=='Churn_Probability' else 'Total '+demo_map_metric} by Geography",
            )
            if demo_map is not None:
                st.plotly_chart(demo_map, use_container_width=True, theme=None)
            else:
                st.info("Add Country or US State codes to render the geography map.")

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Demographic Breakdown Tables</div>', unsafe_allow_html=True)
            st.markdown("##### 📋 Demographic Aggregate Summary")
            demo_tabs = st.tabs(["Gender Summary", "Age Cohort Summary", "State Summary", "Region Summary", "Country Summary", "City Summary"])
            with demo_tabs[0]:
                if 'Gender' in df_demo.columns:
                    g_tab = df_demo.groupby('Gender').agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(g_tab, use_container_width=True)
            with demo_tabs[1]:
                if 'AgeGroup' in df_demo.columns:
                    a_tab = df_demo.groupby('AgeGroup', observed=False).agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(a_tab, use_container_width=True)
            with demo_tabs[2]:
                if 'State' in df_demo.columns:
                    s_tab = df_demo.groupby('State').agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(s_tab, use_container_width=True)
            with demo_tabs[3]:
                if 'Region' in df_demo.columns:
                    r_tab = df_demo.groupby('Region').agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(r_tab, use_container_width=True)
            with demo_tabs[4]:
                if 'Country' in df_demo.columns:
                    c_tab = df_demo.groupby('Country').agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(c_tab, use_container_width=True)
            with demo_tabs[5]:
                if 'City' in df_demo.columns:
                    cy_tab = df_demo.groupby('City').agg(
                        Count=('CustomerID', 'count'),
                        Avg_Recency=('Recency', 'mean'),
                        Avg_Frequency=('Frequency', 'mean'),
                        Avg_Monetary=('Monetary', 'mean'),
                        Avg_Churn_Risk=('Churn_Probability', 'mean')
                    ).round(2).reset_index()
                    st.dataframe(cy_tab, use_container_width=True)

            st.markdown("##### 👥 Sliced Customer Records with Demographics")
            demo_cols = ['CustomerID', 'Age', 'Gender', 'State', 'Region', 'Country', 'City', 'Recency', 'Frequency', 'Monetary', 'Churn_Probability']
            avail_demo_cols = [c for c in demo_cols if c in df_demo.columns]
            st.dataframe(df_demo[avail_demo_cols].head(50), use_container_width=True)

    # =========================================================================
    # TAB 4: ADVANCED SEGMENTATION (DEEP LEARNING & ENSEMBLE)
    # =========================================================================
    if active_view == "🤖 Advanced Segmentation":
        st.header("🤖 Advanced AI Segmentation (Autoencoder & Ensemble Clustering)")
        st.markdown("Extract compressed latent behavioral features via Deep Autoencoders and discover robust customer clusters using Multi-Algorithm Consensus Voting.")

        # Active Segment Column detection
        seg_col = 'Ensemble_Segment' if 'Ensemble_Segment' in customer_df.columns else ('Advanced_Segment' if 'Advanced_Segment' in customer_df.columns else 'Simplified_RFM_Segment')

        # --- DYNAMIC SLICERS FOR TAB 4 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ AI Cluster & Feature Dynamic Slicers</span>
                <span class="slicer-badge">Consensus Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Basic Filters
        adv_s1, adv_s2, adv_s3, adv_s4 = st.columns(4)
        with adv_s1:
            all_clusters = sorted(customer_df[seg_col].dropna().unique())
            adv_sel_clusters = st.multiselect(f"Filter by {seg_col}:", options=all_clusters, default=all_clusters, key="adv_sel_clusters")
        with adv_s2:
            all_rfm_adv = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if 'Simplified_RFM_Segment' in customer_df.columns else []
            adv_sel_rfm = st.multiselect("RFM Segment Overlay:", options=all_rfm_adv, default=all_rfm_adv, key="adv_sel_rfm")
        with adv_s3:
            min_m_adv, max_m_adv = safe_range(customer_df['Monetary'])
            adv_spend_filter = st.slider("Min Customer Spend ($):", min_value=float(min_m_adv), max_value=float(max_m_adv), value=(float(min_m_adv), float(max_m_adv)), key="adv_spend_filter")
        with adv_s4:
            outlier_opt = st.selectbox("Anomaly / Outlier Slicer:", options=["All Clusters", "Exclude Outliers (Cluster != -1)", "Outliers Only (Cluster == -1)"], key="adv_outlier_opt")

        # Row 2: Geographic Filters
        adv_s5, adv_s6, adv_s7, adv_s8 = st.columns(4)
        with adv_s5:
            all_adv_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            adv_sel_state = st.multiselect("State:", options=all_adv_states, default=all_adv_states, key="adv_sel_state")
        with adv_s6:
            all_adv_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            adv_sel_region = st.multiselect("Region:", options=all_adv_regions, default=all_adv_regions, key="adv_sel_region")
        with adv_s7:
            all_adv_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            adv_sel_country = st.multiselect("Country:", options=all_adv_countries, default=all_adv_countries, key="adv_sel_country")
        with adv_s8:
            all_adv_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            adv_sel_city = st.multiselect("City:", options=all_adv_cities, default=all_adv_cities, key="adv_sel_city")

        # Apply Advanced Slicer
        df_adv = customer_df.copy()
        if adv_sel_clusters:
            df_adv = df_adv[df_adv[seg_col].isin(adv_sel_clusters)]
        if adv_sel_rfm:
            df_adv = df_adv[df_adv['Simplified_RFM_Segment'].isin(adv_sel_rfm)]
        if 'Monetary' in df_adv.columns:
            df_adv = df_adv[(df_adv['Monetary'] >= adv_spend_filter[0]) & (df_adv['Monetary'] <= adv_spend_filter[1])]
        if 'State' in df_adv.columns and adv_sel_state:
            df_adv = df_adv[df_adv['State'].astype(str).isin(adv_sel_state)]
        if 'Region' in df_adv.columns and adv_sel_region:
            df_adv = df_adv[df_adv['Region'].astype(str).isin(adv_sel_region)]
        if 'Country' in df_adv.columns and adv_sel_country:
            df_adv = df_adv[df_adv['Country'].astype(str).isin(adv_sel_country)]
        if 'City' in df_adv.columns and adv_sel_city:
            df_adv = df_adv[df_adv['City'].astype(str).isin(adv_sel_city)]
        if -1 in df_adv[seg_col].values:
            if outlier_opt == "Exclude Outliers (Cluster != -1)":
                df_adv = df_adv[df_adv[seg_col] != -1]
            elif outlier_opt == "Outliers Only (Cluster == -1)":
                df_adv = df_adv[df_adv[seg_col] == -1]

        st.caption(f"🔍 **Active Slicer Scope:** Displaying **{len(df_adv):,}** of **{len(customer_df):,}** customers ({len(df_adv)/max(len(customer_df),1):.1%})")

        # --- TIER 1: KPI CARDS ---
        st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> AI Model & Clustering KPIs</div>', unsafe_allow_html=True)
        adv_kpi1, adv_kpi2, adv_kpi3, adv_kpi4 = st.columns(4)
        with adv_kpi1:
            latent_status = f"{customer_df.filter(like='Latent_').shape[1]} dims" if 'Latent_0' in customer_df.columns else "Not Trained"
            st.metric("Autoencoder Latent Dims", latent_status)
        with adv_kpi2:
            num_clusters = len(df_adv[seg_col].unique()) if not df_adv.empty else 0
            st.metric("Active AI Clusters in Scope", str(num_clusters))
        with adv_kpi3:
            ens_status = "Ensemble Active" if 'Ensemble_Segment' in customer_df.columns else "Baseline Active"
            st.metric("Segmentation Mode", ens_status)
        with adv_kpi4:
            outliers_cnt = len(df_adv[df_adv[seg_col] == -1]) if -1 in df_adv[seg_col].values else 0
            st.metric("Detected Outliers in Scope", f"{outliers_cnt:,}")

        # Training Controls
        st.markdown("##### ⚙️ Model Training & Parameter Configurations")
        ctrl_c1, ctrl_c2 = st.columns(2)
        with ctrl_c1:
            with st.expander("🧠 1. Deep Autoencoder Hyperparameters", expanded=False):
                ae_enc = st.slider("Encoding Hidden Layer Neurons", 4, 32, 16, 2, key="ae_enc_slider")
                ae_lat = st.slider("Bottleneck Latent Dimension", 2, 8, 3, 1, key="ae_lat_slider")
                ae_ep = st.slider("Training Epochs", 20, 200, 80, 10, key="ae_ep_slider")
                ae_features = st.multiselect(
                    "Input Features for Autoencoder:",
                    options=['Recency', 'Frequency', 'Monetary', 'Age'],
                    default=['Recency', 'Frequency', 'Monetary'],
                    key="ae_feat_select"
                )
                if st.button("🚀 Train Autoencoder", key="btn_train_ae"):
                    if get_tensorflow() is None:
                        st.error("TensorFlow is not installed.")
                    elif len(ae_features) < 2:
                        st.warning("Please select at least 2 features.")
                    else:
                        features_ae = customer_df[ae_features].dropna()
                        latent_df, ae_model = train_autoencoder(
                            features_ae, encoding_dim=ae_enc, latent_dim=ae_lat, epochs=ae_ep
                        )
                        if latent_df is not None:
                            st.session_state['latent_features'] = latent_df
                            customer_df = customer_df.drop(columns=[c for c in customer_df.columns if c.startswith('Latent_')], errors='ignore').join(latent_df, how='left')
                            st.success(f"Autoencoder successfully trained! Extracted {ae_lat} latent features.")
                            st.rerun()

        with ctrl_c2:
            with st.expander("🧩 2. Ensemble Clustering Hyperparameters", expanded=False):
                ec_k = st.slider("Target Clusters (K-Means & Agglomerative)", 2, 8, 4, 1, key="ec_k_slider")
                ec_eps = st.slider("DBSCAN Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, key="ec_eps_slider")
                ec_min_samples = st.slider("DBSCAN Min Samples", 2, 20, 5, 1, key="ec_min_slider")

                ec_feat_opts = ['Recency', 'Frequency', 'Monetary', 'Age']
                if 'Latent_0' in customer_df.columns:
                    ec_feat_opts += [c for c in customer_df.columns if c.startswith('Latent_')]

                ec_selected_feats = st.multiselect(
                    "Clustering Features:",
                    options=ec_feat_opts,
                    default=['Recency', 'Frequency', 'Monetary'] if not 'Latent_0' in customer_df.columns else ['Latent_0', 'Latent_1', 'Latent_2'] if 'Latent_2' in customer_df.columns else ['Latent_0', 'Latent_1'],
                    key="ec_feat_select"
                )
                if st.button("🚀 Run Multi-Algorithm Ensemble", key="btn_run_ensemble"):
                    if len(ec_selected_feats) < 2:
                        st.warning("Please select at least 2 features.")
                    else:
                        features_ec = customer_df[ec_selected_feats].dropna()
                        ensemble_df, cluster_results = ensemble_clustering(
                            features_ec, n_clusters=ec_k, dbscan_eps=ec_eps, dbscan_min=ec_min_samples
                        )
                        st.session_state['ensemble_segments'] = ensemble_df[['Ensemble_Segment']]
                        customer_df = customer_df.drop(columns=['Ensemble_Segment'], errors='ignore').join(ensemble_df[['Ensemble_Segment']], how='left')
                        st.session_state['ensemble_trained'] = True
                        st.success("Ensemble consensus clustering completed!")
                        st.rerun()

        if df_adv.empty:
            st.warning("⚠️ No customer records match the active clustering slicers.")
        else:
            # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Latent Space & Cluster Profiles</div>', unsafe_allow_html=True)

            # Row 1: Latent Space & Reconstruction
            adv_g_r1_c1, adv_g_r1_c2 = st.columns(2)
            with adv_g_r1_c1:
                if 'Latent_0' in df_adv.columns:
                    color_col = seg_col if seg_col in df_adv.columns else 'Simplified_RFM_Segment'
                    fig_latent = plot_latent_space(df_adv, color_by=color_col, title="Autoencoder Latent Representation (Sliced)")
                    fig_latent.update_layout(height=360)
                    st.plotly_chart(fig_latent, use_container_width=True, theme=None)
                else:
                    st.info("Train the Autoencoder above to project customer behavior into a compressed latent manifold.")

            with adv_g_r1_c2:
                adv_counts = df_adv[seg_col].value_counts().sort_index().reset_index()
                adv_counts.columns = ['Segment', 'Customer Count']
                fig_adv_counts = px.bar(
                    adv_counts, x='Segment', y='Customer Count',
                    color='Segment',
                    title=f"Customer Distribution by {seg_col}",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                style_plotly_fig(fig_adv_counts, height=360, show_legend=False, legend_title=seg_col)
                st.plotly_chart(fig_adv_counts, use_container_width=True, theme=None)

            # Row 2: Cluster Metrics
            adv_g_r2_c1, adv_g_r2_c2 = st.columns(2)
            with adv_g_r2_c1:
                avg_adv = df_adv.groupby(seg_col)[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
                fig_adv_metrics = go.Figure(data=[
                    go.Bar(name='Recency (days)', x=avg_adv[seg_col].astype(str), y=avg_adv['Recency'], marker_color='#3b82f6'),
                    go.Bar(name='Frequency', x=avg_adv[seg_col].astype(str), y=avg_adv['Frequency'], marker_color='#10b981'),
                    go.Bar(name='Monetary ($/10)', x=avg_adv[seg_col].astype(str), y=avg_adv['Monetary']/10, marker_color='#f59e0b')
                ])
                fig_adv_metrics.update_layout(
                    title=f"Average RFM Profile per {seg_col}",
                    barmode='group',
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                style_plotly_fig(fig_adv_metrics, height=360, show_legend=False, legend_title="RFM Metric", legend_orientation="h")
                st.plotly_chart(fig_adv_metrics, use_container_width=True, theme=None)

            with adv_g_r2_c2:
                churn_by_adv = df_adv.groupby(seg_col)['Churn_Probability'].mean().reset_index()
                fig_churn_adv = px.bar(
                    churn_by_adv, x=seg_col, y='Churn_Probability',
                    color=seg_col,
                    title=f"Mean Churn Risk by {seg_col}",
                    labels={'Churn_Probability': 'Avg Churn Risk'},
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                style_plotly_fig(fig_churn_adv, height=360, show_legend=False, legend_title=seg_col)
                st.plotly_chart(fig_churn_adv, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Cluster Geography")
            adv_map, adv_level = build_geo_map(df_adv, value_col='Monetary', agg='sum', title="Cluster Revenue Footprint by Geography")
            if adv_map is not None:
                st.plotly_chart(adv_map, use_container_width=True, theme=None)

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Cluster Profiles & Customer Assignment Table</div>', unsafe_allow_html=True)
            st.markdown(f"##### 📋 Cluster Summary Matrix ({seg_col})")
            cluster_summary = df_adv.groupby(seg_col).agg(
                Members=('CustomerID', 'count'),
                Mean_Recency=('Recency', 'mean'),
                Mean_Frequency=('Frequency', 'mean'),
                Mean_Monetary=('Monetary', 'mean'),
                Avg_Churn_Risk=('Churn_Probability', 'mean')
            ).round(2).reset_index()
            st.dataframe(cluster_summary, use_container_width=True)

            st.markdown("##### 👥 Customer Records with Advanced Clustering & Latent Scores")
            disp_adv_cols = ['CustomerID', seg_col, 'Simplified_RFM_Segment', 'Recency', 'Frequency', 'Monetary', 'Churn_Probability']
            for geo_col in ['State', 'Region', 'Country', 'City']:
                if geo_col in df_adv.columns:
                    disp_adv_cols.append(geo_col)
            if 'Latent_0' in df_adv.columns:
                disp_adv_cols += [c for c in df_adv.columns if c.startswith('Latent_')]
            st.dataframe(df_adv[[c for c in disp_adv_cols if c in df_adv.columns]].head(50), use_container_width=True)

    # =========================================================================
    # TAB 5: CHURN PREDICTION & RISK ANALYSIS
    # =========================================================================
    if active_view == "⚠️ Churn Prediction":
        st.header("⚠️ Predictive Churn Modeling & Risk Scoring")
        st.markdown("Forecast customer defection risk using ML and Deep Learning neural networks, and isolate cohorts requiring immediate retention.")

        # --- DYNAMIC SLICERS FOR TAB 5 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ Churn Risk Dynamic Slicers</span>
                <span class="slicer-badge">Predictive Risk Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Basic Filters
        cp_s1, cp_s2, cp_s3, cp_s4 = st.columns(4)
        with cp_s1:
            cp_prob_range = st.slider("Churn Probability Range:", 0.0, 1.0, (0.0, 1.0), 0.05, key="cp_prob_range")
        with cp_s2:
            cp_risk_tiers = st.multiselect("Risk Tier Slicer:", options=['High Risk', 'Medium Risk', 'Low Risk'], default=['High Risk', 'Medium Risk', 'Low Risk'], key="cp_risk_tiers")
        with cp_s3:
            all_cp_segs = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if 'Simplified_RFM_Segment' in customer_df.columns else []
            cp_sel_segs = st.multiselect("RFM Segment Filter:", options=all_cp_segs, default=all_cp_segs, key="cp_sel_segs")
        with cp_s4:
            min_m_cp, max_m_cp = safe_range(customer_df['Monetary'])
            cp_spend_range = st.slider("Customer Spend Filter ($):", min_value=float(min_m_cp), max_value=float(max_m_cp), value=(float(min_m_cp), float(max_m_cp)), key="cp_spend_range")

        # Row 2: Geographic Filters
        cp_s5, cp_s6, cp_s7, cp_s8 = st.columns(4)
        with cp_s5:
            all_cp_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            cp_sel_states = st.multiselect("State:", options=all_cp_states, default=all_cp_states, key="cp_sel_states")
        with cp_s6:
            all_cp_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            cp_sel_regions = st.multiselect("Region:", options=all_cp_regions, default=all_cp_regions, key="cp_sel_regions")
        with cp_s7:
            all_cp_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            cp_sel_countries = st.multiselect("Country:", options=all_cp_countries, default=all_cp_countries, key="cp_sel_countries")
        with cp_s8:
            all_cp_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            cp_sel_cities = st.multiselect("City:", options=all_cp_cities, default=all_cp_cities, key="cp_sel_cities")

        # Threshold sliders
        thresh_c1, thresh_c2 = st.columns(2)
        with thresh_c1:
            high_risk_thresh = st.slider("Set High Risk Threshold (prob >= X)", 0.5, 0.95, 0.70, 0.05, key="slider_high_risk")
        with thresh_c2:
            med_risk_thresh = st.slider("Set Medium Risk Threshold (prob >= Y)", 0.2, high_risk_thresh, 0.40, 0.05, key="slider_med_risk")

        # Dynamic Risk Tagging
        df_cp = customer_df.copy()
        df_cp['Risk_Level'] = 'Low Risk'
        df_cp.loc[df_cp['Churn_Probability'] >= med_risk_thresh, 'Risk_Level'] = 'Medium Risk'
        df_cp.loc[df_cp['Churn_Probability'] >= high_risk_thresh, 'Risk_Level'] = 'High Risk'

        # Apply Slicers
        df_cp = df_cp[
            (df_cp['Churn_Probability'] >= cp_prob_range[0]) & (df_cp['Churn_Probability'] <= cp_prob_range[1]) &
            (df_cp['Monetary'] >= cp_spend_range[0]) & (df_cp['Monetary'] <= cp_spend_range[1])
        ]
        if cp_risk_tiers:
            df_cp = df_cp[df_cp['Risk_Level'].isin(cp_risk_tiers)]
        if cp_sel_segs:
            df_cp = df_cp[df_cp['Simplified_RFM_Segment'].isin(cp_sel_segs)]
        if 'State' in df_cp.columns and cp_sel_states:
            df_cp = df_cp[df_cp['State'].astype(str).isin(cp_sel_states)]
        if 'Region' in df_cp.columns and cp_sel_regions:
            df_cp = df_cp[df_cp['Region'].astype(str).isin(cp_sel_regions)]
        if 'Country' in df_cp.columns and cp_sel_countries:
            df_cp = df_cp[df_cp['Country'].astype(str).isin(cp_sel_countries)]
        if 'City' in df_cp.columns and cp_sel_cities:
            df_cp = df_cp[df_cp['City'].astype(str).isin(cp_sel_cities)]

        st.caption(f"🔍 **Active Slicer Scope:** Displaying **{len(df_cp):,}** of **{len(customer_df):,}** customers ({len(df_cp)/max(len(customer_df),1):.1%})")

        # --- TIER 1: KPI CARDS ---
        st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> Churn & Portfolio Risk KPIs</div>', unsafe_allow_html=True)
        high_risk_df = df_cp[df_cp['Risk_Level'] == 'High Risk']
        med_risk_df = df_cp[df_cp['Risk_Level'] == 'Medium Risk']
        low_risk_df = df_cp[df_cp['Risk_Level'] == 'Low Risk']
        rev_at_risk = high_risk_df['Monetary'].sum() if not high_risk_df.empty else 0

        cp_kpi1, cp_kpi2, cp_kpi3, cp_kpi4, cp_kpi5 = st.columns(5)
        with cp_kpi1:
            avg_cp = df_cp['Churn_Probability'].mean() if not df_cp.empty else 0
            st.metric("Avg Churn Risk", f"{avg_cp:.1%}")
        with cp_kpi2:
            st.metric("High Risk Customers", f"{len(high_risk_df):,}", delta=f"{(len(high_risk_df)/max(len(df_cp),1)):.1%}", delta_color="inverse")
        with cp_kpi3:
            st.metric("Medium Risk", f"{len(med_risk_df):,}", delta=f"{(len(med_risk_df)/max(len(df_cp),1)):.1%}")
        with cp_kpi4:
            st.metric("Low Risk / Retained", f"{len(low_risk_df):,}", delta=f"{(len(low_risk_df)/max(len(df_cp),1)):.1%}")
        with cp_kpi5:
            st.metric("Revenue at High Risk", f"${rev_at_risk:,.2f}")

        # Deep Learning Training Expander
        with st.expander("🤖 Deep Learning Neural Network Churn Model Trainer", expanded=False):
            dl_features = st.multiselect(
                "Features for Deep Learning Churn Classifier:",
                options=['Recency', 'Frequency', 'Monetary', 'Age'] + [c for c in customer_df.columns if c.startswith('Latent_')],
                default=['Recency', 'Frequency', 'Monetary'],
                key="dl_churn_features"
            )
            if st.button("🚀 Train Deep Neural Network Churn Model", key="btn_train_dl_churn"):
                if get_tensorflow() is None:
                    st.error("TensorFlow not installed.")
                elif len(dl_features) < 1:
                    st.warning("Select at least 1 feature.")
                else:
                    dl_m, dl_metrics, dl_sc, dl_hist = train_churn_model(customer_df, dl_features, epochs=50)
                    if dl_m is not None:
                        customer_df['Churn_Probability'] = predict_churn_deep_learning(customer_df, dl_m, dl_sc, dl_features)
                        st.session_state['dl_model'] = dl_m
                        st.session_state['dl_scaler'] = dl_sc
                        st.session_state['dl_history'] = dl_hist.history
                        st.session_state['uploaded_data'] = customer_df
                        st.success(f"Deep Learning Model Trained! Test Accuracy: {dl_metrics['accuracy']:.4f}, AUC: {dl_metrics['auc']:.4f}")
                        st.rerun()

        if df_cp.empty:
            st.warning("⚠️ No customer records match the active churn prediction slicers.")
        else:
            # --- TIER 2: VISUAL GRAPHS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Churn Risk Distribution & Vulnerability Visuals</div>', unsafe_allow_html=True)
            cp_g_r1_c1, cp_g_r1_c2 = st.columns(2)
            with cp_g_r1_c1:
                fig_churn_hist = px.histogram(
                    df_cp, x='Churn_Probability', nbins=40,
                    title="Overall Churn Probability Distribution (Filtered)",
                    labels={'Churn_Probability': 'Predicted Churn Probability'},
                    color_discrete_sequence=['#ef4444'],
                    template="plotly_dark"
                )
                fig_churn_hist.add_vline(
                    x=high_risk_thresh, line_dash="dash", line_color="#b91c1c",
                    annotation_text="High Risk", annotation_font_color="#991b1b"
                )
                fig_churn_hist.add_vline(
                    x=med_risk_thresh, line_dash="dash", line_color="#d97706",
                    annotation_text="Medium Risk", annotation_font_color="#92400e"
                )
                style_plotly_fig(fig_churn_hist, height=350, show_legend=False)
                st.plotly_chart(fig_churn_hist, use_container_width=True, theme=None)

            with cp_g_r1_c2:
                risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
                risk_values = [len(low_risk_df), len(med_risk_df), len(high_risk_df)]
                fig_risk_pie = px.pie(
                    values=risk_values, names=risk_labels,
                    title="Customer Distribution by Churn Risk Tier",
                    hole=0.45,
                    color=risk_labels,
                    color_discrete_map=RISK_TIER_COLORS,
                    template="plotly_dark"
                )
                fig_risk_pie.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(color="#e8eefc", size=12)
                )
                style_plotly_fig(fig_risk_pie, height=350, show_legend=False)
                st.plotly_chart(fig_risk_pie, use_container_width=True, theme=None)

            cp_g_r2_c1, cp_g_r2_c2 = st.columns(2)
            with cp_g_r2_c1:
                churn_rfm = df_cp.groupby('Simplified_RFM_Segment')['Churn_Probability'].mean().reset_index()
                fig_churn_rfm = px.bar(
                    churn_rfm, x='Simplified_RFM_Segment', y='Churn_Probability',
                    color='Simplified_RFM_Segment',
                    title="Average Churn Probability by RFM Segment",
                    labels={'Churn_Probability': 'Avg Churn Risk'},
                    template="plotly_dark",
                    color_discrete_map=SEGMENT_COLORS
                )
                style_plotly_fig(fig_churn_rfm, height=350, show_legend=False, legend_title="RFM Segment")
                st.plotly_chart(fig_churn_rfm, use_container_width=True, theme=None)

            with cp_g_r2_c2:
                seg_tag = 'Ensemble_Segment' if 'Ensemble_Segment' in df_cp.columns else ('Advanced_Segment' if 'Advanced_Segment' in df_cp.columns else 'Simplified_RFM_Segment')
                churn_adv_plot = df_cp.groupby(seg_tag)['Churn_Probability'].mean().reset_index()
                fig_adv_churn_bar = px.bar(
                    churn_adv_plot, x=seg_tag, y='Churn_Probability',
                    color=seg_tag,
                    title=f"Average Churn Probability by {seg_tag}",
                    labels={'Churn_Probability': 'Avg Churn Risk'},
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                style_plotly_fig(fig_adv_churn_bar, height=350, show_legend=False, legend_title=seg_tag)
                st.plotly_chart(fig_adv_churn_bar, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Churn Risk Geography")
            cp_map, cp_level = build_geo_map(df_cp, value_col='Churn_Probability', agg='mean', title="Average Churn Risk by Geography")
            if cp_map is not None:
                st.plotly_chart(cp_map, use_container_width=True, theme=None)
            else:
                st.info("Add Country or US State codes to render the churn map.")

            # Optional Deep Learning Training Curves
            if 'dl_history' in st.session_state:
                st.markdown("##### 📈 Deep Learning Training History")
                dl_curve_c1, dl_curve_c2 = st.columns(2)
                hist_df = pd.DataFrame(st.session_state['dl_history'])
                with dl_curve_c1:
                    fig_loss = px.line(
                        hist_df, y=['loss', 'val_loss'], title="Training & Validation Loss",
                        template="plotly_dark",
                        color_discrete_map={'loss': '#2563eb', 'val_loss': '#ef4444'}
                    )
                    style_plotly_fig(fig_loss, height=300, show_legend=False, legend_title="Metric", legend_orientation="h")
                    st.plotly_chart(fig_loss, use_container_width=True, theme=None)
                with dl_curve_c2:
                    fig_auc = px.line(
                        hist_df, y=['auc', 'val_auc'], title="Training & Validation AUC",
                        template="plotly_dark",
                        color_discrete_map={'auc': '#10b981', 'val_auc': '#8b5cf6'}
                    )
                    style_plotly_fig(fig_auc, height=300, show_legend=False, legend_title="Metric", legend_orientation="h")
                    st.plotly_chart(fig_auc, use_container_width=True, theme=None)

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Churn Risk Matrices & High-Risk Customer Queue</div>', unsafe_allow_html=True)
            st.markdown("##### 📋 Segment-Level Risk Exposure Matrix")
            risk_matrix = df_cp.groupby('Simplified_RFM_Segment').agg(
                Total_Customers=('CustomerID', 'count'),
                High_Risk_Customers=('Churn_Probability', lambda x: (x >= high_risk_thresh).sum()),
                Avg_Churn_Probability=('Churn_Probability', 'mean'),
                Total_Spend_at_Risk=('Monetary', lambda x: df_cp.loc[x.index][df_cp.loc[x.index, 'Churn_Probability'] >= high_risk_thresh]['Monetary'].sum())
            ).round(2).reset_index()
            risk_matrix['High_Risk_%'] = ((risk_matrix['High_Risk_Customers'] / risk_matrix['Total_Customers'].replace(0, 1)) * 100).round(1)
            st.dataframe(risk_matrix, use_container_width=True)

            st.markdown("##### 🚨 Top Customers Ranked by Churn Risk")
            ranked_cols = ['CustomerID', 'Churn_Probability', 'Risk_Level', 'Simplified_RFM_Segment', 'Monetary', 'Recency', 'Frequency']
            if 'Age' in df_cp.columns: ranked_cols.append('Age')
            for geo_col in ['State', 'Region', 'Country', 'City']:
                if geo_col in df_cp.columns:
                    ranked_cols.append(geo_col)
            st.dataframe(df_cp[ranked_cols].sort_values('Churn_Probability', ascending=False).head(100), use_container_width=True)

    # =========================================================================
    # TAB 6: RETENTION STRATEGIES
    # =========================================================================
    if active_view == "🎯 Retention Strategies":
        st.header("🎯 AI-Driven Retention Playbooks & Action Engine")
        st.markdown("Prescriptive intervention strategies and campaign actions tailored to customer RFM segments and churn risk tiers.")

        # --- DYNAMIC SLICERS FOR TAB 6 ---
        st.markdown("""
        <div class="slicer-card">
            <div class="slicer-header">
                <span class="slicer-title">🎛️ Retention Campaign Dynamic Slicers</span>
                <span class="slicer-badge">Target Audience Filtering</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 1: Basic Filters
        strat_s1, strat_s2, strat_s3, strat_s4 = st.columns(4)
        with strat_s1:
            strat_priority_filter = st.multiselect(
                "Campaign Priority Scope:",
                options=['High Risk (>=70% Churn)', 'Medium Risk (40-70% Churn)', 'Low Risk (<40% Churn)'],
                default=['High Risk (>=70% Churn)', 'Medium Risk (40-70% Churn)'],
                key="strat_priority_filter"
            )
        with strat_s2:
            all_strat_segs = sorted(customer_df['Simplified_RFM_Segment'].dropna().unique()) if 'Simplified_RFM_Segment' in customer_df.columns else []
            strat_sel_segs = st.multiselect("Target Segments:", options=all_strat_segs, default=all_strat_segs, key="strat_sel_segs")
        with strat_s3:
            min_m_strat, max_m_strat = safe_range(customer_df['Monetary'])
            strat_min_spend = st.slider("Min Customer Lifetime Spend ($):", min_value=float(min_m_strat), max_value=float(max_m_strat), value=float(min_m_strat), key="strat_min_spend")
        with strat_s4:
            all_strat_states = sorted(customer_df['State'].dropna().astype(str).unique()) if 'State' in customer_df.columns else []
            strat_sel_state = st.multiselect("Target States:", options=all_strat_states, default=all_strat_states, key="strat_sel_state")

        # Row 2: Geographic Filters
        strat_s5, strat_s6, strat_s7 = st.columns(3)
        with strat_s5:
            all_strat_regions = sorted(customer_df['Region'].dropna().astype(str).unique()) if 'Region' in customer_df.columns else []
            strat_sel_region = st.multiselect("Target Regions:", options=all_strat_regions, default=all_strat_regions, key="strat_sel_region")
        with strat_s6:
            all_strat_countries = sorted(customer_df['Country'].dropna().astype(str).unique()) if 'Country' in customer_df.columns else []
            strat_sel_country = st.multiselect("Target Countries:", options=all_strat_countries, default=all_strat_countries, key="strat_sel_country")
        with strat_s7:
            all_strat_cities = sorted(customer_df['City'].dropna().astype(str).unique()) if 'City' in customer_df.columns else []
            strat_sel_city = st.multiselect("Target Cities:", options=all_strat_cities, default=all_strat_cities, key="strat_sel_city")

        # Tag for Strategy Slicing
        df_strat = customer_df.copy()
        df_strat['Strat_Tier'] = 'Low Risk (<40% Churn)'
        df_strat.loc[df_strat['Churn_Probability'] >= 0.40, 'Strat_Tier'] = 'Medium Risk (40-70% Churn)'
        df_strat.loc[df_strat['Churn_Probability'] >= 0.70, 'Strat_Tier'] = 'High Risk (>=70% Churn)'

        # Apply Slicers
        if strat_priority_filter:
            df_strat = df_strat[df_strat['Strat_Tier'].isin(strat_priority_filter)]
        if strat_sel_segs:
            df_strat = df_strat[df_strat['Simplified_RFM_Segment'].isin(strat_sel_segs)]
        if 'State' in df_strat.columns and strat_sel_state:
            df_strat = df_strat[df_strat['State'].astype(str).isin(strat_sel_state)]
        if 'Region' in df_strat.columns and strat_sel_region:
            df_strat = df_strat[df_strat['Region'].astype(str).isin(strat_sel_region)]
        if 'Country' in df_strat.columns and strat_sel_country:
            df_strat = df_strat[df_strat['Country'].astype(str).isin(strat_sel_country)]
        if 'City' in df_strat.columns and strat_sel_city:
            df_strat = df_strat[df_strat['City'].astype(str).isin(strat_sel_city)]
        if 'Monetary' in df_strat.columns:
            df_strat = df_strat[df_strat['Monetary'] >= strat_min_spend]

        st.caption(f"🔍 **Active Slicer Scope:** Targeting **{len(df_strat):,}** of **{len(customer_df):,}** customers ({len(df_strat)/max(len(customer_df),1):.1%})")

        # --- TIER 1: KPI CARDS ---
        st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 1</span> Strategic Retention Targets</div>', unsafe_allow_html=True)
        high_risk_ret_df = df_strat[df_strat['Strat_Tier'] == 'High Risk (>=70% Churn)']
        total_rev_protected = df_strat['Monetary'].sum() if not df_strat.empty else 0

        strat_kpi1, strat_kpi2, strat_kpi3, strat_kpi4 = st.columns(4)
        with strat_kpi1:
            st.metric("Actionable Target Users", f"{len(df_strat):,} users", delta="Active in Filter")
        with strat_kpi2:
            st.metric("Revenue to Protect", f"${total_rev_protected:,.2f}")
        with strat_kpi3:
            vulnerable_seg = df_strat.groupby('Simplified_RFM_Segment')['Churn_Probability'].mean().idxmax() if not df_strat.empty else "N/A"
            st.metric("Top Vulnerable Segment", str(vulnerable_seg))
        with strat_kpi4:
            st.metric("Retention Tracks Active", "5 Automated Tracks")

        if df_strat.empty:
            st.warning("⚠️ No customer records match the active retention campaign filters. Please broaden your slicers above.")
        else:
            # --- TIER 2: VISUAL GRAPHS & STRATEGY ACTION CARDS (NEATLY ALIGNED) ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 2</span> Strategic Visualizations & Action Playbook Grid</div>', unsafe_allow_html=True)
            strat_g_c1, strat_g_c2 = st.columns(2)
            with strat_g_c1:
                rev_by_seg_risk = df_strat.groupby('Simplified_RFM_Segment')['Monetary'].sum().reset_index()
                rev_by_seg_risk.columns = ['Segment', 'Revenue at Risk ($)']
                fig_strat_rev = px.bar(
                    rev_by_seg_risk, x='Segment', y='Revenue at Risk ($)',
                    color='Segment',
                    title="Vulnerable Revenue Pool by Targeted Segment ($)",
                    template="plotly_dark",
                    color_discrete_map=SEGMENT_COLORS
                )
                style_plotly_fig(fig_strat_rev, height=340, show_legend=False, legend_title="RFM Segment")
                st.plotly_chart(fig_strat_rev, use_container_width=True, theme=None)

            with strat_g_c2:
                strat_alloc = pd.DataFrame({
                    'Intervention Tier': ['Immediate VIP Outreach', 'Automated Win-Back Offers', 'Value Reinforcement & Nurture', 'Loyalty Rewards'],
                    'Resource Allocation %': [40, 30, 20, 10]
                })
                fig_alloc = px.pie(
                    strat_alloc, names='Intervention Tier', values='Resource Allocation %',
                    title="Recommended Retention Budget & Effort Allocation",
                    hole=0.45,
                    template="plotly_dark",
                    color_discrete_sequence=['#ef4444', '#f59e0b', '#2563eb', '#10b981']
                )
                fig_alloc.update_traces(
                    textinfo="label+percent",
                    textposition="outside",
                    textfont=dict(color="#e8eefc", size=11)
                )
                style_plotly_fig(fig_alloc, height=340, show_legend=False)
                st.plotly_chart(fig_alloc, use_container_width=True, theme=None)

            st.markdown("##### 🗺️ Revenue-at-Risk Geography")
            strat_map, strat_level = build_geo_map(df_strat, value_col='Monetary', agg='sum', title="Targeted Revenue Pool by Geography")
            if strat_map is not None:
                st.plotly_chart(strat_map, use_container_width=True, theme=None)
            else:
                st.info("Add Country or US State codes to render the retention map.")

            # Aligned Strategy Playbook Grid
            st.markdown("##### 📌 Segment-Specific Strategic Playbooks")
            playbook_col1, playbook_col2 = st.columns(2)
            with playbook_col1:
                st.markdown("""
                <div class="action-card" style="border-left-color: #10b981;">
                    <h4>🏆 Champions (High Value, Low Churn)</h4>
                    <p><strong>Goal:</strong> Maximize lifetime advocacy and premium referrals.</p>
                    <p><strong>Action:</strong> VIP status invitations, early access to new feature releases, personalized concierge support, and exclusive partner benefits.</p>
                </div>
                <div class="action-card" style="border-left-color: #3b82f6;">
                    <h4>💎 Loyal Customers (Consistent Repeat Purchases)</h4>
                    <p><strong>Goal:</strong> Increase basket size and brand stickiness.</p>
                    <p><strong>Action:</strong> Personalized loyalty reward tiers, bundled upsell offers based on purchase history, and satisfaction check-in surveys.</p>
                </div>
                <div class="action-card" style="border-left-color: #8b5cf6;">
                    <h4>🌱 Potential Loyalists (Recent or Moderate Frequency)</h4>
                    <p><strong>Goal:</strong> Accelerate transition into higher spending frequency tiers.</p>
                    <p><strong>Action:</strong> Limited-time repeat purchase discounts, onboarding educational content, and gamified membership milestones.</p>
                </div>
                """, unsafe_allow_html=True)

            with playbook_col2:
                st.markdown("""
                <div class="action-card" style="border-left-color: #f59e0b;">
                    <h4>⚠️ Need Attention (Declining Recency / Medium Risk)</h4>
                    <p><strong>Goal:</strong> Re-engage before defection becomes permanent.</p>
                    <p><strong>Action:</strong> Dynamic "We Miss You" win-back email workflows, customized price promotions, and proactive customer success consultations.</p>
                </div>
                <div class="action-card" style="border-left-color: #ef4444;">
                    <h4>❄️ Hibernating (High Inactivity / High Churn Risk)</h4>
                    <p><strong>Goal:</strong> Last-resort reactivations or cost-efficient sunsetting.</p>
                    <p><strong>Action:</strong> High-incentive clearance discounts, direct SMS re-activation offers, and low-cost automated nurture drips.</p>
                </div>
                <div class="action-card" style="border-left-color: #64748b;">
                    <h4>🤖 Deep Learning & Latent Insights</h4>
                    <p><strong>Goal:</strong> Address non-linear behavioral anomaly triggers.</p>
                    <p><strong>Action:</strong> Segment customers with high autoencoder reconstruction errors for dedicated support audits to preempt unexpected churn.</p>
                </div>
                """, unsafe_allow_html=True)

            # --- TIER 3: TABULAR REPRESENTATION ---
            st.markdown('<div class="section-tier-header"><span class="tier-tag">Tier 3</span> Prescriptive Action Matrix & CRM Export Queue</div>', unsafe_allow_html=True)
            st.markdown("##### 📋 Retention Action Playbook Matrix")
            playbook_matrix = pd.DataFrame([
                {"Segment": "Champions", "Risk Tier": "Low", "Trigger": "R >= 4, F >= 4, M >= 4", "Action": "VIP Program, Concierge Access", "Channel": "Direct Call / Dedicated Email", "Priority": "P1 (Advocacy)"},
                {"Segment": "Loyal Customers", "Risk Tier": "Low-Medium", "Trigger": "R >= 3, F >= 3, M >= 3", "Action": "Loyalty Points Booster, Upselling", "Channel": "Email / App Notification", "Priority": "P2 (Expansion)"},
                {"Segment": "Potential Loyalists", "Risk Tier": "Medium", "Trigger": "R >= 3, F <= 2, M >= 3", "Action": "Next-Order Incentive, Education", "Channel": "Push Notification / Email", "Priority": "P2 (Growth)"},
                {"Segment": "Need Attention", "Risk Tier": "Medium-High", "Trigger": "R >= 3, F >= 3, M <= 2", "Action": "Customized Win-Back Discount", "Channel": "Targeted Email / SMS", "Priority": "P1 (Urgent)"},
                {"Segment": "Hibernating", "Risk Tier": "High", "Trigger": "R <= 2, F <= 2, M <= 2", "Action": "Reactivation Deep Offer or Sunset", "Channel": "Automated Drip Campaign", "Priority": "P3 (Efficiency)"}
            ])
            st.dataframe(playbook_matrix, use_container_width=True)

            st.markdown("##### 🚀 Exportable Action Target Customer Queue (Filtered by Slicers)")
            export_cols = ['CustomerID', 'Churn_Probability', 'Strat_Tier', 'Simplified_RFM_Segment', 'Monetary', 'Recency', 'Frequency']
            if 'Age' in df_strat.columns: export_cols.append('Age')
            if 'Gender' in df_strat.columns: export_cols.append('Gender')
            if 'State' in df_strat.columns: export_cols.append('State')
            if 'Region' in df_strat.columns: export_cols.append('Region')
            if 'Country' in df_strat.columns: export_cols.append('Country')
            if 'City' in df_strat.columns: export_cols.append('City')

            st.download_button(
                "⬇️ Export Filtered Target Customers to CSV (for CRM / Marketing)",
                data=df_strat[export_cols].sort_values('Churn_Probability', ascending=False).to_csv(index=False),
                file_name="targeted_retention_customers.csv",
                mime="text/csv",
                key="btn_export_retention_csv"
            )
            st.dataframe(df_strat[export_cols].sort_values('Churn_Probability', ascending=False).head(50), use_container_width=True)

else:
    # Show message when other tabs are selected but no data is loaded
    if active_view != "📤 Upload Data":
        st.warning("⚠️ Please upload a dataset first using the **Upload Data** tab to access this view.")
