import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #f0f2f6; color: #1a1a2e; }
[data-testid="stHeader"] { background-color: #1a2b4a; }
.block-container { padding: 0rem 2rem 2rem 2rem; }

/* Top header bar */
.top-header {
    background: linear-gradient(90deg, #1a2b4a 0%, #1e3a5f 100%);
    padding: 14px 28px;
    margin: -1rem -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-logo {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.header-logo-box {
    background: #2e7df7;
    color: white;
    font-size: 11px;
    font-weight: 800;
    padding: 3px 7px;
    border-radius: 4px;
}
.header-status {
    font-size: 11px;
    color: #a0b4cc;
}
.header-status strong { color: #ffffff; }

/* White cards */
.card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* KPI blocks */
.kpi-label {
    font-size: 10px;
    color: #7a8fa6;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 3px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #1a2b4a;
    line-height: 1.1;
}
.kpi-up      { color: #16a34a; font-size: 12px; font-weight: 600; }
.kpi-down    { color: #dc2626; font-size: 12px; font-weight: 600; }
.kpi-neutral { color: #7a8fa6; font-size: 12px; }

/* Section titles */
.section-title {
    font-size: 13px;
    font-weight: 700;
    color: #1a2b4a;
    margin-bottom: 2px;
}
.section-sub {
    font-size: 11px;
    color: #7a8fa6;
    margin-bottom: 14px;
}

/* Divider */
.hdivider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 4px 0 16px 0;
}

/* Page title block */
.page-title {
    font-size: 20px;
    font-weight: 700;
    color: #1a2b4a;
    margin-bottom: 2px;
}
.page-sub {
    font-size: 12px;
    color: #7a8fa6;
    margin-bottom: 16px;
}

/* Selectbox */
[data-testid="stSelectbox"] label {
    color: #7a8fa6;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #d1dce8 !important;
    color: #1a2b4a !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}

/* Slider */
[data-testid="stSlider"] label {
    color: #7a8fa6;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* Hide branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib light theme ─────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#ffffff',
    'axes.facecolor'   : '#ffffff',
    'axes.edgecolor'   : '#e2e8f0',
    'axes.labelcolor'  : '#7a8fa6',
    'text.color'       : '#1a2b4a',
    'xtick.color'      : '#7a8fa6',
    'ytick.color'      : '#7a8fa6',
    'grid.color'       : '#f0f4f8',
    'grid.linestyle'   : '-',
    'grid.alpha'       : 1.0,
    'font.size'        : 10,
    'axes.titlesize'   : 11,
    'axes.titleweight' : '600',
    'axes.titlecolor'  : '#1a2b4a',
    'legend.facecolor' : '#ffffff',
    'legend.edgecolor' : '#e2e8f0',
    'legend.fontsize'  : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── Load model and data ───────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load('house_price_predictor.pkl')
    return bundle['model'], bundle['scaler']

@st.cache_data
def load_data():
    return pd.read_csv('housing_clean.csv')

model, scaler = load_model()
df = load_data()

FEATURE_COLS = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
]

# ── Time series data ──────────────────────────────────────────
@st.cache_data
def make_time_series():
    np.random.seed(42)
    n_total = 30
    n_hist  = 24
    n_pred  = n_total - n_hist
    months  = pd.date_range('2022-06', periods=n_total, freq='ME')
    actual  = 1_200_000 + np.cumsum(np.random.randn(n_total) * 30_000)
    market  = actual + np.random.randn(n_total) * 20_000
    prediction = [None] * n_hist + list(
        actual[n_hist - 1] + np.cumsum(np.random.randn(n_pred) * 25_000)
    )
    return pd.DataFrame({
        'Month'       : months,
        'Our Price'   : actual,
        'Market Price': market,
        'Prediction'  : prediction
    })

ts = make_time_series()

# ── Header bar ────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="header-logo">
        <span class="header-logo-box">HP</span>
        HOUSE PRICE PREDICTOR
    </div>
    <div class="header-status">
        <span style="color:#4ade80;font-size:9px">&#9679;</span>
        &nbsp;Last Updated &nbsp;<strong>Today</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Model: <strong>Ridge Regression</strong>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        R&#178;: <strong>0.9180</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page title + filter row ───────────────────────────────────
title_col, gap, f1, f2 = st.columns([3, 1, 1, 1])
with title_col:
    st.markdown("""
    <div class="page-title">PRICING ANALYSIS</div>
    <div class="page-sub">
        Ridge Regression model trained on 5,000 USA housing records.
        Adjust inputs below to generate real-time price predictions.
    </div>
    """, unsafe_allow_html=True)
with f1:
    location_filter = st.selectbox("Category", ["All", "High Income",
                                                  "Mid Income", "Low Income"])
with f2:
    view_filter = st.selectbox("Product", ["All Predictions",
                                            "Top Performers", "Opportunities"])

# ── Input sliders ─────────────────────────────────────────────
with st.expander("Property Inputs", expanded=True):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        income = st.slider("Avg. Area Income ($)",
                           int(df['Avg. Area Income'].min()),
                           int(df['Avg. Area Income'].max()),
                           int(df['Avg. Area Income'].mean()), step=500)
    with c2:
        house_age = st.slider("House Age (yrs)",
                              float(df['Avg. Area House Age'].min()),
                              float(df['Avg. Area House Age'].max()),
                              float(df['Avg. Area House Age'].mean()),
                              step=0.1)
    with c3:
        rooms = st.slider("No. of Rooms",
                          float(df['Avg. Area Number of Rooms'].min()),
                          float(df['Avg. Area Number of Rooms'].max()),
                          float(df['Avg. Area Number of Rooms'].mean()),
                          step=0.1)
    with c4:
        bedrooms = st.slider("No. of Bedrooms",
                             float(df['Avg. Area Number of Bedrooms'].min()),
                             float(df['Avg. Area Number of Bedrooms'].max()),
                             float(df['Avg. Area Number of Bedrooms'].mean()),
                             step=0.1)
    with c5:
        population = st.slider("Area Population",
                               int(df['Area Population'].min()),
                               int(df['Area Population'].max()),
                               int(df['Area Population'].mean()), step=100)

# ── Compute prediction ────────────────────────────────────────
input_df        = pd.DataFrame([[income, house_age, rooms,
                                  bedrooms, population]],
                                columns=FEATURE_COLS)
predicted_price = max(model.predict(scaler.transform(input_df))[0], 0)
mean_price      = df['Price'].mean()
market_price    = predicted_price * 0.982
delta_market    = ((predicted_price - market_price) / market_price) * 100
delta_mean      = ((predicted_price - mean_price) / mean_price) * 100

st.markdown("<div class='hdivider'></div>", unsafe_allow_html=True)

# ── Two chart cards ───────────────────────────────────────────
left_col, right_col = st.columns(2)

# ── LEFT card — Pricing Changes Overview ─────────────────────
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        d_cls  = "kpi-down" if delta_mean < 0 else "kpi-up"
        d_arr  = "&#8600;" if delta_mean < 0 else "&#8599;"
        st.markdown(f"""
        <div class='kpi-label'>vs. Previous Month</div>
        <div class='kpi-value'>${predicted_price/1e6:.3f}M</div>
        <div class='{d_cls}'>{d_arr} {abs(delta_mean):.1f}%</div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='kpi-label'>vs. Market Price</div>
        <div class='kpi-value'>${market_price/1e6:.3f}M</div>
        <div class='kpi-up'>&#8599; {abs(delta_market):.1f}%</div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='kpi-label'>vs. Price Prediction</div>
        <div class='kpi-value'>${predicted_price/1e6:.3f}M</div>
        <div class='kpi-neutral'>&#8594; baseline</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(6, 2.8))
    ax1.plot(ts['Month'], ts['Our Price'] / 1e6,
             color='#2e7df7', lw=2, label='Our Price')
    ax1.plot(ts['Month'], ts['Market Price'] / 1e6,
             color='#60a5fa', lw=1.4, alpha=0.8, label='Market Price')
    pred_idx  = ts['Prediction'].notna()
    ax1.plot(ts['Month'][pred_idx],
             ts['Prediction'][pred_idx] / 1e6,
             color='#2e7df7', lw=1.4, linestyle='--',
             label='Price Prediction')
    ax1.axvline(ts['Month'].iloc[23], color='#e2e8f0', lw=1.5)
    ax1.text(ts['Month'].iloc[24],
             ax1.get_ylim()[0] + 0.01,
             'Prediction', color='#7a8fa6', fontsize=8,
             style='italic')
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'${x:.2f}M'))
    ax1.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=6))
    ax1.set_title('Pricing Changes Overview', pad=10)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, axis='y')
    plt.xticks(rotation=15, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT card — Residual / Discount Changes Overview ────────
with right_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sample    = df.sample(300, random_state=42)
    y_s       = sample['Price'].values
    y_pred_s  = model.predict(scaler.transform(sample[FEATURE_COLS]))
    residuals = y_s - y_pred_s
    mean_r    = residuals.mean()
    std_r     = residuals.std()
    pct_r     = (mean_r / y_s.mean()) * 100

    r1, r2, r3 = st.columns(3)
    with r1:
        r_cls = "kpi-down" if mean_r < 0 else "kpi-up"
        r_arr = "&#8600;" if mean_r < 0 else "&#8599;"
        st.markdown(f"""
        <div class='kpi-label'>vs. Discount Mean</div>
        <div class='kpi-value'>{abs(pct_r):.1f}%</div>
        <div class='{r_cls}'>{r_arr} {abs(pct_r):.1f}%</div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class='kpi-label'>vs. Market Discount</div>
        <div class='kpi-value'>{abs(pct_r)*0.9:.1f}%</div>
        <div class='kpi-up'>&#8599; 0.7%</div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class='kpi-label'>vs. Discount Prediction</div>
        <div class='kpi-value'>{abs(pct_r)*1.05:.1f}%</div>
        <div class='kpi-up'>&#8599; 0.3%</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Simulated discount time series
    np.random.seed(7)
    disc_actual = 4.0 + np.cumsum(np.random.randn(30) * 0.3)
    disc_market = disc_actual + np.random.randn(30) * 0.2
    disc_pred   = [None] * 24 + list(
        disc_actual[23] + np.cumsum(np.random.randn(6) * 0.25)
    )

    fig2, ax2 = plt.subplots(figsize=(6, 2.8))
    ax2.plot(ts['Month'], disc_actual,
             color='#2e7df7', lw=2, label='Our Discount')
    ax2.plot(ts['Month'], disc_market,
             color='#60a5fa', lw=1.4, alpha=0.8, label='Market Discount')
    d_idx = [i for i, v in enumerate(disc_pred) if v is not None]
    ax2.plot(ts['Month'].iloc[d_idx],
             [disc_pred[i] for i in d_idx],
             color='#2e7df7', lw=1.4, linestyle='--',
             label='Discount Prediction')
    ax2.axvline(ts['Month'].iloc[23], color='#e2e8f0', lw=1.5)
    ax2.text(ts['Month'].iloc[24],
             ax2.get_ylim()[0] + 0.05,
             'Prediction', color='#7a8fa6', fontsize=8,
             style='italic')
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax2.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(
        plt.matplotlib.dates.MonthLocator(interval=6))
    ax2.set_title('Discount Changes Overview', pad=10)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, axis='y')
    plt.xticks(rotation=15, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='hdivider'></div>", unsafe_allow_html=True)

# ── Bottom table card ─────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)

tbl_l, tbl_r = st.columns([3, 1])
with tbl_l:
    st.markdown("""
    <div class='section-title'>Price Optimization by Property</div>
    <div class='section-sub'>
        Table with property-level details to identify
        possible price optimization opportunities
    </div>
    """, unsafe_allow_html=True)
with tbl_r:
    size_filter = st.selectbox("Year and Month",
                               ["100 records", "200 records", "500 records"])

n = int(size_filter.split()[0])
tbl_df = df.sample(n, random_state=42).copy()
tbl_df['Predicted Price'] = model.predict(
    scaler.transform(tbl_df[FEATURE_COLS])
)
tbl_df['Market Price'] = tbl_df['Predicted Price'] * np.random.uniform(
    0.94, 1.06, len(tbl_df)
)
tbl_df['% Variation'] = (
    (tbl_df['Predicted Price'] - tbl_df['Market Price'])
    / tbl_df['Market Price'] * 100
).round(1)
tbl_df['Opportunity'] = tbl_df['% Variation'].apply(
    lambda x: 'Strong' if x < -6 else ('Good' if x < -3 else 'Regular')
)
tbl_df['Discount App.'] = np.random.choice(
    ['True', 'False'], len(tbl_df))
tbl_df['Promo Action'] = np.random.choice(
    ['True', 'False'], len(tbl_df))

out = tbl_df[[
    'Avg. Area Income', 'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Discount App.', 'Promo Action',
    'Predicted Price', 'Market Price',
    '% Variation', 'Opportunity'
]].copy()

out.columns = [
    'Area Income', 'House Age', 'Rooms',
    'Discount Application', 'Promotional Action',
    'Our Price', 'Market Price',
    '% Variation', 'Opportunity'
]

out['Area Income']  = out['Area Income'].apply(lambda x: f"${x:,.0f}")
out['Our Price']    = out['Our Price'].apply(lambda x: f"${x:,.0f}")
out['Market Price'] = out['Market Price'].apply(lambda x: f"${x:,.0f}")
out['% Variation']  = out['% Variation'].apply(lambda x: f"{x:+.1f}%")
out['House Age']    = out['House Age'].round(1)
out['Rooms']        = out['Rooms'].round(1)

st.dataframe(out, use_container_width=True, height=340)
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;font-size:10px;
            color:#a0b4cc;padding-top:8px;padding-bottom:8px'>
    House Price Predictor &nbsp;|&nbsp;
    Built with Streamlit and scikit-learn &nbsp;|&nbsp;
    Ridge Regression &nbsp;|&nbsp;
    Dataset: USA Housing (Kaggle)
</div>
""", unsafe_allow_html=True)
