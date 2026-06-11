"""
Tesla Stock Price Prediction — Streamlit Dashboard
Run:  streamlit run app.py
"""

import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Dark background */
  .stApp { background-color: #0d1117; color: #e6edf3; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background-color: #161b22;
      border-right: 1px solid #21262d;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
      background: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
      color: #8b949e !important;
      font-size: 11px !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-family: 'IBM Plex Mono', monospace !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 1.6rem !important;
      color: #58a6ff !important;
  }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.78rem !important;
  }

  /* Section headers */
  .section-header {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #58a6ff;
      border-bottom: 1px solid #21262d;
      padding-bottom: 8px;
      margin-bottom: 20px;
      margin-top: 8px;
  }

  /* Hero banner */
  .hero {
      background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
      border: 1px solid #21262d;
      border-left: 3px solid #58a6ff;
      border-radius: 8px;
      padding: 28px 32px;
      margin-bottom: 28px;
  }
  .hero h1 {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.6rem;
      font-weight: 600;
      color: #e6edf3;
      margin: 0 0 6px 0;
      letter-spacing: -0.02em;
  }
  .hero p {
      color: #8b949e;
      font-size: 0.9rem;
      margin: 0;
      font-weight: 300;
  }
  .hero .accent { color: #58a6ff; }

  /* Status pills */
  .pill {
      display: inline-block;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      padding: 3px 10px;
      border-radius: 20px;
      letter-spacing: 0.05em;
  }
  .pill-blue  { background: #1f3a5c; color: #58a6ff; border: 1px solid #264a7a; }
  .pill-green { background: #1a3a2a; color: #3fb950; border: 1px solid #238636; }
  .pill-red   { background: #3a1a1a; color: #f85149; border: 1px solid #6e2222; }

  /* Win badge */
  .win-badge {
      background: #1a3a2a;
      border: 1px solid #238636;
      color: #3fb950;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      padding: 2px 8px;
      border-radius: 4px;
      margin-left: 8px;
  }

  /* Scrollable table */
  .results-table { overflow-x: auto; }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Tabs */
  button[data-baseweb="tab"] {
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 0.75rem !important;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }

  /* Upload zone */
  [data-testid="stFileUploadDropzone"] {
      background: #161b22 !important;
      border: 1px dashed #30363d !important;
      border-radius: 8px !important;
  }

  /* Divider */
  hr { border-color: #21262d; }

  /* Sidebar nav labels */
  .sidebar-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      color: #8b949e;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACK   = 60
HORIZONS   = {1: "Next Day", 5: "Next 5 Days", 10: "Next 10 Days"}
MODEL_DIR  = "."   # looks for best_lstm.keras / best_simplernn.keras here

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#8b949e", size=11),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1,
                font=dict(size=10)),
)

@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd       = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_features(df):
    df = df.copy()
    df["MA30"]        = df["Adj Close"].rolling(30).mean()
    df["MA90"]        = df["Adj Close"].rolling(90).mean()
    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Volatility"]  = df["Daily_Return"].rolling(21).std() * np.sqrt(252)
    df["RSI"]         = compute_rsi(df["Adj Close"])
    df["MACD"], df["MACD_Signal"] = compute_macd(df["Adj Close"])
    df["BB_Upper"]    = df["Adj Close"].rolling(20).mean() + 2 * df["Adj Close"].rolling(20).std()
    df["BB_Lower"]    = df["Adj Close"].rolling(20).mean() - 2 * df["Adj Close"].rolling(20).std()
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df

def create_sequences(data, lookback=60, horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback + horizon - 1])
    return np.array(X), np.array(y)

def build_and_train_model(model_type, X_train, y_train, lookback,
                           units_1=64, units_2=32, dropout=0.2, lr=1e-3, epochs=40):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    layer = SimpleRNN if model_type == "SimpleRNN" else LSTM
    model = Sequential([
        layer(units_1, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout),
        layer(units_2, return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ], name=f"{model_type}_Model")
    model.compile(optimizer=Adam(lr), loss="mse")
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs,
              batch_size=32,
              callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
              verbose=0)
    return model

@st.cache_resource(show_spinner=False)
def get_models_and_results(data_hash, scaled_bytes, n_rows):
    """Train (or load) models for all horizons. Cached by data fingerprint."""
    import tensorflow as tf

    scaled = np.frombuffer(scaled_bytes, dtype=np.float32).reshape(-1, 1)
    results = {}

    for h, label in HORIZONS.items():
        X, y   = create_sequences(scaled, LOOKBACK, h)
        split  = int(len(X) * 0.80)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:],  y[split:]

        horizon_res = {}
        for mtype in ["SimpleRNN", "LSTM"]:
            # Try loading saved model first
            model_path = os.path.join(MODEL_DIR, f"best_{mtype.lower()}.keras")
            try:
                model = tf.keras.models.load_model(model_path)
                source = "loaded"
            except Exception:
                model  = build_and_train_model(mtype, X_tr, y_tr, LOOKBACK)
                source = "trained"

            y_pred_s = model.predict(X_te, verbose=0)
            horizon_res[mtype] = {
                "y_pred_scaled": y_pred_s,
                "y_test_scaled": y_te,
                "source": source,
            }

        results[h] = {"X_tr": X_tr, "y_tr": y_tr,
                      "X_te": X_te, "y_te": y_te,
                      "label": label, "models": horizon_res}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family: IBM Plex Mono, monospace; font-size: 1.1rem;
                font-weight: 600; color: #58a6ff; margin-bottom: 4px;'>
        TSLA<span style='color:#8b949e;'>/PRED</span>
    </div>
    <div style='font-size: 0.72rem; color: #8b949e; margin-bottom: 24px;
                font-family: IBM Plex Mono, monospace; letter-spacing: 0.05em;'>
        Deep Learning Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload TSLA CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "page",
        ["Overview", "EDA & Indicators", "Predictions", "Model Comparison"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    selected_horizon = st.selectbox(
        "Forecast Horizon",
        options=list(HORIZONS.keys()),
        format_func=lambda h: HORIZONS[h],
    )
    show_volume = st.toggle("Show Volume Overlay", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.65rem; color: #484f58; font-family: IBM Plex Mono, monospace;
                line-height: 1.6;'>
        Models: SimpleRNN + LSTM<br>
        Lookback: 60 trading days<br>
        Horizons: 1 / 5 / 10 days<br>
        Scaler: MinMaxScaler [0,1]
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# GATE: require file upload
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is None:
    st.markdown("""
    <div class="hero">
        <h1>Tesla Stock Price<br><span class="accent">Prediction Dashboard</span></h1>
        <p>SimpleRNN vs LSTM · 1-day / 5-day / 10-day forecasts · Full EDA suite</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:20px;'>
            <div style='font-size:1.4rem; margin-bottom:8px;'>📊</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.75rem;
                        color:#58a6ff; margin-bottom:6px;'>EDA SUITE</div>
            <div style='font-size:0.82rem; color:#8b949e;'>
                Price history, volume, returns distribution, volatility, and correlation heatmap.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:20px;'>
            <div style='font-size:1.4rem; margin-bottom:8px;'>🔮</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.75rem;
                        color:#58a6ff; margin-bottom:6px;'>PREDICTIONS</div>
            <div style='font-size:0.82rem; color:#8b949e;'>
                Actual vs predicted overlays for both models across all three forecast horizons.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:20px;'>
            <div style='font-size:1.4rem; margin-bottom:8px;'>⚖️</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.75rem;
                        color:#58a6ff; margin-bottom:6px;'>MODEL COMPARISON</div>
            <div style='font-size:0.82rem; color:#8b949e;'>
                RMSE / MAE / MAPE comparison tables and charts. Grid search results included.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("⬆️  Upload **TSLA.csv** in the sidebar to begin.", icon="📁")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading data…"):
    df_raw  = load_and_clean(uploaded)
    df      = add_features(df_raw)

# Scale for models
scaler     = MinMaxScaler(feature_range=(0, 1))
price_data = df[["Adj Close"]].dropna()
scaled     = scaler.fit_transform(price_data).astype(np.float32)

# Fingerprint for cache key
data_hash  = str(len(df)) + str(df["Adj Close"].iloc[-1])
scaled_bytes = scaled.tobytes()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("""
    <div class="hero">
        <h1>Tesla Stock Price<br><span class="accent">Prediction Dashboard</span></h1>
        <p>SimpleRNN vs LSTM · 1-day / 5-day / 10-day forecasts · Full EDA suite</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    latest     = df["Adj Close"].iloc[-1]
    prev       = df["Adj Close"].iloc[-2]
    daily_ret  = (latest - prev) / prev * 100
    ytd_start  = df["Adj Close"][df.index.year == df.index[-1].year].iloc[0]
    ytd_ret    = (latest - ytd_start) / ytd_start * 100
    vol_annual = df["Volatility"].iloc[-1] * 100
    avg_vol    = df["Volume"].tail(30).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",  f"${latest:,.2f}", f"{daily_ret:+.2f}% today")
    c2.metric("YTD Return",    f"{ytd_ret:+.1f}%", f"from ${ytd_start:,.2f}")
    c3.metric("21d Volatility",f"{vol_annual:.1f}%", "annualised")
    c4.metric("Avg Volume (30d)", f"{avg_vol/1e6:.1f}M", "shares/day")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Price History</div>', unsafe_allow_html=True)

    # ── Price chart ───────────────────────────────────────────────────────────
    fig = make_subplots(rows=2 if show_volume else 1, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.75, 0.25] if show_volume else [1.0],
                        vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Adj Close"], name="Adj Close",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.05)"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA30"], name="MA 30",
        line=dict(color="#f85149", width=1, dash="dot")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA90"], name="MA 90",
        line=dict(color="#3fb950", width=1, dash="dot")
    ), row=1, col=1)

    if show_volume:
        colors = ["#3fb950" if df["Close"].iloc[i] >= df["Close"].iloc[i-1]
                  else "#f85149" for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, marker_line_width=0, opacity=0.7
        ), row=2, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=500 if show_volume else 380,
                      title=dict(text="TSLA — Full Adjusted Close History",
                                 font=dict(size=13, color="#e6edf3")))
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                     title_font=dict(size=10))
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         title_font=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Dataset info ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    ci, cii, ciii = st.columns(3)
    with ci:
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px;'>
            <div class='sidebar-label'>Date Range</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.85rem; color:#e6edf3;'>
                {df.index.min().strftime('%d %b %Y')}<br>→ {df.index.max().strftime('%d %b %Y')}
            </div>
        </div>""", unsafe_allow_html=True)
    with cii:
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px;'>
            <div class='sidebar-label'>Trading Days</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.85rem; color:#e6edf3;'>
                {len(df):,} rows<br>{df.shape[1]} columns
            </div>
        </div>""", unsafe_allow_html=True)
    with ciii:
        missing = df[["Open","High","Low","Close","Adj Close","Volume"]].isnull().sum().sum()
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px;'>
            <div class='sidebar-label'>Data Quality</div>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.85rem; color:#e6edf3;'>
                Missing values: {missing}<br>
                <span style='color:#3fb950;'>{"✓ Clean" if missing == 0 else "⚠ Filled"}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA & INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "EDA & Indicators":
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "RETURNS & VOLATILITY", "CORRELATION", "TECHNICAL INDICATORS", "DISTRIBUTION"
    ])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Daily_Return"] * 100,
                mode="lines", name="Daily Return",
                line=dict(color="#58a6ff", width=0.8),
                fill="tozeroy", fillcolor="rgba(88,166,255,0.07)"
            ))
            fig.add_hline(y=0, line_color="#30363d", line_width=1)
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              title=dict(text="Daily Returns (%)",
                                         font=dict(size=12, color="#e6edf3")))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Volatility"] * 100,
                mode="lines", name="21d Volatility",
                line=dict(color="#f85149", width=1.2),
                fill="tozeroy", fillcolor="rgba(248,81,73,0.07)"
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              title=dict(text="21-Day Annualised Volatility (%)",
                                         font=dict(size=12, color="#e6edf3")))
            st.plotly_chart(fig, use_container_width=True)

        # Stats table
        ret = df["Daily_Return"].dropna() * 100
        st.markdown('<div class="section-header">Return Statistics</div>',
                    unsafe_allow_html=True)
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Mean Return",  f"{ret.mean():.3f}%")
        sc2.metric("Std Dev",      f"{ret.std():.3f}%")
        sc3.metric("Skewness",     f"{ret.skew():.3f}")
        sc4.metric("Kurtosis",     f"{ret.kurt():.3f}")
        sc5.metric("Max Drawdown", f"{((df['Adj Close'] / df['Adj Close'].cummax()) - 1).min() * 100:.1f}%")

    with tab2:
        corr_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=[[0, "#f85149"], [0.5, "#0d1117"], [1, "#58a6ff"]],
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=420,
                          title=dict(text="Feature Correlation Matrix",
                                     font=dict(size=12, color="#e6edf3")),
                          coloraxis_colorbar=dict(
                              tickfont=dict(family="IBM Plex Mono, monospace",
                                            color="#8b949e", size=9)))
        fig.update_traces(textfont=dict(family="IBM Plex Mono, monospace",
                                         color="#e6edf3", size=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Last N days slider
        n_days = st.slider("Days to display", 90, min(730, len(df)), 365, step=30)
        recent = df.tail(n_days)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.4, 0.2, 0.2, 0.2],
                            vertical_spacing=0.04,
                            subplot_titles=["Price + Bollinger Bands",
                                            "RSI (14)", "MACD", "OBV"])

        # Price + BB
        fig.add_trace(go.Scatter(x=recent.index, y=recent["Adj Close"],
                                  name="Adj Close", line=dict(color="#58a6ff", width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_Upper"],
                                  name="BB Upper", line=dict(color="#8b949e", width=0.8, dash="dot")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["BB_Lower"],
                                  name="BB Lower", fill="tonexty",
                                  fillcolor="rgba(139,148,158,0.07)",
                                  line=dict(color="#8b949e", width=0.8, dash="dot")),
                      row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=recent.index, y=recent["RSI"],
                                  name="RSI", line=dict(color="#f0883e", width=1.2)),
                      row=2, col=1)
        fig.add_hline(y=70, line_color="#f85149", line_width=0.8, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_color="#3fb950", line_width=0.8, line_dash="dot", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD"],
                                  name="MACD", line=dict(color="#a371f7", width=1.2)),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=recent.index, y=recent["MACD_Signal"],
                                  name="Signal", line=dict(color="#f85149", width=1, dash="dot")),
                      row=3, col=1)

        # OBV
        fig.add_trace(go.Scatter(x=recent.index, y=recent["OBV"] / 1e6,
                                  name="OBV (M)", line=dict(color="#3fb950", width=1)),
                      row=4, col=1)

        fig.update_layout(**PLOTLY_LAYOUT, height=680,
                          showlegend=False,
                          title=dict(text=f"Technical Indicators — Last {n_days} Days",
                                     font=dict(size=12, color="#e6edf3")))
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig.update_yaxes(gridcolor="#21262d", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        ret_clean = df["Daily_Return"].dropna() * 100
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Daily Return Distribution", "Q-Q Plot (vs Normal)"])

        fig.add_trace(go.Histogram(
            x=ret_clean, nbinsx=120, name="Returns",
            marker_color="#58a6ff", marker_line_width=0, opacity=0.8
        ), row=1, col=1)
        fig.add_vline(x=0, line_color="#f85149", line_width=1.5, row=1, col=1)

        # Q-Q
        sorted_ret = np.sort(ret_clean.values)
        n = len(sorted_ret)
        theoretical = np.random.normal(ret_clean.mean(), ret_clean.std(), n)
        theoretical.sort()
        fig.add_trace(go.Scatter(
            x=theoretical, y=sorted_ret, mode="markers",
            marker=dict(color="#3fb950", size=2, opacity=0.5), name="Q-Q"
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[theoretical[0], theoretical[-1]],
            y=[theoretical[0], theoretical[-1]],
            mode="lines", line=dict(color="#f85149", width=1.5, dash="dot"), name="Normal"
        ), row=1, col=2)

        fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                          title=dict(text="Return Distribution Analysis",
                                     font=dict(size=12, color="#e6edf3")))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Predictions":
    st.markdown('<div class="section-header">Model Predictions</div>',
                unsafe_allow_html=True)

    with st.spinner("Running models… this may take 1–2 minutes on first load."):
        model_results = get_models_and_results(data_hash, scaled_bytes, len(df))

    h    = selected_horizon
    data = model_results[h]

    # Source badges
    for mtype in ["SimpleRNN", "LSTM"]:
        src = data["models"][mtype]["source"]
        badge_class = "pill-green" if src == "loaded" else "pill-blue"
        st.markdown(
            f'<span class="pill {badge_class}">{mtype}: {src}</span> ',
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction metrics ────────────────────────────────────────────────────
    cols = st.columns(6)
    metrics_display = {}
    for idx, mtype in enumerate(["SimpleRNN", "LSTM"]):
        m      = data["models"][mtype]
        y_pred = scaler.inverse_transform(m["y_pred_scaled"]).flatten()
        y_true = scaler.inverse_transform(m["y_test_scaled"]).flatten()
        rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
        mae    = mean_absolute_error(y_true, y_pred)
        mape   = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        metrics_display[mtype] = dict(y_pred=y_pred, y_true=y_true,
                                       rmse=rmse, mae=mae, mape=mape)

    # Highlight winner
    rnn_rmse  = metrics_display["SimpleRNN"]["rmse"]
    lstm_rmse = metrics_display["LSTM"]["rmse"]
    winner    = "SimpleRNN" if rnn_rmse < lstm_rmse else "LSTM"

    for mtype, metrics in metrics_display.items():
        win_html = '<span class="win-badge">BEST</span>' if mtype == winner else ""
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:8px;
                    padding:14px 18px; margin-bottom:12px;'>
            <div style='font-family: IBM Plex Mono, monospace; font-size:0.75rem;
                        color:#58a6ff; margin-bottom:10px;'>
                {mtype}{win_html}
            </div>
        """, unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RMSE", f"{metrics['rmse']:.4f}")
        mc2.metric("MAE",  f"{metrics['mae']:.4f}")
        mc3.metric("MAPE", f"{metrics['mape']:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Prediction chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Actual vs Predicted</div>',
                unsafe_allow_html=True)

    fig = go.Figure()
    y_true = metrics_display["SimpleRNN"]["y_true"]
    fig.add_trace(go.Scatter(
        x=list(range(len(y_true))), y=y_true,
        name="Actual", line=dict(color="#e6edf3", width=2)
    ))
    colors_map = {"SimpleRNN": "#f0883e", "LSTM": "#a371f7"}
    for mtype, metrics in metrics_display.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(metrics["y_pred"]))),
            y=metrics["y_pred"],
            name=f"{mtype} (RMSE={metrics['rmse']:.2f})",
            line=dict(color=colors_map[mtype], width=1.5, dash="dot")
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      title=dict(text=f"Predictions — {HORIZONS[h]}",
                                 font=dict(size=13, color="#e6edf3")),
                      xaxis_title="Test Sample Index",
                      yaxis_title="Adj Close (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: Actual vs Predicted ──────────────────────────────────────────
    st.markdown('<div class="section-header">Actual vs Predicted Scatter</div>',
                unsafe_allow_html=True)

    fig2 = make_subplots(rows=1, cols=2,
                          subplot_titles=["SimpleRNN", "LSTM"])

    for idx, (mtype, metrics) in enumerate(metrics_display.items(), start=1):
        fig2.add_trace(go.Scatter(
            x=metrics["y_true"], y=metrics["y_pred"],
            mode="markers",
            marker=dict(color=colors_map[mtype], size=3, opacity=0.5),
            name=mtype
        ), row=1, col=idx)
        lo = min(metrics["y_true"].min(), metrics["y_pred"].min()) - 5
        hi = max(metrics["y_true"].max(), metrics["y_pred"].max()) + 5
        fig2.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", line=dict(color="#30363d", dash="dot", width=1.5),
            showlegend=False
        ), row=1, col=idx)

    fig2.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                        title=dict(text="Scatter: Actual vs Predicted (perfect = on diagonal)",
                                   font=dict(size=12, color="#e6edf3")))
    for i in [1, 2]:
        fig2.update_xaxes(title_text="Actual (USD)", gridcolor="#21262d", row=1, col=i)
        fig2.update_yaxes(title_text="Predicted (USD)", gridcolor="#21262d", row=1, col=i)

    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Model Comparison":
    st.markdown('<div class="section-header">Full Model Comparison</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading model results…"):
        model_results = get_models_and_results(data_hash, scaled_bytes, len(df))

    # Build comparison rows from your actual run results
    actual_results = [
        {"Model": "SimpleRNN", "Horizon": "Next Day",      "RMSE": 15.2671, "MAE":  9.1872, "MAPE": 2.90},
        {"Model": "LSTM",      "Horizon": "Next Day",      "RMSE": 24.8977, "MAE": 17.2109, "MAPE": 5.41},
        {"Model": "SimpleRNN", "Horizon": "Next 5 Days",   "RMSE": 63.9781, "MAE": 44.9340, "MAPE": 13.51},
        {"Model": "LSTM",      "Horizon": "Next 5 Days",   "RMSE": 38.8018, "MAE": 27.3325, "MAPE": 8.49},
        {"Model": "SimpleRNN", "Horizon": "Next 10 Days",  "RMSE": 55.8022, "MAE": 43.5321, "MAPE": 14.75},
        {"Model": "LSTM",      "Horizon": "Next 10 Days",  "RMSE": 47.0268, "MAE": 34.1000, "MAPE": 10.76},
        {"Model": "Tuned LSTM","Horizon": "Next Day",      "RMSE": 13.1162, "MAE":  8.3303, "MAPE": None},
    ]
    comp_df = pd.DataFrame(actual_results)

    # ── Highlight table ───────────────────────────────────────────────────────
    st.markdown("**Results from actual notebook run:**")

    def highlight_winner(row):
        horizon_rows = comp_df[comp_df["Horizon"] == row["Horizon"]]
        min_rmse     = horizon_rows["RMSE"].min()
        if row["RMSE"] == min_rmse:
            return ["background-color: #1a3a2a; color: #3fb950"] * len(row)
        return [""] * len(row)

    display_df = comp_df.copy()
    display_df["MAPE"] = display_df["MAPE"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
    )
    styled = display_df.style.apply(highlight_winner, axis=1)\
                              .format({"RMSE": "{:.4f}", "MAE": "{:.4f}"})\
                              .set_properties(**{
                                  "font-family": "IBM Plex Mono, monospace",
                                  "font-size": "12px"
                              })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; font-family: IBM Plex Mono, monospace;
                margin-top:6px;'>
        🟢 Green rows = best RMSE per horizon
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RMSE grouped bar chart ────────────────────────────────────────────────
    st.markdown('<div class="section-header">RMSE by Horizon</div>',
                unsafe_allow_html=True)

    plot_df = comp_df[comp_df["Model"] != "Tuned LSTM"].copy()
    horizons_order = ["Next Day", "Next 5 Days", "Next 10 Days"]
    clr_map = {"SimpleRNN": "#f0883e", "LSTM": "#a371f7"}

    fig = go.Figure()
    for mtype in ["SimpleRNN", "LSTM"]:
        subset = plot_df[plot_df["Model"] == mtype]
        subset = subset.set_index("Horizon").reindex(horizons_order).reset_index()
        fig.add_trace(go.Bar(
            x=subset["Horizon"], y=subset["RMSE"],
            name=mtype, marker_color=clr_map[mtype],
            marker_line_width=0,
            text=subset["RMSE"].round(2), textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=9, color="#e6edf3")
        ))

    fig.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                      title=dict(text="RMSE Comparison — SimpleRNN vs LSTM",
                                 font=dict(size=13, color="#e6edf3")),
                      xaxis_title="Forecast Horizon",
                      yaxis_title="RMSE (USD)",
                      bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(fig, use_container_width=True)

    # ── MAPE comparison ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">MAPE by Horizon</div>',
                unsafe_allow_html=True)

    fig2 = go.Figure()
    for mtype in ["SimpleRNN", "LSTM"]:
        subset = plot_df[plot_df["Model"] == mtype]
        subset = subset.set_index("Horizon").reindex(horizons_order).reset_index()
        fig2.add_trace(go.Bar(
            x=subset["Horizon"], y=subset["MAPE"],
            name=mtype, marker_color=clr_map[mtype],
            marker_line_width=0,
            text=subset["MAPE"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=9, color="#e6edf3")
        ))
    fig2.update_layout(**PLOTLY_LAYOUT, height=350, barmode="group",
                       title=dict(text="MAPE Comparison (%)",
                                  font=dict(size=13, color="#e6edf3")),
                       xaxis_title="Forecast Horizon",
                       yaxis_title="MAPE (%)",
                       bargap=0.25, bargroupgap=0.08)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Grid search results ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Hyperparameter Tuning — Grid Search</div>',
                unsafe_allow_html=True)

    gs_data = {
        "units_1": [64, 32, 64, 64, 32, 32, 32, 64],
        "dropout": [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
        "lr":      [0.001, 0.001, 0.001, 0.0005, 0.001, 0.0005, 0.0005, 0.0005],
        "val_loss":[0.000194, 0.000357, 0.000362, 0.000369, 0.000381, 0.000426, 0.000427, 0.000457],
        "test_rmse":[16.60, 23.29, 23.42, 22.91, 25.91, 27.95, 24.82, 26.20],
        "epochs":  [50, 23, 25, 35, 23, 25, 34, 27],
    }
    gs_df = pd.DataFrame(gs_data).sort_values("val_loss")

    def highlight_best_gs(row):
        if row.name == gs_df.index[0]:
            return ["background-color: #1a3a2a; color: #3fb950"] * len(row)
        return [""] * len(row)

    styled_gs = gs_df.style.apply(highlight_best_gs, axis=1)\
                            .format({"val_loss": "{:.6f}", "test_rmse": "{:.4f}",
                                     "lr": "{:.4f}"})\
                            .set_properties(**{
                                "font-family": "IBM Plex Mono, monospace",
                                "font-size": "11px"
                            })
    st.dataframe(styled_gs, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style='background:#1a3a2a; border:1px solid #238636; border-radius:6px;
                padding:12px 16px; margin-top:12px;'>
        <span style='font-family: IBM Plex Mono, monospace; font-size:0.75rem; color:#3fb950;'>
            🏆 BEST CONFIG: units_1=64 · dropout=0.1 · lr=0.001
        </span><br>
        <span style='font-family: IBM Plex Mono, monospace; font-size:0.72rem; color:#8b949e;'>
            Tuned LSTM RMSE: 13.1162 &nbsp;|&nbsp; Default LSTM RMSE: 24.8977 &nbsp;|&nbsp;
            Improvement: 47.3%
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Profile — Radar</div>',
                unsafe_allow_html=True)
    st.caption("Normalised scores (higher = better). RMSE/MAE/MAPE are inverted.")

    # Normalise to 0-1 (higher = better) for radar
    metrics_radar = {
        "SimpleRNN": {"1-day RMSE": 15.27, "5-day RMSE": 63.98, "10-day RMSE": 55.80,
                      "1-day MAPE": 2.90,  "5-day MAPE": 13.51, "10-day MAPE": 14.75},
        "LSTM":      {"1-day RMSE": 24.90, "5-day RMSE": 38.80, "10-day RMSE": 47.03,
                      "1-day MAPE": 5.41,  "5-day MAPE":  8.49, "10-day MAPE": 10.76},
    }
    categories = list(list(metrics_radar.values())[0].keys())
    # Invert (lower RMSE = better → higher score)
    all_vals = {cat: [metrics_radar[m][cat] for m in metrics_radar] for cat in categories}
    max_vals = {cat: max(v) for cat, v in all_vals.items()}

    # Convert hex to rgba for fillcolor
    def hex_to_rgba(hex_color, alpha=0.15):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig3 = go.Figure()
    for mtype, metrics in metrics_radar.items():
        scores = [1 - (metrics[cat] / max_vals[cat]) for cat in categories]
        scores += [scores[0]]  # close polygon
        fig3.add_trace(go.Scatterpolar(
            r=scores, theta=categories + [categories[0]],
            fill="toself", name=mtype,
            line=dict(color=clr_map[mtype]),
            fillcolor=hex_to_rgba(clr_map[mtype])
        ))

    fig3.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono, monospace", color="#8b949e", size=10),
        height=380,
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d",
                            tickfont=dict(size=8)),
            angularaxis=dict(gridcolor="#21262d",
                             tickfont=dict(size=9, color="#8b949e"))
        ),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1),
        title=dict(text="SimpleRNN vs LSTM — Performance Radar",
                   font=dict(size=12, color="#e6edf3"))
    )
    st.plotly_chart(fig3, use_container_width=True)