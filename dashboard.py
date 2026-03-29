import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Azure Demand Forecasting & Capacity Optimization",
    page_icon="bolt",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS - Premium Dark Theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --accent-blue: #6382ff;
        --accent-cyan: #22d3ee;
        --accent-green: #34d399;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
        --accent-purple: #a78bfa;
        --bg-card: #111827;
        --border: rgba(99, 130, 255, 0.12);
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(10, 14, 26, 0.85);
        backdrop-filter: blur(16px);
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Metric cards glow */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(17,24,39,0.95), rgba(26,32,53,0.9));
        border: 1px solid rgba(99, 130, 255, 0.15);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(99, 130, 255, 0.35);
        box-shadow: 0 0 25px rgba(99, 130, 255, 0.12);
        transform: translateY(-2px);
    }

    /* Metric value color */
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        opacity: 0.65;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(17,24,39,0.6);
        padding: 4px;
        border-radius: 12px;
        border: 1px solid rgba(99,130,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 14px;
        color: #8b95b0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6382ff, #22d3ee) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 2px 12px rgba(99,130,255,0.35);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a 0%, #111827 100%);
        border-right: 1px solid rgba(99,130,255,0.1);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Cards / Expanders */
    .stExpander {
        background: rgba(17,24,39,0.7);
        border: 1px solid rgba(99,130,255,0.12);
        border-radius: 14px;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 10px;
        border-color: rgba(99,130,255,0.2);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6382ff, #22d3ee);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 28px;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 20px rgba(99,130,255,0.4);
        transform: translateY(-1px);
    }

    /* Custom header styling */
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #6382ff 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #5a6380;
        font-weight: 400;
        margin-bottom: 28px;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e8ecf4;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99,130,255,0.1);
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 100px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-online {
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.25);
        color: #34d399;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Metric accent bars */
    .metric-blue { border-left: 4px solid #6382ff !important; }
    .metric-green { border-left: 4px solid #34d399 !important; }
    .metric-orange { border-left: 4px solid #f59e0b !important; }
    .metric-purple { border-left: 4px solid #a78bfa !important; }

    /* Plot image cards */
    .plot-container {
        background: rgba(17,24,39,0.7);
        border: 1px solid rgba(99,130,255,0.12);
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.35s ease;
    }
    .plot-container:hover {
        border-color: rgba(99,130,255,0.3);
        box-shadow: 0 4px 25px rgba(0,0,0,0.35);
    }
    .plot-label {
        padding: 12px 16px;
        font-size: 12px;
        font-weight: 500;
        color: #8b95b0;
        text-align: center;
        border-top: 1px solid rgba(99,130,255,0.08);
    }

    /* Divider */
    hr {
        border-color: rgba(99,130,255,0.08) !important;
    }

    /* Progress bar override */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6382ff, #22d3ee);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_historical():
    try:
        df = pd.read_csv("azure_dataset_3_service_types.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_forecast():
    try:
        df = pd.read_csv("azure_30_day_forecast.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_plots():
    plot_dir = "forecast_plots"
    if not os.path.isdir(plot_dir):
        return []
    return sorted(glob.glob(os.path.join(plot_dir, "*.png")))

@st.cache_data
def run_model_metrics():
    """Run the stacked ensemble from model2 and return metrics dict."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv("azure_dataset_3_service_types.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Service_Type", "Region", "Timestamp"]).reset_index(drop=True)

    le_service = LabelEncoder()
    le_region = LabelEncoder()
    df["Service_Type_Encoded"] = le_service.fit_transform(df["Service_Type"])
    df["Region_Encoded"] = le_region.fit_transform(df["Region"])

    def create_features(group):
        g = group.copy().sort_values("Timestamp").reset_index(drop=True)
        g["Usage_Hours"] = g["Usage_Hours"].interpolate().bfill().ffill()
        g["Azure_Demand"] = g["Azure_Demand"].interpolate().bfill().ffill()
        u, d = g["Usage_Hours"], g["Azure_Demand"]
        for lag in [1,2,3,4,5,6,7,14,21,28,30]:
            g[f"U_Lag_{lag}"] = u.shift(lag).fillna(0)
            g[f"D_Lag_{lag}"] = d.shift(lag).fillna(0)
        for w in [7,14,30]:
            g[f"U_Roll{w}_Mean"] = u.rolling(w, min_periods=1).mean()
            g[f"U_Roll{w}_Std"]  = u.rolling(w, min_periods=1).std().fillna(0)
            g[f"D_Roll{w}_Mean"] = d.rolling(w, min_periods=1).mean()
            g[f"D_Roll{w}_Std"]  = d.rolling(w, min_periods=1).std().fillna(0)
        g["U_Roll7_Min"] = u.rolling(7, min_periods=1).min()
        g["U_Roll7_Max"] = u.rolling(7, min_periods=1).max()
        g["D_Roll7_Min"] = d.rolling(7, min_periods=1).min()
        g["D_Roll7_Max"] = d.rolling(7, min_periods=1).max()
        for span in [3,7,14]:
            g[f"U_EWM_{span}"] = u.ewm(span=span, adjust=False).mean()
            g[f"D_EWM_{span}"] = d.ewm(span=span, adjust=False).mean()
        g["D_Mom_1_7"]  = g["D_Lag_1"] - g["D_Lag_7"]
        g["D_Mom_7_28"] = g["D_Lag_7"] - g["D_Lag_28"]
        g["U_Growth"]   = u.pct_change().fillna(0)
        g["D_Growth"]   = d.pct_change().fillna(0)
        g["D_Ratio_7"]  = d / (g["D_Roll7_Mean"] + 1e-6)
        g["D_Ratio_30"] = d / (g["D_Roll30_Mean"] + 1e-6)
        g["DOW"]        = g["Timestamp"].dt.dayofweek
        g["Month"]      = g["Timestamp"].dt.month
        g["Quarter"]    = g["Timestamp"].dt.quarter
        g["WOY"]        = g["Timestamp"].dt.isocalendar().week.astype(int)
        g["DOM"]        = g["Timestamp"].dt.day
        g["Year"]       = g["Timestamp"].dt.year
        g["Is_Weekend"] = g["DOW"].isin([5,6]).astype(int)
        g["Is_MonFri"]  = g["DOW"].isin([0,4]).astype(int)
        g["DOW_sin"]    = np.sin(2*np.pi*g["DOW"]/7)
        g["DOW_cos"]    = np.cos(2*np.pi*g["DOW"]/7)
        g["Month_sin"]  = np.sin(2*np.pi*g["Month"]/12)
        g["Month_cos"]  = np.cos(2*np.pi*g["Month"]/12)
        g["WOY_sin"]    = np.sin(2*np.pi*g["WOY"]/52)
        g["WOY_cos"]    = np.cos(2*np.pi*g["WOY"]/52)
        g["U_x_D_EWM7"]  = u * g["D_EWM_7"]
        g["U_x_DOW_sin"] = u * g["DOW_sin"]
        g["Spike"] = (u > g["U_Roll7_Mean"]*1.5).astype(int)
        return g

    df = df.groupby(["Service_Type","Region"]).apply(create_features).reset_index(drop=True)

    def split_group(group):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        group["split"] = "train"
        group.loc[len(group)-30:, "split"] = "test"
        return group
    df = df.groupby(["Service_Type","Region"]).apply(split_group).reset_index(drop=True)

    train_df = df[df["split"]=="train"].copy()
    test_df  = df[df["split"]=="test"].copy()

    lag_u = [f"U_Lag_{l}" for l in [1,2,3,4,5,6,7,14,21,28,30]]
    lag_d = [f"D_Lag_{l}" for l in [1,2,3,4,5,6,7,14,21,28,30]]
    roll_u = [f"U_Roll{w}_{s}" for w in [7,14,30] for s in ["Mean","Std"]] + ["U_Roll7_Min","U_Roll7_Max"]
    roll_d = [f"D_Roll{w}_{s}" for w in [7,14,30] for s in ["Mean","Std"]] + ["D_Roll7_Min","D_Roll7_Max"]
    ewm_c = [f"U_EWM_{s}" for s in [3,7,14]] + [f"D_EWM_{s}" for s in [3,7,14]]
    mom_c = ["D_Mom_1_7","D_Mom_7_28","U_Growth","D_Growth","D_Ratio_7","D_Ratio_30"]
    cal_c = ["DOW","Month","Quarter","WOY","DOM","Year","Is_Weekend","Is_MonFri",
             "DOW_sin","DOW_cos","Month_sin","Month_cos","WOY_sin","WOY_cos"]
    misc  = ["Service_Type_Encoded","Region_Encoded","Usage_Hours","U_x_D_EWM7","U_x_DOW_sin","Spike"]
    features = misc + lag_u + lag_d + roll_u + roll_d + ewm_c + mom_c + cal_c

    X_train = train_df[features]; y_train = train_df["Azure_Demand"]
    X_test  = test_df[features];  y_test  = test_df["Azure_Demand"]

    # XGBoost
    xgb = XGBRegressor(n_estimators=3000, learning_rate=0.005, max_depth=4,
        subsample=0.75, colsample_bytree=0.6, colsample_bylevel=0.6, colsample_bynode=0.6,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=15, gamma=0.2,
        early_stopping_rounds=100, eval_metric="rmse", random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

    # LightGBM
    lgb = LGBMRegressor(n_estimators=3000, learning_rate=0.005, max_depth=6, num_leaves=31,
        subsample=0.75, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
        min_child_samples=20, early_stopping_round=100, random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # CatBoost
    cb = CatBoostRegressor(iterations=3000, learning_rate=0.02, depth=5, l2_leaf_reg=3,
        subsample=0.75, colsample_bylevel=0.6, min_data_in_leaf=15,
        early_stopping_rounds=100, eval_metric="RMSE", random_seed=42, verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Stack
    xgb_tr, xgb_te = xgb.predict(X_train), xgb.predict(X_test)
    lgb_tr, lgb_te = lgb.predict(X_train), lgb.predict(X_test)
    cb_tr,  cb_te  = cb.predict(X_train),  cb.predict(X_test)

    stack_train = np.column_stack([xgb_tr, lgb_tr, cb_tr,
                                   (xgb_tr+lgb_tr)/2, (xgb_tr+cb_tr)/2, (lgb_tr+cb_tr)/2])
    stack_test  = np.column_stack([xgb_te, lgb_te, cb_te,
                                   (xgb_te+lgb_te)/2, (xgb_te+cb_te)/2, (lgb_te+cb_te)/2])

    meta = Ridge(alpha=0.5)
    meta.fit(stack_train, y_train)

    y_pred_train = meta.predict(stack_train)
    y_pred_test  = meta.predict(stack_test)

    # Residual correction
    res_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=3,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
        random_state=99, n_jobs=-1)
    res_model.fit(X_train, y_train.values - y_pred_train, verbose=0)

    y_pred_test_final  = y_pred_test  + res_model.predict(X_test)
    y_pred_train_final = y_pred_train + res_model.predict(X_train)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return {
        "train_mae":  mean_absolute_error(y_train, y_pred_train_final),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train_final)),
        "test_mae":   mean_absolute_error(y_test, y_pred_test_final),
        "test_rmse":  np.sqrt(mean_squared_error(y_test, y_pred_test_final)),
        "n_features": len(features),
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "xgb_best":   xgb.best_iteration,
        "lgb_best":   lgb.best_iteration_,
        "cb_best":    cb.best_iteration_,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8b95b0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11), bgcolor="rgba(0,0,0,0)"
    ),
    xaxis=dict(gridcolor="rgba(99,130,255,0.06)", zerolinecolor="rgba(99,130,255,0.06)"),
    yaxis=dict(gridcolor="rgba(99,130,255,0.06)", zerolinecolor="rgba(99,130,255,0.06)"),
)

PALETTE = ["#6382ff","#22d3ee","#34d399","#f59e0b","#a78bfa",
           "#ef4444","#ec4899","#14b8a6","#f97316","#8b5cf6","#06b6d4","#84cc16"]


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
hist_df     = load_historical()
forecast_df = load_forecast()
plot_paths  = load_plots()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="width:52px;height:52px;border-radius:12px;background:linear-gradient(135deg,#6382ff,#22d3ee);
             display:inline-flex;align-items:center;justify-content:center;font-size:24px;
             box-shadow:0 0 25px rgba(99,130,255,0.35);">
            <span style="filter: brightness(0) invert(1);">&#9889;</span>
        </div>
        <h2 style="margin:12px 0 2px 0;font-size:18px;font-weight:700;
            background:linear-gradient(135deg,#6382ff,#22d3ee);-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;">Azure Forecasting</h2>
        <p style="color:#5a6380;font-size:12px;margin:0;">ML-Powered Capacity Planning</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Filters
    st.markdown("##### Filters")
    if not hist_df.empty:
        all_services = sorted(hist_df["Service_Type"].unique().tolist())
        all_regions  = sorted(hist_df["Region"].unique().tolist())
    else:
        all_services = ["AI","Compute","Storage"]
        all_regions  = ["East","North","South","West"]

    selected_services = st.multiselect("Service Type", all_services, default=all_services)
    selected_regions  = st.multiselect("Region", all_regions, default=all_regions)

    st.markdown("---")

    # Model training trigger
    st.markdown("##### Model Training")
    train_btn = st.button("Train & Evaluate Model", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:8px 0;">
        <span class="status-badge status-online">
            <span class="status-dot"></span>
            System Online
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Azure Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">XGBoost + LightGBM + CatBoost Stacked Ensemble &mdash; Capacity Optimization Dashboard</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Performance Metrics", "Forecast Chart", "Demand Trends",
                "Forecast Visualizations", "Forecast Data", "Risk Alerts"])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1: PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

    if train_btn:
        with st.spinner("Training XGBoost + LightGBM + CatBoost stacked ensemble..."):
            metrics = run_model_metrics()
            st.session_state["metrics"] = metrics
    else:
        metrics = st.session_state.get("metrics", None)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train MAE", f"{metrics['train_mae']:.4f}",
                  "Excellent" if metrics['train_mae'] < 1 else "Moderate")
        c2.metric("Train RMSE", f"{metrics['train_rmse']:.4f}",
                  "Excellent" if metrics['train_rmse'] < 1 else "Moderate")
        c3.metric("Test MAE", f"{metrics['test_mae']:.4f}",
                  "On Target" if metrics['test_mae'] < 5 else "Above Target")
        c4.metric("Test RMSE", f"{metrics['test_rmse']:.4f}",
                  "TARGET MET" if metrics['test_rmse'] < 5 else "Above 5.0",
                  delta_color="normal" if metrics['test_rmse'] < 5 else "inverse")

        st.markdown("")

        # Training details
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Features", metrics["n_features"])
        d2.metric("Train Rows", f"{metrics['train_rows']:,}")
        d3.metric("Test Rows", f"{metrics['test_rows']:,}")

        st.markdown("")

        e1, e2, e3 = st.columns(3)
        e1.metric("XGBoost Best Iter", metrics["xgb_best"])
        e2.metric("LightGBM Best Iter", metrics["lgb_best"])
        e3.metric("CatBoost Best Iter", metrics["cb_best"])

        if metrics["test_rmse"] < 5:
            st.success(f"TARGET MET -- Test RMSE {metrics['test_rmse']:.4f} is below 5.0")
        else:
            st.warning(f"Test RMSE {metrics['test_rmse']:.4f} is above the 5.0 target.")
    else:
        st.info("Click **Train & Evaluate Model** in the sidebar to train the stacked ensemble and see metrics here.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2: FORECAST CHART
# ──────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">30-Day Demand Forecast</div>', unsafe_allow_html=True)

    if not forecast_df.empty:
        # Apply filters
        fdf = forecast_df[
            forecast_df["Service_Type"].isin(selected_services) &
            forecast_df["Region"].isin(selected_regions)
        ]

        if fdf.empty:
            st.warning("No forecast data for selected filters.")
        else:
            fdf["Series"] = fdf["Service_Type"] + " - " + fdf["Region"]
            fig = px.line(fdf, x="Timestamp", y="Azure_Demand", color="Series",
                          color_discrete_sequence=PALETTE,
                          labels={"Azure_Demand": "Predicted Demand", "Timestamp": "Date"})
            fig.update_traces(line=dict(width=2.5), mode="lines+markers",
                              marker=dict(size=4))
            fig.update_layout(**PLOTLY_LAYOUT, height=480,
                              title=dict(text="30-Day Azure Demand Forecast by Series",
                                         font=dict(size=15, color="#e8ecf4")))
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            st.markdown("")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Avg Demand", f"{fdf['Azure_Demand'].mean():.2f}")
            s2.metric("Max Demand", f"{fdf['Azure_Demand'].max():.2f}")
            s3.metric("Min Demand", f"{fdf['Azure_Demand'].min():.2f}")
            s4.metric("Total Records", f"{len(fdf):,}")
    else:
        st.warning("No forecast file found. Please run model2.py first to generate `azure_30_day_forecast.csv`.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3: DEMAND TRENDS (Historical vs Forecast)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Historical vs Forecast Trends</div>', unsafe_allow_html=True)

    if not hist_df.empty and not forecast_df.empty:
        # Aggregate to daily totals
        h_agg = hist_df[
            hist_df["Service_Type"].isin(selected_services) &
            hist_df["Region"].isin(selected_regions)
        ].groupby(hist_df["Timestamp"].dt.date)["Azure_Demand"].sum().reset_index()
        h_agg["Timestamp"] = pd.to_datetime(h_agg["Timestamp"])

        f_agg = forecast_df[
            forecast_df["Service_Type"].isin(selected_services) &
            forecast_df["Region"].isin(selected_regions)
        ].groupby(forecast_df["Timestamp"].dt.date)["Azure_Demand"].sum().reset_index()
        f_agg["Timestamp"] = pd.to_datetime(f_agg["Timestamp"])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=h_agg["Timestamp"], y=h_agg["Azure_Demand"],
            mode="lines", name="Historical Actual",
            line=dict(color="#6382ff", width=1.8),
            fill="tozeroy", fillcolor="rgba(99,130,255,0.06)"
        ))
        fig2.add_trace(go.Scatter(
            x=f_agg["Timestamp"], y=f_agg["Azure_Demand"],
            mode="lines+markers", name="30-Day Forecast",
            line=dict(color="#ef4444", width=3, dash="dash"),
            marker=dict(size=5, color="#ef4444")
        ))
        # Divider line
        if not f_agg.empty:
            fig2.add_vline(x=f_agg["Timestamp"].iloc[0], line_dash="dot",
                           line_color="rgba(255,255,255,0.2)", line_width=1)
            fig2.add_annotation(x=f_agg["Timestamp"].iloc[0], y=1.05, yref="paper",
                                text="Forecast Start", showarrow=False,
                                font=dict(size=10, color="#5a6380"))

        fig2.update_layout(**PLOTLY_LAYOUT, height=420,
                           title=dict(text="Actual vs Forecast (Aggregated Daily)",
                                      font=dict(size=15, color="#e8ecf4")),
                           yaxis_title="Total Azure Demand")
        st.plotly_chart(fig2, use_container_width=True)

        # Per-service breakdown
        st.markdown("")
        st.markdown('<div class="section-header">Per-Service Breakdown</div>', unsafe_allow_html=True)

        group_col = st.radio("Group by", ["Service_Type", "Region"], horizontal=True)

        h_grp = hist_df[
            hist_df["Service_Type"].isin(selected_services) &
            hist_df["Region"].isin(selected_regions)
        ].copy()
        h_grp["Month"] = h_grp["Timestamp"].dt.to_period("M")
        monthly = h_grp.groupby(["Month", group_col])["Azure_Demand"].mean().reset_index()
        monthly["Month"] = monthly["Month"].dt.to_timestamp()

        fig3 = px.line(monthly, x="Month", y="Azure_Demand", color=group_col,
                       color_discrete_sequence=PALETTE,
                       labels={"Azure_Demand": "Avg Demand", "Month": ""})
        fig3.update_traces(line=dict(width=2.5))
        fig3.update_layout(**PLOTLY_LAYOUT, height=360,
                           title=dict(text=f"Monthly Avg Demand by {group_col.replace('_',' ')}",
                                      font=dict(size=14, color="#e8ecf4")))
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("Historical or forecast data not available.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 4: VISUALIZATIONS (Plot Gallery)
# ──────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Forecast Visualizations</div>', unsafe_allow_html=True)

    if plot_paths:
        # Category filter
        cat_options = ["All", "Overview", "Seasonality", "Weekly", "Comparison"]
        selected_cat = st.radio("Plot Category", cat_options, horizontal=True)

        def matches_category(path, cat):
            if cat == "All": return True
            name = os.path.basename(path).lower()
            if cat == "Overview":     return "overview" in name
            if cat == "Seasonality":  return "dow" in name or "seasonality" in name
            if cat == "Weekly":       return "weekly" in name
            if cat == "Comparison":   return "comparison" in name
            return True

        filtered_plots = [p for p in plot_paths if matches_category(p, selected_cat)]

        if not filtered_plots:
            st.info("No plots match this category.")
        else:
            # Service/region filter for plots
            if selected_cat != "Comparison":
                svc_filter = st.selectbox("Filter by Service", ["All"] + all_services, key="plot_svc")
                reg_filter = st.selectbox("Filter by Region", ["All"] + all_regions, key="plot_reg")

                if svc_filter != "All":
                    filtered_plots = [p for p in filtered_plots if svc_filter in os.path.basename(p)]
                if reg_filter != "All":
                    filtered_plots = [p for p in filtered_plots if reg_filter in os.path.basename(p)]

            # Grid display
            cols_per_row = 2 if selected_cat in ["Overview", "Comparison"] else 3
            for i in range(0, len(filtered_plots), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(filtered_plots):
                        path = filtered_plots[idx]
                        label = os.path.basename(path).replace(".png","").replace("_"," ").title()
                        with col:
                            st.image(path, use_container_width=True)
                            st.caption(label)
    else:
        st.warning("No plots found. Run model2.py to generate forecast plots in `forecast_plots/`.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 5: FORECAST DATA TABLE
# ──────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">Forecast Data</div>', unsafe_allow_html=True)

    if not forecast_df.empty:
        fdf_table = forecast_df[
            forecast_df["Service_Type"].isin(selected_services) &
            forecast_df["Region"].isin(selected_regions)
        ][["Timestamp","Service_Type","Region","Azure_Demand"]].copy()
        fdf_table["Timestamp"] = fdf_table["Timestamp"].dt.strftime("%b %d, %Y")

        st.markdown(f"**{len(fdf_table):,} records** matching filters")

        st.dataframe(
            fdf_table.rename(columns={
                "Timestamp": "Date",
                "Service_Type": "Service Type",
                "Region": "Region",
                "Azure_Demand": "Predicted Demand"
            }),
            use_container_width=True,
            height=500,
        )

        st.markdown("")
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Forecast CSV",
            data=csv,
            file_name="azure_30_day_forecast.csv",
            mime="text/csv",
        )
    else:
        st.warning("No forecast data available.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 6: RISK ALERTS
# ──────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-header">Capacity Risk Alerts</div>', unsafe_allow_html=True)

    if not forecast_df.empty:
        # Dynamic risk analysis based on forecast data
        for svc in selected_services:
            for reg in selected_regions:
                series = forecast_df[
                    (forecast_df["Service_Type"] == svc) &
                    (forecast_df["Region"] == reg)
                ].sort_values("Timestamp")

                if series.empty:
                    continue

                avg_demand = series["Azure_Demand"].mean()
                max_demand = series["Azure_Demand"].max()
                trend = series["Azure_Demand"].iloc[-1] - series["Azure_Demand"].iloc[0]

                # Historical baseline for comparison
                if not hist_df.empty:
                    hist_avg = hist_df[
                        (hist_df["Service_Type"] == svc) & (hist_df["Region"] == reg)
                    ]["Azure_Demand"].mean()
                    pct_change = ((avg_demand - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
                else:
                    pct_change = 0

                if pct_change > 5:
                    st.error(f"**CRITICAL** - {svc} ({reg}): Forecast demand is **{pct_change:.1f}%** above historical average. Max projected: {max_demand:.1f}")
                elif pct_change > 2:
                    st.warning(f"**WARNING** - {svc} ({reg}): Demand trending **{pct_change:.1f}%** above baseline. Monitor closely.")
                elif trend > 5:
                    st.warning(f"**WARNING** - {svc} ({reg}): Upward trend detected (+{trend:.1f} over 30 days)")
                else:
                    st.success(f"**OK** - {svc} ({reg}): Demand stable (avg: {avg_demand:.1f}, trend: {trend:+.1f})")
    else:
        st.info("Load forecast data to see risk alerts.")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:12px 0;color:#5a6380;font-size:12px;">
    Azure Demand Forecasting & Capacity Optimization System &middot;
    XGBoost + LightGBM + CatBoost Stacked Ensemble &middot;
    Group 2
</div>
""", unsafe_allow_html=True)
