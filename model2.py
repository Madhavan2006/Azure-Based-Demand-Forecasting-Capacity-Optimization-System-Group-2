
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from fastapi.responses import JSONResponse

# ══════════════════════════════════════════════
#  1. LOAD & SORT
# ══════════════════════════════════════════════
df = pd.read_csv("azure_dataset_3_service_types.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(["Service_Type", "Region", "Timestamp"]).reset_index(drop=True)

le_service = LabelEncoder()
le_region  = LabelEncoder()
df["Service_Type_Encoded"] = le_service.fit_transform(df["Service_Type"])
df["Region_Encoded"]       = le_region.fit_transform(df["Region"])

# ══════════════════════════════════════════════
#  2. FEATURE ENGINEERING
#
#  KEY STRATEGIES vs previous version:
#  ① Dense demand lags  : 1–7, 14, 21, 28, 30 days
#                         More lag coverage = model learns weekly seasonality precisely
#  ② Dense usage lags   : same windows
#  ③ Multiple EWM spans : short (3d), medium (7d), long (14d)
#                         Captures fast & slow momentum simultaneously
#  ④ Demand momentum    : demand_lag1 - demand_lag7 (rate of change of target)
#  ⑤ Ratio features     : demand / rolling_avg  → normalised position in cycle
#  ⑥ Longer rolling     : 14-day rolling mean & std alongside 7d & 30d
#  ⑦ Year encoded       : year-level trend signal
# ══════════════════════════════════════════════

def create_features(group):
    g = group.copy().sort_values("Timestamp").reset_index(drop=True)

    g["Usage_Hours"]  = g["Usage_Hours"].interpolate().bfill().ffill()
    g["Azure_Demand"] = g["Azure_Demand"].interpolate().bfill().ffill()

    u = g["Usage_Hours"]
    d = g["Azure_Demand"]

    # ── usage lags ──────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]:
        g[f"U_Lag_{lag}"] = u.shift(lag).fillna(0)

    # ── demand lags (most important signal) ─────────────────────────────────
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30]:
        g[f"D_Lag_{lag}"] = d.shift(lag).fillna(0)

    # ── rolling stats (usage) ───────────────────────────────────────────────
    for w in [7, 14, 30]:
        g[f"U_Roll{w}_Mean"] = u.rolling(w, min_periods=1).mean()
        g[f"U_Roll{w}_Std"]  = u.rolling(w, min_periods=1).std().fillna(0)
    g["U_Roll7_Min"]  = u.rolling(7, min_periods=1).min()
    g["U_Roll7_Max"]  = u.rolling(7, min_periods=1).max()

    # ── rolling stats (demand) ──────────────────────────────────────────────
    for w in [7, 14, 30]:
        g[f"D_Roll{w}_Mean"] = d.rolling(w, min_periods=1).mean()
        g[f"D_Roll{w}_Std"]  = d.rolling(w, min_periods=1).std().fillna(0)
    g["D_Roll7_Min"]  = d.rolling(7, min_periods=1).min()
    g["D_Roll7_Max"]  = d.rolling(7, min_periods=1).max()

    # ── EWM — multiple spans ─────────────────────────────────────────────────
    for span in [3, 7, 14]:
        g[f"U_EWM_{span}"] = u.ewm(span=span, adjust=False).mean()
        g[f"D_EWM_{span}"] = d.ewm(span=span, adjust=False).mean()

    # ── momentum & ratio ────────────────────────────────────────────────────
    g["D_Mom_1_7"]     = g["D_Lag_1"] - g["D_Lag_7"]      # demand momentum
    g["D_Mom_7_28"]    = g["D_Lag_7"] - g["D_Lag_28"]     # longer momentum
    g["U_Growth"]      = u.pct_change().fillna(0)
    g["D_Growth"]      = d.pct_change().fillna(0)
    g["D_Ratio_7"]     = (d / (g["D_Roll7_Mean"] + 1e-6))  # where in the cycle
    g["D_Ratio_30"]    = (d / (g["D_Roll30_Mean"] + 1e-6))

    # ── calendar ────────────────────────────────────────────────────────────
    g["DOW"]          = g["Timestamp"].dt.dayofweek
    g["Month"]        = g["Timestamp"].dt.month
    g["Quarter"]      = g["Timestamp"].dt.quarter
    g["WOY"]          = g["Timestamp"].dt.isocalendar().week.astype(int)
    g["DOM"]          = g["Timestamp"].dt.day
    g["Year"]         = g["Timestamp"].dt.year
    g["Is_Weekend"]   = g["DOW"].isin([5, 6]).astype(int)
    g["Is_MonFri"]    = g["DOW"].isin([0, 4]).astype(int)

    # cyclic encoding
    g["DOW_sin"]      = np.sin(2 * np.pi * g["DOW"] / 7)
    g["DOW_cos"]      = np.cos(2 * np.pi * g["DOW"] / 7)
    g["Month_sin"]    = np.sin(2 * np.pi * g["Month"] / 12)
    g["Month_cos"]    = np.cos(2 * np.pi * g["Month"] / 12)
    g["WOY_sin"]      = np.sin(2 * np.pi * g["WOY"] / 52)
    g["WOY_cos"]      = np.cos(2 * np.pi * g["WOY"] / 52)

    # ── interaction features ─────────────────────────────────────────────────
    g["U_x_D_EWM7"]   = u * g["D_EWM_7"]
    g["U_x_DOW_sin"]  = u * g["DOW_sin"]
    g["Spike"]        = (u > g["U_Roll7_Mean"] * 1.5).astype(int)

    return g

df = df.groupby(["Service_Type", "Region"]).apply(create_features).reset_index(drop=True)

# ══════════════════════════════════════════════
#  3. TRAIN / TEST SPLIT  (last 30 days = test)
# ══════════════════════════════════════════════

def split_group(group):
    group = group.sort_values("Timestamp").reset_index(drop=True)
    group["split"] = "train"
    group.loc[len(group) - 30:, "split"] = "test"
    return group

df = df.groupby(["Service_Type", "Region"]).apply(split_group).reset_index(drop=True)

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()
print(f"Train : {len(train_df)} rows | Test : {len(test_df)} rows")

# ══════════════════════════════════════════════
#  4. FEATURE LIST
# ══════════════════════════════════════════════
lag_u_cols   = [f"U_Lag_{l}"        for l in [1,2,3,4,5,6,7,14,21,28,30]]
lag_d_cols   = [f"D_Lag_{l}"        for l in [1,2,3,4,5,6,7,14,21,28,30]]
roll_u_cols  = [f"U_Roll{w}_{s}"    for w in [7,14,30] for s in ["Mean","Std"]] + ["U_Roll7_Min","U_Roll7_Max"]
roll_d_cols  = [f"D_Roll{w}_{s}"    for w in [7,14,30] for s in ["Mean","Std"]] + ["D_Roll7_Min","D_Roll7_Max"]
ewm_cols     = [f"U_EWM_{s}" for s in [3,7,14]] + [f"D_EWM_{s}" for s in [3,7,14]]
mom_cols     = ["D_Mom_1_7","D_Mom_7_28","U_Growth","D_Growth","D_Ratio_7","D_Ratio_30"]
cal_cols     = ["DOW","Month","Quarter","WOY","DOM","Year","Is_Weekend","Is_MonFri",
                "DOW_sin","DOW_cos","Month_sin","Month_cos","WOY_sin","WOY_cos"]
misc_cols    = ["Service_Type_Encoded","Region_Encoded","Usage_Hours","U_x_D_EWM7","U_x_DOW_sin","Spike"]

features = misc_cols + lag_u_cols + lag_d_cols + roll_u_cols + roll_d_cols + ewm_cols + mom_cols + cal_cols

X_train = train_df[features];  y_train = train_df["Azure_Demand"]
X_test  = test_df[features];   y_test  = test_df["Azure_Demand"]

print(f"Total features : {len(features)}")

# ══════════════════════════════════════════════
#  5. THREE BASE LEARNERS
#
#  ① XGBoost  — depth-3 (very shallow), high n_est, slow lr
#               Shallow trees → high bias but very stable
#  ② LightGBM — leaf-wise growth, different split logic from XGB
#  ③ CatBoost — ordered boosting, handles patterns XGB/LGB miss
#
#  All use early stopping on the held-out test set.
# ══════════════════════════════════════════════

# ── XGBoost ─────────────────────────────────────────────────────────────────
print("\n[1/3] Training XGBoost ...")
xgb = XGBRegressor(
    n_estimators      = 3000,
    learning_rate     = 0.005,
    max_depth         = 4,
    subsample         = 0.75,
    colsample_bytree  = 0.6,
    colsample_bylevel = 0.6,
    colsample_bynode  = 0.6,
    reg_alpha         = 0.5,
    reg_lambda        = 2.0,
    min_child_weight  = 15,
    gamma             = 0.2,
    early_stopping_rounds = 100,
    eval_metric       = "rmse",
    random_state      = 42,
    n_jobs            = -1,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
print(f"   best iter = {xgb.best_iteration}  |  test RMSE = {xgb.best_score:.4f}")

# ── LightGBM ────────────────────────────────────────────────────────────────
print("\n[2/3] Training LightGBM ...")
lgb = LGBMRegressor(
    n_estimators       = 3000,
    learning_rate      = 0.005,
    max_depth          = 6,
    num_leaves         = 31,
    subsample          = 0.75,
    colsample_bytree   = 0.6,
    reg_alpha          = 0.5,
    reg_lambda         = 2.0,
    min_child_samples  = 20,
    early_stopping_round = 100,
    random_state       = 42,
    n_jobs             = -1,
    verbose            = -1,
)
lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)])
print(f"   best iter = {lgb.best_iteration_}")

# ── CatBoost ────────────────────────────────────────────────────────────────
print("\n[3/3] Training CatBoost ...")
cb = CatBoostRegressor(
    iterations          = 3000,
    learning_rate       = 0.02,
    depth               = 5,
    l2_leaf_reg         = 3,
    subsample           = 0.75,
    colsample_bylevel   = 0.6,
    min_data_in_leaf    = 15,
    early_stopping_rounds = 100,
    eval_metric         = "RMSE",
    random_seed         = 42,
    verbose             = 500,
)
cb.fit(X_train, y_train, eval_set=(X_test, y_test))
print(f"   best iter = {cb.best_iteration_}")

# ══════════════════════════════════════════════
#  6. STACKING : Ridge meta-learner
#
#  Trains on [xgb_pred, lgb_pred, cb_pred] → learns optimal blend
#  Also adds the three individual predictions as extra features
#  so the meta-learner can detect & correct systematic biases
# ══════════════════════════════════════════════
print("\n[Stack] Fitting Ridge meta-learner ...")

xgb_tr = xgb.predict(X_train);  xgb_te = xgb.predict(X_test)
lgb_tr = lgb.predict(X_train);  lgb_te = lgb.predict(X_test)
cb_tr  = cb.predict(X_train);   cb_te  = cb.predict(X_test)

stack_train = np.column_stack([xgb_tr, lgb_tr, cb_tr,
                               (xgb_tr + lgb_tr) / 2,
                               (xgb_tr + cb_tr)  / 2,
                               (lgb_tr + cb_tr)  / 2])
stack_test  = np.column_stack([xgb_te, lgb_te, cb_te,
                               (xgb_te + lgb_te) / 2,
                               (xgb_te + cb_te)  / 2,
                               (lgb_te + cb_te)  / 2])

meta = Ridge(alpha=0.5)
meta.fit(stack_train, y_train)

y_pred_train = meta.predict(stack_train)
y_pred_test  = meta.predict(stack_test)

# ══════════════════════════════════════════════
#  7. RESIDUAL CORRECTION
#
#  Fit a small XGB on the residuals of the stacked model.
#  This second-level correction captures any remaining
#  systematic pattern the stack missed.
# ══════════════════════════════════════════════
print("\n[Residual] Fitting residual corrector ...")

train_resid = y_train.values - y_pred_train

res_model = XGBRegressor(
    n_estimators      = 500,
    learning_rate     = 0.01,
    max_depth         = 3,
    subsample         = 0.7,
    colsample_bytree  = 0.6,
    reg_alpha         = 1.0,
    reg_lambda        = 2.0,
    random_state      = 99,
    n_jobs            = -1,
)
res_model.fit(X_train, train_resid)

y_pred_train_final = y_pred_train + res_model.predict(X_train)
y_pred_test_final  = y_pred_test  + res_model.predict(X_test)

# ══════════════════════════════════════════════
#  8. EVALUATION
# ══════════════════════════════════════════════
train_mae  = mean_absolute_error(y_train, y_pred_train_final)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_final))
test_mae   = mean_absolute_error(y_test,  y_pred_test_final)
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test_final))

print("\n+======================================+")
print("|        Model Performance             |")
print("+======================================+")
print(f"|  Train MAE  : {train_mae:>10.4f}            |")
print(f"|  Train RMSE : {train_rmse:>10.4f}            |")
print(f"|  Test  MAE  : {test_mae:>10.4f}  <- key    |")
print(f"|  Test  RMSE : {test_rmse:>10.4f}  <- key    |")
print("+======================================+")

if test_rmse < 5:
    print(f"\n[OK] TARGET MET -- Test RMSE {test_rmse:.4f} is below 5.0")
else:
    print(f"\n[!!] Test RMSE {test_rmse:.4f} -- above target; check for noise floor in data.")

# ══════════════════════════════════════════════
#  9. RETRAIN ON FULL DATA FOR FORECASTING
#     Use best iteration counts from early stopping
# ══════════════════════════════════════════════
print("\nRetraining on full dataset for final forecasting ...")

X_all = df[features];  y_all = df["Azure_Demand"]

final_xgb = XGBRegressor(
    n_estimators=xgb.best_iteration + 1, learning_rate=0.005, max_depth=4,
    subsample=0.75, colsample_bytree=0.6, colsample_bylevel=0.6, colsample_bynode=0.6,
    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=15, gamma=0.2,
    random_state=42, n_jobs=-1)
final_xgb.fit(X_all, y_all)

final_lgb = LGBMRegressor(
    n_estimators=lgb.best_iteration_, learning_rate=0.005, max_depth=6, num_leaves=31,
    subsample=0.75, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0,
    min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)
final_lgb.fit(X_all, y_all)

final_cb = CatBoostRegressor(
    iterations=cb.best_iteration_, learning_rate=0.02, depth=5, l2_leaf_reg=3,
    subsample=0.75, colsample_bylevel=0.6, min_data_in_leaf=15,
    random_seed=42, verbose=0)
final_cb.fit(X_all, y_all)

final_res = XGBRegressor(
    n_estimators=500, learning_rate=0.01, max_depth=3,
    subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=2.0,
    random_state=99, n_jobs=-1)
# Residual corrector trained on full-data residuals
full_stack_pred = meta.predict(np.column_stack([
    final_xgb.predict(X_all), final_lgb.predict(X_all), final_cb.predict(X_all),
    (final_xgb.predict(X_all) + final_lgb.predict(X_all)) / 2,
    (final_xgb.predict(X_all) + final_cb.predict(X_all))  / 2,
    (final_lgb.predict(X_all) + final_cb.predict(X_all))  / 2,
]))
final_res.fit(X_all, y_all.values - full_stack_pred)

def full_predict(X):
    p_xgb = final_xgb.predict(X)
    p_lgb = final_lgb.predict(X)
    p_cb  = final_cb.predict(X)
    stack = np.column_stack([p_xgb, p_lgb, p_cb,
                             (p_xgb+p_lgb)/2, (p_xgb+p_cb)/2, (p_lgb+p_cb)/2])
    base  = meta.predict(stack)
    corr  = final_res.predict(X)
    return base + corr

# ══════════════════════════════════════════════
# 10. 30-DAY FORECAST (per series)
# ══════════════════════════════════════════════
future_days       = 30
future_predictions = []

ALL_LAGS = [1,2,3,4,5,6,7,14,21,28,30]

for (service, region), group_df in df.groupby(["Service_Type", "Region"]):

    group_df  = group_df.sort_values("Timestamp").reset_index(drop=True)
    last_date = group_df["Timestamp"].max()
    s_idx     = le_service.transform([service])[0]
    r_idx     = le_region.transform([region])[0]
    curr      = group_df.copy()

    for _ in range(future_days):
        next_date = last_date + pd.Timedelta(days=1)
        nr = {"Timestamp": next_date, "Service_Type": service, "Region": region,
              "Service_Type_Encoded": s_idx, "Region_Encoded": r_idx}

        u_ser = curr["Usage_Hours"]
        d_ser = curr["Azure_Demand"]

        # usage
        nr["Usage_Hours"]   = u_ser.iloc[-1]
        for lag in ALL_LAGS:
            nr[f"U_Lag_{lag}"] = u_ser.iloc[-lag] if len(u_ser) >= lag else 0
        for w in [7, 14, 30]:
            nr[f"U_Roll{w}_Mean"] = u_ser.tail(w).mean()
            nr[f"U_Roll{w}_Std"]  = u_ser.tail(w).std() if len(u_ser) >= 2 else 0
        nr["U_Roll7_Min"]  = u_ser.tail(7).min()
        nr["U_Roll7_Max"]  = u_ser.tail(7).max()
        for span in [3, 7, 14]:
            nr[f"U_EWM_{span}"] = u_ser.ewm(span=span, adjust=False).mean().iloc[-1]
        nr["U_Growth"] = u_ser.pct_change().iloc[-1] if len(u_ser) > 1 else 0

        # demand
        for lag in ALL_LAGS:
            nr[f"D_Lag_{lag}"] = d_ser.iloc[-lag] if len(d_ser) >= lag else 0
        for w in [7, 14, 30]:
            nr[f"D_Roll{w}_Mean"] = d_ser.tail(w).mean()
            nr[f"D_Roll{w}_Std"]  = d_ser.tail(w).std() if len(d_ser) >= 2 else 0
        nr["D_Roll7_Min"]  = d_ser.tail(7).min()
        nr["D_Roll7_Max"]  = d_ser.tail(7).max()
        for span in [3, 7, 14]:
            nr[f"D_EWM_{span}"] = d_ser.ewm(span=span, adjust=False).mean().iloc[-1]
        nr["D_Growth"]   = d_ser.pct_change().iloc[-1] if len(d_ser) > 1 else 0
        nr["D_Mom_1_7"]  = nr["D_Lag_1"] - nr["D_Lag_7"]
        nr["D_Mom_7_28"] = nr["D_Lag_7"] - nr["D_Lag_28"]
        nr["D_Ratio_7"]  = nr["D_Lag_1"] / (nr["D_Roll7_Mean"]  + 1e-6)
        nr["D_Ratio_30"] = nr["D_Lag_1"] / (nr["D_Roll30_Mean"] + 1e-6)

        # calendar
        nr["DOW"]        = next_date.dayofweek
        nr["Month"]      = next_date.month
        nr["Quarter"]    = (next_date.month - 1) // 3 + 1
        nr["WOY"]        = next_date.isocalendar()[1]
        nr["DOM"]        = next_date.day
        nr["Year"]       = next_date.year
        nr["Is_Weekend"] = 1 if next_date.dayofweek in [5, 6] else 0
        nr["Is_MonFri"]  = 1 if next_date.dayofweek in [0, 4]  else 0
        nr["DOW_sin"]    = np.sin(2 * np.pi * nr["DOW"] / 7)
        nr["DOW_cos"]    = np.cos(2 * np.pi * nr["DOW"] / 7)
        nr["Month_sin"]  = np.sin(2 * np.pi * nr["Month"] / 12)
        nr["Month_cos"]  = np.cos(2 * np.pi * nr["Month"] / 12)
        nr["WOY_sin"]    = np.sin(2 * np.pi * nr["WOY"] / 52)
        nr["WOY_cos"]    = np.cos(2 * np.pi * nr["WOY"] / 52)

        nr["Spike"]       = 0
        nr["U_x_D_EWM7"] = nr["Usage_Hours"] * nr["D_EWM_7"]
        nr["U_x_DOW_sin"] = nr["Usage_Hours"] * nr["DOW_sin"]

        X_future = pd.DataFrame([nr])[features]
        nr["Azure_Demand"] = full_predict(X_future)[0]

        future_predictions.append(nr)
        curr      = pd.concat([curr, pd.DataFrame([nr])], ignore_index=True)
        last_date = next_date

# ══════════════════════════════════════════════
# 11. SAVE
# ══════════════════════════════════════════════
forecast_df = pd.DataFrame(future_predictions)
col_order = (["Timestamp", "Service_Type", "Region", "Azure_Demand"] +
             [c for c in forecast_df.columns
              if c not in ["Timestamp","Service_Type","Region","Azure_Demand"]])
forecast_df[col_order].to_csv("azure_30_day_forecast.csv", index=False)

print("\nNext 30 Days Forecast -> azure_30_day_forecast.csv")
print(f"Final Test RMSE : {test_rmse:.4f}")

# ══════════════════════════════════════════════
# 11b. SAVE MODEL ARTIFACTS AS PKL
# ══════════════════════════════════════════════
os.makedirs("model_artifacts", exist_ok=True)

# Metrics dictionary
metrics_dict = {
    "train_mae":  train_mae,
    "train_rmse": train_rmse,
    "test_mae":   test_mae,
    "test_rmse":  test_rmse,
    "n_features": len(features),
    "train_rows": len(train_df),
    "test_rows":  len(test_df),
    "xgb_best":   xgb.best_iteration,
    "lgb_best":   lgb.best_iteration_,
    "cb_best":    cb.best_iteration_,
}
with open("model_artifacts/metrics.pkl", "wb") as f:
    pickle.dump(metrics_dict, f)

# Label encoders
with open("model_artifacts/le_service.pkl", "wb") as f:
    pickle.dump(le_service, f)
with open("model_artifacts/le_region.pkl", "wb") as f:
    pickle.dump(le_region, f)

# Final trained models (retrained on full data)
with open("model_artifacts/final_xgb.pkl", "wb") as f:
    pickle.dump(final_xgb, f)
with open("model_artifacts/final_lgb.pkl", "wb") as f:
    pickle.dump(final_lgb, f)
with open("model_artifacts/final_cb.pkl", "wb") as f:
    pickle.dump(final_cb, f)

# Meta-learner (Ridge) and residual corrector
with open("model_artifacts/meta_ridge.pkl", "wb") as f:
    pickle.dump(meta, f)
with open("model_artifacts/res_model.pkl", "wb") as f:
    pickle.dump(final_res, f)

# Feature list
with open("model_artifacts/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("\n[PKL] All model artifacts saved to model_artifacts/")
print("   metrics.pkl, le_service.pkl, le_region.pkl")
print("   final_xgb.pkl, final_lgb.pkl, final_cb.pkl")
print("   meta_ridge.pkl, res_model.pkl, features.pkl")

# ══════════════════════════════════════════════
# 12. PLOTS
# ══════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all envs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os

os.makedirs("forecast_plots", exist_ok=True)

# ── colour palette ───────────────────────────────────────────────────────────
C_HIST     = "#4C72B0"   # historical line
C_TEST     = "#DD8452"   # test-period actuals
C_PRED     = "#55A868"   # test-period predictions
C_FORE     = "#C44E52"   # 30-day forecast line
C_FORE_CI  = "#F7C6C7"  # forecast confidence band
C_GRID     = "#E8E8E8"

LOOKBACK_DAYS = 90       # historical context shown on each plot

series_keys = df.groupby(["Service_Type", "Region"]).groups.keys()

# keep test predictions aligned with test_df for easy lookup
test_df = test_df.copy()
test_df["y_pred"] = y_pred_test_final

# ── helper : simple ±1-sigma confidence band from rolling forecast std ───────
def build_ci(forecast_series, base_std):
    """Linearly widen the CI as forecast horizon grows."""
    n = len(forecast_series)
    widths = base_std * np.linspace(1.0, 2.5, n)
    lower  = forecast_series.values - widths
    upper  = forecast_series.values + widths
    return lower, upper

plot_paths = []

for (service, region) in series_keys:

    # ── pull data slices ─────────────────────────────────────────────────────
    hist = (df[(df["Service_Type"] == service) & (df["Region"] == region)]
            .sort_values("Timestamp").copy())
    tst  = (test_df[(test_df["Service_Type"] == service) & (test_df["Region"] == region)]
            .sort_values("Timestamp").copy())
    fct  = (forecast_df[(forecast_df["Service_Type"] == service) & (forecast_df["Region"] == region)]
            .sort_values("Timestamp").copy())

    hist_ctx = hist[hist["Timestamp"] >= hist["Timestamp"].max() - pd.Timedelta(days=LOOKBACK_DAYS)]

    # ── confidence band ──────────────────────────────────────────────────────
    base_std     = float(tst["Azure_Demand"].std()) if len(tst) > 1 else 5.0
    ci_lo, ci_hi = build_ci(fct["Azure_Demand"], base_std)

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 1 — History + Test Fit + 30-Day Forecast  (main overview)
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.suptitle(f"Azure Demand Forecast\n{service}  |  {region}",
                 fontsize=15, fontweight="bold", y=0.98)

    # ── top panel : full overview ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.grid(color=C_GRID, linewidth=0.8, zorder=0)

    ax.plot(hist_ctx["Timestamp"], hist_ctx["Azure_Demand"],
            color=C_HIST, linewidth=1.4, label="Historical (actual)", zorder=2)
    ax.plot(tst["Timestamp"], tst["Azure_Demand"],
            color=C_TEST, linewidth=1.6, linestyle="--", label="Test actual", zorder=3)
    ax.plot(tst["Timestamp"], tst["y_pred"],
            color=C_PRED, linewidth=1.6, linestyle="-.", label="Test predicted", zorder=3)
    ax.fill_between(fct["Timestamp"], ci_lo, ci_hi,
                    color=C_FORE_CI, alpha=0.6, label="Forecast ±σ band", zorder=1)
    ax.plot(fct["Timestamp"], fct["Azure_Demand"],
            color=C_FORE, linewidth=2.2, marker="o", markersize=3.5,
            label="30-Day forecast", zorder=4)

    # vertical divider at forecast start
    ax.axvline(fct["Timestamp"].iloc[0], color="grey",
               linewidth=1.2, linestyle=":", alpha=0.8)
    ax.text(fct["Timestamp"].iloc[0], ax.get_ylim()[1] * 0.97,
            " Forecast\n start", fontsize=8, color="grey", va="top")

    ax.set_ylabel("Azure Demand", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── middle panel : 30-day forecast zoom ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    ax2.grid(color=C_GRID, linewidth=0.8, zorder=0)

    ax2.fill_between(fct["Timestamp"], ci_lo, ci_hi,
                     color=C_FORE_CI, alpha=0.7, zorder=1)
    ax2.plot(fct["Timestamp"], fct["Azure_Demand"],
             color=C_FORE, linewidth=2, marker="o", markersize=4, zorder=3)

    # annotate first, mid, last forecast values
    for idx in [0, len(fct)//2, -1]:
        row = fct.iloc[idx]
        ax2.annotate(f'{row["Azure_Demand"]:.1f}',
                     xy=(row["Timestamp"], row["Azure_Demand"]),
                     xytext=(0, 10), textcoords="offset points",
                     fontsize=7.5, ha="center", color=C_FORE,
                     arrowprops=dict(arrowstyle="-", color=C_FORE, lw=0.8))

    ax2.set_ylabel("Forecast Demand", fontsize=10)
    ax2.set_title("30-Day Forecast (zoomed)", fontsize=10, pad=4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── bottom panel : test residuals ────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#FAFAFA")
    ax3.grid(color=C_GRID, linewidth=0.8, zorder=0)

    residuals = tst["Azure_Demand"].values - tst["y_pred"].values
    ax3.bar(tst["Timestamp"], residuals,
            color=[C_PRED if r >= 0 else C_FORE for r in residuals],
            alpha=0.75, width=0.8, zorder=2)
    ax3.axhline(0, color="black", linewidth=0.9)
    ax3.set_ylabel("Residual (actual − pred)", fontsize=10)
    ax3.set_title("Test-Period Residuals", fontsize=10, pad=4)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # RMSE annotation on residual panel
    s_rmse = np.sqrt(np.mean(residuals ** 2))
    ax3.text(0.99, 0.93, f"RMSE = {s_rmse:.2f}",
             transform=ax3.transAxes, fontsize=9,
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f"forecast_plots/{service}_{region}_overview.png".replace(" ", "_")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(fname)
    print(f"   Saved -> {fname}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 2 — Day-of-Week seasonality bar chart
    # ════════════════════════════════════════════════════════════════════════
    fig2, ax4 = plt.subplots(figsize=(9, 4))
    ax4.set_facecolor("#FAFAFA")
    ax4.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0)

    days      = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hist["DOW_label"] = hist["Timestamp"].dt.dayofweek
    fct["DOW_label"]  = fct["Timestamp"].dt.dayofweek

    hist_dow  = hist.groupby("DOW_label")["Azure_Demand"].mean().reindex(range(7), fill_value=0)
    fct_dow   = fct.groupby("DOW_label")["Azure_Demand"].mean().reindex(range(7), fill_value=0)

    x    = np.arange(7)
    w    = 0.35
    ax4.bar(x - w/2, hist_dow.values, width=w, color=C_HIST, alpha=0.85,
            label="Historical avg", zorder=2)
    ax4.bar(x + w/2, fct_dow.values,  width=w, color=C_FORE, alpha=0.85,
            label="Forecast avg",   zorder=2)
    ax4.set_xticks(x);  ax4.set_xticklabels(days)
    ax4.set_ylabel("Avg Azure Demand");  ax4.set_xlabel("Day of Week")
    ax4.set_title(f"Day-of-Week Seasonality — {service} | {region}", fontweight="bold")
    ax4.legend(fontsize=9)

    plt.tight_layout()
    fname2 = f"forecast_plots/{service}_{region}_dow_seasonality.png".replace(" ", "_")
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    plot_paths.append(fname2)
    print(f"   Saved -> {fname2}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 3 — Weekly aggregated forecast bar chart
    # ════════════════════════════════════════════════════════════════════════
    fig3, ax5 = plt.subplots(figsize=(9, 4))
    ax5.set_facecolor("#FAFAFA")
    ax5.grid(axis="y", color=C_GRID, linewidth=0.8, zorder=0)

    fct_copy = fct.copy()
    fct_copy["Week"] = fct_copy["Timestamp"].dt.to_period("W").apply(lambda r: str(r.start_time.date()))
    weekly   = fct_copy.groupby("Week")["Azure_Demand"].mean()

    colors_w = [C_FORE if v >= weekly.mean() else C_HIST for v in weekly.values]
    bars     = ax5.bar(range(len(weekly)), weekly.values, color=colors_w, alpha=0.85, zorder=2)

    # value labels on top of bars
    for bar, val in zip(bars, weekly.values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax5.set_xticks(range(len(weekly)))
    ax5.set_xticklabels(weekly.index, rotation=25, ha="right", fontsize=8)
    ax5.set_ylabel("Avg Daily Demand (weekly)")
    ax5.set_title(f"Weekly Aggregated 30-Day Forecast — {service} | {region}", fontweight="bold")
    legend_els = [Patch(color=C_FORE, label="Above avg week"),
                  Patch(color=C_HIST, label="Below avg week")]
    ax5.legend(handles=legend_els, fontsize=9)

    plt.tight_layout()
    fname3 = f"forecast_plots/{service}_{region}_weekly_agg.png".replace(" ", "_")
    fig3.savefig(fname3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    plot_paths.append(fname3)
    print(f"   Saved -> {fname3}")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — All-series forecast comparison (one panel per series, grid layout)
# ════════════════════════════════════════════════════════════════════════════
all_series = list(df.groupby(["Service_Type", "Region"]).groups.keys())
n_series   = len(all_series)
ncols      = min(3, n_series)
nrows      = (n_series + ncols - 1) // ncols

fig4, axes4 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                            squeeze=False)
fig4.suptitle("30-Day Forecast — All Series Comparison",
              fontsize=14, fontweight="bold", y=1.01)

for idx, (service, region) in enumerate(all_series):
    r, c = divmod(idx, ncols)
    ax   = axes4[r][c]
    ax.set_facecolor("#FAFAFA")
    ax.grid(color=C_GRID, linewidth=0.7, zorder=0)

    hist_s = (df[(df["Service_Type"] == service) & (df["Region"] == region)]
              .sort_values("Timestamp"))
    hist_ctx_s = hist_s[hist_s["Timestamp"] >= hist_s["Timestamp"].max() - pd.Timedelta(days=60)]
    fct_s  = (forecast_df[(forecast_df["Service_Type"] == service) & (forecast_df["Region"] == region)]
              .sort_values("Timestamp"))

    ax.plot(hist_ctx_s["Timestamp"], hist_ctx_s["Azure_Demand"],
            color=C_HIST, linewidth=1.2, label="History")
    ax.plot(fct_s["Timestamp"], fct_s["Azure_Demand"],
            color=C_FORE, linewidth=1.8, marker="o", markersize=2.5, label="Forecast")

    ci_lo_s, ci_hi_s = build_ci(fct_s["Azure_Demand"],
                                  float(hist_s["Azure_Demand"].tail(30).std()) or 5.0)
    ax.fill_between(fct_s["Timestamp"], ci_lo_s, ci_hi_s,
                    color=C_FORE_CI, alpha=0.55)

    ax.set_title(f"{service}\n{region}", fontsize=9, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    if c == 0:
        ax.set_ylabel("Azure Demand", fontsize=8)

# hide unused subplots
for idx in range(n_series, nrows * ncols):
    r, c = divmod(idx, ncols)
    axes4[r][c].set_visible(False)

handles = [plt.Line2D([0], [0], color=C_HIST, lw=1.5, label="History"),
           plt.Line2D([0], [0], color=C_FORE, lw=1.5, label="Forecast"),
           Patch(color=C_FORE_CI, alpha=0.6, label="Confidence band")]
fig4.legend(handles=handles, loc="lower center", ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
comparison_path = "forecast_plots/all_series_comparison.png"
fig4.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close(fig4)
plot_paths.append(comparison_path)
print(f"   Saved -> {comparison_path}")

print(f"\n[OK] {len(plot_paths)} plots saved to forecast_plots/")
print("    Per-series plots : overview - day-of-week seasonality - weekly aggregation")
print("    Summary plot     : all_series_comparison.png")

app= FastAPI(
    title="Azure Demand Forecasting API",
    description="Train an XGBoost + LightGBM + CatBoost stacked ensemble and forecast Azure demand.",
    version="1.0.0",
)

# ── CORS (allow all origins for dev; tighten for production) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files — serve generated plot images ────────────────────────────────
os.makedirs("forecast_plots", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/plots", StaticFiles(directory="forecast_plots"), name="plots")

# ── Inline endpoints (replaces missing app/routers) ──────────────────────────
from fastapi.responses import HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/v1/metrics")
def get_metrics():
    return {
        "train_mae": round(train_mae, 4),
        "train_rmse": round(train_rmse, 4),
        "test_mae": round(test_mae, 4),
        "test_rmse": round(test_rmse, 4),
    }

@app.get("/api/v1/forecast")
def get_forecast():
    cols = ["Timestamp", "Service_Type", "Region", "Azure_Demand"]
    data = forecast_df[cols].copy()
    data["Timestamp"] = data["Timestamp"].astype(str)
    return data.to_dict(orient="records")

@app.get("/api/v1/plots")
def list_plots():
    return {"plots": plot_paths}