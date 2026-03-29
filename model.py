import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Load Dataset

df = pd.read_csv("azure_dataset_3_service_types.csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

# ------------------------------
#  Handle Missing Values

df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()

# ------------------------------
# Feature Engineering

df["Usage_7D_Avg"] = df["Usage_Hours"].rolling(7).mean()

df["Is_Weekend"] = df["Timestamp"].dt.dayofweek.isin([5,6]).astype(int)

# FIX NaN values (important for Prophet)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method="ffill").fillna(method="bfill")

# -----------------------------
# Holiday Effects (India)

holidays = pd.DataFrame({
    "holiday": ["diwali","christmas","new_year","republic_day","independence_day"],
    "ds": pd.to_datetime([
        "2024-11-01",
        "2024-12-25",
        "2025-01-01",
        "2025-01-26",
        "2025-08-15"
    ]),
    "lower_window": 0,
    "upper_window": 1
})

# ------------------------------
# Hyperparameter Grid

param_grid = {
    "changepoint_prior_scale":[0.01,0.05,0.1],
    "seasonality_prior_scale":[5,10]
}

grid = list(ParameterGrid(param_grid))

# -----------------------------
# Train Service-wise Models

services = df["Service_Type"].unique()

all_forecasts = []

for service in services:

    print("\nTraining for service:", service)

    service_df = df[df["Service_Type"] == service]

    # --------------------------
    # Prepare Prophet dataset

    prophet_df = service_df[[
        "Timestamp",
        "Azure_Demand",
        "Usage_Hours",
        "Usage_7D_Avg",
        "Is_Weekend"
    ]].copy()

    prophet_df.columns = [
        "ds","y","Usage_Hours","Usage_7D_Avg","Is_Weekend"
    ]

    prophet_df = prophet_df.fillna(method="ffill").fillna(method="bfill")

    # --------------------------
    #  Forecast Regressor
    
    usage_df = prophet_df[["ds","Usage_Hours"]].copy()
    usage_df.columns = ["ds","y"]

    usage_model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    usage_model.fit(usage_df)

    future_usage = usage_model.make_future_dataframe(periods=30)

    usage_forecast = usage_model.predict(future_usage)

    usage_future = usage_forecast[["ds","yhat"]]
    usage_future.columns = ["ds","Usage_Hours"]

    # --------------------------
    # Hyperparameter tuning
    
    best_rmse = float("inf")
    best_params = None

    for params in grid:

        model = Prophet(
            holidays=holidays,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"]
        )

        model.add_regressor("Usage_Hours")
        model.add_regressor("Usage_7D_Avg")
        model.add_regressor("Is_Weekend")

        model.fit(prophet_df)

        df_cv = cross_validation(
            model,
            initial="180 days",
            period="30 days",
            horizon="30 days"
        )

        df_perf = performance_metrics(df_cv)

        rmse = df_perf["rmse"].mean()

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print("Best params:", best_params)

    # --------------------------
    # Train Final Model
    
    model = Prophet(
        holidays=holidays,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"]
    )

    model.add_regressor("Usage_Hours")
    model.add_regressor("Usage_7D_Avg")
    model.add_regressor("Is_Weekend")

    model.fit(prophet_df)

    # --------------------------
    #  Future Dataset
    
    future = model.make_future_dataframe(periods=30)

    future = future.merge(
        usage_future,
        on="ds",
        how="left"
    )

    future["Usage_7D_Avg"] = future["Usage_Hours"].rolling(7).mean()

    future["Is_Weekend"] = future["ds"].dt.dayofweek.isin([5,6]).astype(int)

    future = future.replace([np.inf, -np.inf], np.nan)
    future = future.fillna(method="ffill").fillna(method="bfill")

    # --------------------------
    #  Forecast
   
    forecast = model.predict(future)

    forecast["Service_Type"] = service

    all_forecasts.append(forecast)

    # Plot forecast
    model.plot(forecast)
    plt.title(f"Forecast for {service}")
    plt.show()

# ------------------------------
#  Combine All Forecasts

final_forecast = pd.concat(all_forecasts)

future_30 = final_forecast.tail(30 * len(services))

print("\nNext 30 Day Forecast")
print(future_30[["ds","Service_Type","yhat","yhat_lower","yhat_upper"]])

# ------------------------------
#  Save Forecast

future_30.to_csv("azure_prophet_final_forecast.csv", index=False)

print("\nForecast saved as azure_prophet_final_forecast.csv")

print("RMSE:", rmse)