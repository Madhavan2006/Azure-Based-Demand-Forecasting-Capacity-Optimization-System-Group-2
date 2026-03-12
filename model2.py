import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ==============================
# 1️⃣ Load Dataset
# ==============================
df = pd.read_csv("azure_dataset_3_service_types.csv")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

# ==============================
# 2️⃣ Handle Missing Values
# ==============================
df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()

# ==============================
# 3️⃣ Feature Engineering
# ==============================

# Rolling Trend
df["Usage_7D_Avg"] = df["Usage_Hours"].rolling(7).mean()

# Growth Rate
df["Usage_Growth"] = df["Usage_Hours"].pct_change()

# Seasonality
df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
df["Month"] = df["Timestamp"].dt.month
df["Is_Weekend"] = df["Day_of_Week"].isin([5,6]).astype(int)

# Spike Detection
df["Usage_Spike"] = (df["Usage_Hours"] > df["Usage_7D_Avg"] * 1.5).astype(int)

# Lag Features
df["Lag_1"] = df["Usage_Hours"].shift(1)
df["Lag_7"] = df["Usage_Hours"].shift(7)

# Fill remaining NaN
df = df.fillna(0)

# ==============================
# 4️⃣ Define Features & Target
# ==============================
features = [
    "Usage_Hours", "Usage_7D_Avg", "Usage_Growth",
    "Day_of_Week", "Month", "Is_Weekend",
    "Usage_Spike", "Lag_1", "Lag_7"
]

X = df[features]
y = df["Azure_Demand"]

# ==============================
# 5️⃣ Time-Based Train-Test Split
# ==============================
split_index = int(len(df) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ==============================
# 6️⃣ Train XGBoost Model
# ==============================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7️⃣ Model Evaluation
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance")
print("MAE :", mae)
print("RMSE:", rmse)

# ==============================
# 8️⃣ Feature Importance
# ==============================
importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.show()

# ==============================
# 9️⃣ 30-Day Forecast
# ==============================

future_days = 30
last_date = df["Timestamp"].max()
future_predictions = []
future_df = df.copy()

for i in range(future_days):

    next_date = last_date + pd.Timedelta(days=1)

    new_row = {}

    new_row["Timestamp"] = next_date
    new_row["Usage_Hours"] = future_df["Usage_Hours"].iloc[-1]

    new_row["Usage_7D_Avg"] = future_df["Usage_Hours"].tail(7).mean()
    new_row["Usage_Growth"] = future_df["Usage_Hours"].pct_change().iloc[-1]

    new_row["Day_of_Week"] = next_date.dayofweek
    new_row["Month"] = next_date.month
    new_row["Is_Weekend"] = 1 if next_date.dayofweek in [5,6] else 0

    new_row["Usage_Spike"] = 0
    new_row["Lag_1"] = future_df["Usage_Hours"].iloc[-1]
    new_row["Lag_7"] = future_df["Usage_Hours"].iloc[-7] if len(future_df) >= 7 else 0

    new_df = pd.DataFrame([new_row])
    X_future = new_df[features]

    prediction = model.predict(X_future)[0]

    new_row["Azure_Demand"] = prediction
    future_predictions.append(new_row)

    future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
    last_date = next_date

forecast_df = pd.DataFrame(future_predictions)

print("\nNext 30 Days Forecast")
print(forecast_df[["Timestamp", "Azure_Demand"]])

# ==============================
# 🔟 Plot Historical + Forecast
# ==============================
plt.figure(figsize=(12,6))
plt.plot(df["Timestamp"], df["Azure_Demand"], label="Historical")
plt.plot(forecast_df["Timestamp"], forecast_df["Azure_Demand"], label="Forecast")
plt.legend()
plt.title("Azure Demand - 30 Day Forecast")
plt.show()

# ==============================
# 1️⃣1️⃣ Save Forecast
# ==============================
forecast_df.to_csv("azure_30_day_forecast.csv", index=False)

print("\n✅ Forecast saved as azure_30_day_forecast.csv")