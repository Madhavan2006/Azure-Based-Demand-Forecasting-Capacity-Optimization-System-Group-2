import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =========================
#  Load Dataset
# =========================
df = pd.read_csv("azure_dataset_3_service_types.csv")

# Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.sort_values("Timestamp")

# =========================
#  Handle Missing Values
# =========================
df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()

# =========================
# Feature Engineering
# =========================

# Rolling Mean (Trend)
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

# Remove NaN
df = df.fillna(0)

# =========================
#  Define Features & Target
# =========================
features = [
    "Usage_Hours", "Usage_7D_Avg", "Usage_Growth",
    "Day_of_Week", "Month", "Is_Weekend",
    "Usage_Spike", "Lag_1", "Lag_7"
]

X = df[features]
y = df["Azure_Demand"]

# =========================
#  Time-Based Train-Test Split
# =========================
split_index = int(len(df) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# =========================
#  Train XGBoost Model
# =========================
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
#  Predictions & Evaluation
# =========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# =========================
#  Feature Importance
# =========================
importance = pd.Series(model.feature_importances_, index=features)
importance.sort_values().plot(kind="barh")
plt.title("Feature Importance")
plt.show()

# =========================
#  Actual vs Predicted Plot
# =========================
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Demand")
plt.show()