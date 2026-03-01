import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("azure_dataset_3_service_types.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort by time
df = df.sort_values("Timestamp")

# =========================
# 1️⃣ Handle Missing Values
# =========================
df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()

# =========================
# 2️⃣ Demand Driving Features
# =========================

# Rolling mean (7 days trend)
df["Usage_7D_Avg"] = df["Usage_Hours"].rolling(7).mean()

# Growth rate
df["Usage_Growth"] = df["Usage_Hours"].pct_change()

# =========================
# 3️⃣ Derived Features
# =========================

# Seasonality Features
df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
df["Month"] = df["Timestamp"].dt.month
df["Is_Weekend"] = df["Day_of_Week"].isin([5,6]).astype(int)

# Usage spike flag
df["Usage_Spike"] = (df["Usage_Hours"] > df["Usage_7D_Avg"] * 1.5).astype(int)

# Lag Features
df["Lag_1"] = df["Usage_Hours"].shift(1)
df["Lag_7"] = df["Usage_Hours"].shift(7)

# =========================
# 4️⃣ Remove Remaining NaN
# =========================
df = df.fillna(0)

# =========================
# 5️⃣ Statistics
# =========================
print("Mean Demand:", st.mean(df["Azure_Demand"]))
print("Median Usage:", st.median(df["Usage_Hours"]))

# =========================
# 6️⃣ Plot
# =========================
df.plot(x="Timestamp", y="Usage_Hours")
plt.title("Usage Trend")
plt.show()

# =========================
# 7️⃣ Save Model Ready File
# =========================
df.to_csv("azure_model_ready_dataset.csv", index=False)

print("✅ Dataset Prepared for Modeling")
print(df.head())