import pandas as pd
import numpy as np
import statistics as st 
import matplotlib.pyplot as plt
df=pd.read_csv("azure_dataset_3_service_types.csv")
df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()
df["Market_Demand_Trend"] = df["Market_Demand_Trend"].interpolate()
df["Cost_USD"] = df["Cost_USD"].interpolate()
print(df)
me=st.mean(df["Market_Demand_Trend"])
md=st.median(df["Cost_USD"])
print(me,md)
df.plot("Timestamp","Usage_Hours")
plt.show()