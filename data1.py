import pandas as pd
import numpy as np
import statistics as st 
import matplotlib.pyplot as plt
df=pd.read_csv("azure_dataset_3_service_types.csv")
df["Usage_Hours"] = df["Usage_Hours"].interpolate()
df["Azure_Demand"] = df["Azure_Demand"].interpolate()
print(df)
print(st.mean(df["Azure_Demand"]))
print(st.median(df["Usage_Hours"]))
df.plot("Timestamp","Usage_Hours")
plt.show()