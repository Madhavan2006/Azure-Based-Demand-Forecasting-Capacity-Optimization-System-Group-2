# Azure-Based-Demand-Forecasting-Capacity-Optimization-System-Group-2

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)

## 📝 Description

The Azure-Based Demand Forecasting and Capacity Optimization System is a robust analytical solution designed to streamline resource management and operational efficiency. Developed using Python, this system leverages advanced data processing techniques to predict future demand patterns with high accuracy, enabling organizations to optimize their capacity planning on the Microsoft Azure cloud platform. By integrating sophisticated forecasting models, the project helps businesses minimize waste and align their resources precisely with market needs. The system features a comprehensive testing suite to ensure model reliability and data integrity, providing a scalable and dependable framework for data-driven decision-making.

## ✨ Features

- 🧪 Testing


## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
streamlit: latest
pandas: latest
numpy: latest
plotly: latest
scikit-learn: latest
xgboost: latest
lightgbm: latest
catboost: latest
```

## 📁 Project Structure

```
.
├── .streamlit
│   └── config.toml
├── Agile__Madhavan P.xlsx
├── Defect_tracker_Azure_Demand_Forecasting__Madhavan P.xlsx
├── MIT license.txt
├── Unit_Test_Plan_Azure_Demand_Forecasting__Madhavan P.xlsx
├── azure_30_day_forecast.csv
├── azure_dataset_3_service_types.csv
├── azure_prophet_final_forecast.csv
├── catboost_info
│   ├── catboost_training.json
│   ├── learn
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   ├── test
│   │   └── events.out.tfevents
│   ├── test_error.tsv
│   └── time_left.tsv
├── dashboard.py
├── forecast_plots
│   ├── AI_East_dow_seasonality.png
│   ├── AI_East_overview.png
│   ├── AI_East_weekly_agg.png
│   ├── AI_North_dow_seasonality.png
│   ├── AI_North_overview.png
│   ├── AI_North_weekly_agg.png
│   ├── AI_South_dow_seasonality.png
│   ├── AI_South_overview.png
│   ├── AI_South_weekly_agg.png
│   ├── AI_West_dow_seasonality.png
│   ├── AI_West_overview.png
│   ├── AI_West_weekly_agg.png
│   ├── Compute_East_dow_seasonality.png
│   ├── Compute_East_overview.png
│   ├── Compute_East_weekly_agg.png
│   ├── Compute_North_dow_seasonality.png
│   ├── Compute_North_overview.png
│   ├── Compute_North_weekly_agg.png
│   ├── Compute_South_dow_seasonality.png
│   ├── Compute_South_overview.png
│   ├── Compute_South_weekly_agg.png
│   ├── Compute_West_dow_seasonality.png
│   ├── Compute_West_overview.png
│   ├── Compute_West_weekly_agg.png
│   ├── Storage_East_dow_seasonality.png
│   ├── Storage_East_overview.png
│   ├── Storage_East_weekly_agg.png
│   ├── Storage_North_dow_seasonality.png
│   ├── Storage_North_overview.png
│   ├── Storage_North_weekly_agg.png
│   ├── Storage_South_dow_seasonality.png
│   ├── Storage_South_overview.png
│   ├── Storage_South_weekly_agg.png
│   ├── Storage_West_dow_seasonality.png
│   ├── Storage_West_overview.png
│   ├── Storage_West_weekly_agg.png
│   └── all_series_comparison.png
├── model.py
├── model2.py
├── model_artifacts
│   ├── features.pkl
│   ├── final_cb.pkl
│   ├── final_lgb.pkl
│   ├── final_xgb.pkl
│   ├── le_region.pkl
│   ├── le_service.pkl
│   ├── meta_ridge.pkl
│   ├── metrics.pkl
│   └── res_model.pkl
├── requirements.txt
└── testing
    ├── app.py
    ├── azure_30_day_forecast.csv
    ├── azure_dataset_3_service_types.csv
    ├── catboost_info
    │   ├── catboost_training.json
    │   ├── learn
    │   │   └── events.out.tfevents
    │   ├── learn_error.tsv
    │   ├── test
    │   │   └── events.out.tfevents
    │   ├── test_error.tsv
    │   └── time_left.tsv
    ├── forecast_plots
    │   ├── AI_East_dow_seasonality.png
    │   ├── AI_East_overview.png
    │   ├── AI_East_weekly_agg.png
    │   ├── AI_North_dow_seasonality.png
    │   ├── AI_North_overview.png
    │   ├── AI_North_weekly_agg.png
    │   ├── AI_South_dow_seasonality.png
    │   ├── AI_South_overview.png
    │   ├── AI_South_weekly_agg.png
    │   ├── AI_West_dow_seasonality.png
    │   ├── AI_West_overview.png
    │   ├── AI_West_weekly_agg.png
    │   ├── Compute_East_dow_seasonality.png
    │   ├── Compute_East_overview.png
    │   ├── Compute_East_weekly_agg.png
    │   ├── Compute_North_dow_seasonality.png
    │   ├── Compute_North_overview.png
    │   ├── Compute_North_weekly_agg.png
    │   ├── Compute_South_dow_seasonality.png
    │   ├── Compute_South_overview.png
    │   ├── Compute_South_weekly_agg.png
    │   ├── Compute_West_dow_seasonality.png
    │   ├── Compute_West_overview.png
    │   ├── Compute_West_weekly_agg.png
    │   ├── Storage_East_dow_seasonality.png
    │   ├── Storage_East_overview.png
    │   ├── Storage_East_weekly_agg.png
    │   ├── Storage_North_dow_seasonality.png
    │   ├── Storage_North_overview.png
    │   ├── Storage_North_weekly_agg.png
    │   ├── Storage_South_dow_seasonality.png
    │   ├── Storage_South_overview.png
    │   ├── Storage_South_weekly_agg.png
    │   ├── Storage_West_dow_seasonality.png
    │   ├── Storage_West_overview.png
    │   ├── Storage_West_weekly_agg.png
    │   └── all_series_comparison.png
    └── testing.py
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/Madhavan2006/Azure-Based-Demand-Forecasting-Capacity-Optimization-System-Group-2.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request
7. **dashboard link**:`https://madhavanp007.streamlit.app/`