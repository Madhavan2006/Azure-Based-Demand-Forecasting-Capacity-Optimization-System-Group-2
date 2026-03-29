# Azure Demand Forecasting — FastAPI

## Project Structure

```
fastapi_app/
├── main.py                        # FastAPI app entry point
├── requirements.txt
├── forecast_plots/                # auto-created; PNG plots saved here
└── app/
    ├── core/
    │   ├── schemas.py             # Pydantic request / response models
    │   ├── model_store.py         # In-memory singleton for trained artefacts
    │   ├── ml_engine.py           # Feature engineering, training, forecast logic
    │   └── plotter.py             # All matplotlib plotting code
    └── routers/
        ├── health.py              # GET  /
        ├── train.py               # POST /api/v1/train
        ├── forecast.py            # POST /api/v1/forecast  + helpers
        └── plots.py               # POST /api/v1/plots/generate + helpers
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset in the project root (or supply full path at train time)
cp azure_dataset_3_service_types.csv fastapi_app/

# 3. Start the server
cd fastapi_app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

---

## API Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check — shows whether a model is loaded |

### Training
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/train` | Train the full ensemble model |
| GET  | `/api/v1/train/status` | Check if a background train is running |

**Train request body:**
```json
{
  "csv_path": "azure_dataset_3_service_types.csv",
  "test_days": 30,
  "forecast_days": 30
}
```

### Forecast
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/forecast` | Run / re-run forecast, get JSON |
| GET  | `/api/v1/forecast/series` | List all (service, region) combinations |
| GET  | `/api/v1/forecast/download` | Download forecast as CSV |
| GET  | `/api/v1/forecast/metrics` | Get MAE / RMSE metrics |

**Forecast request body:**
```json
{
  "service_type": "Virtual Machines",
  "region": "East US",
  "forecast_days": 30
}
```
Omit `service_type` / `region` to get all series.

### Plots
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/plots/generate` | Generate all PNG plots |
| GET  | `/api/v1/plots/list` | List saved plot files |
| GET  | `/api/v1/plots/view/{filename}` | View a single plot image |
| GET  | `/plots/{filename}` | Static file access (served by Starlette) |

**Plot request body:**
```json
{
  "service_type": null,
  "region": null,
  "lookback_days": 90
}
```

---

## Typical Workflow

```bash
# 1. Train
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "azure_dataset_3_service_types.csv"}'

# 2. Forecast (all series)
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 30}'

# 3. Generate plots
curl -X POST http://localhost:8000/api/v1/plots/generate \
  -H "Content-Type: application/json" \
  -d '{"lookback_days": 90}'

# 4. Download CSV
curl http://localhost:8000/api/v1/forecast/download -o forecast.csv

# 5. View a plot
open http://localhost:8000/plots/all_series_comparison.png
```
