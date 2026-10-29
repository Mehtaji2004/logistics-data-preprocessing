# Logistics Data Collection, Cleaning and Preprocessing

Week 2 internship task: a reproducible Python pipeline that simulates a raw logistics shipment dataset with realistic quality problems and then cleans and preprocesses it.

## What it does
1. **Simulates** 1,035 shipment records (8 Indian cities, 4 transport modes) with injected defects: missing values, 35 duplicates, inconsistent labels, outliers and negative distances (`generate_data.py`).
2. **Cleans** the data (`preprocess.py`): duplicate removal, label standardisation, invalid-value fixing, IQR outlier detection, median/mode imputation, min-max and z-score normalisation.
3. **Reports** measured before/after results in `results.json` and three charts.

## Key results
| Metric | Raw | Cleaned |
|---|---|---|
| Rows | 1,035 | 1,000 |
| Missing values | 338 | 0 |
| Mean shipping hours | 44.27 | 21.23 |
| Mean shipping cost (Rs) | 2,104.71 | 1,272.27 |
| Mean weight (kg) | 1,619.61 | 1,266.14 |
| Distinct origin labels | 16 | 8 |

## Run it
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python generate_data.py
python preprocess.py
```

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Creates `raw_logistics_data.csv` |
| `preprocess.py` | Cleaning pipeline, creates `clean_logistics_data.csv`, `results.json`, figures |
| `Week2_Logistics_Data_Preprocessing_Report.docx` | Submitted report |

Libraries: pandas, NumPy, Matplotlib.
