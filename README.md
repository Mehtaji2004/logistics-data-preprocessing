# Logistics Data Collection, Cleaning, and Preprocessing Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](test_pipeline.py)
[![Code Style](https://img.shields.io/badge/code%20style-pep8-blueviolet.svg)](https://pep8.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Yuva Intern - Week 2 Task**: Comprehensive, reproducible end-to-end data engineering pipeline for enterprise logistics and multimodal supply chain analytics.

---

## 1. Project Overview & Reference Benchmark

Data-driven logistics operations—including route optimization, carrier selection, freight cost budgeting, and inventory safety stock planning—depend directly on high-integrity data. 

### Reference Logistics Dataset
To simulate realistic enterprise transport conditions, this project benchmarks its schema, multimodal features, and distribution metrics against publicly available logistics datasets:
* **Primary Public Reference:** [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (Mendeley Data / Kaggle), covering multimodal shipment durations, carrier transport modes, delivery statuses, and warehouse stock levels.
* **Operational Geographic Context:** Inter-city freight transport corridors connecting major Indian logistics hubs (Delhi, Mumbai, Chennai, Kolkata, Bengaluru, Hyderabad, Pune, Ahmedabad).

---

## 2. Pipeline Architecture & Execution Flow

The pipeline executes a strictly ordered sequence of transformations to ensure that intermediate data corruptions do not distort downstream imputation or scaling:

```
[ Raw Logistics Feed ] (1,035 records with injected real-world anomalies)
         │
         ▼
[ Step 1: Structural Inspection & Profiling ] (Profile nulls, dtypes, summary stats)
         │
         ▼
[ Step 2: Deduplication ] (Drop 35 duplicate rows -> 1,000 unique records)
         │
         ▼
[ Step 3: Categorical Label Standardization ] (Whitespace trimming, Title Casing: 16 -> 8 cities)
         │
         ▼
[ Step 4: Domain Constraint Remediation ] (Convert negative distances < 0 to NaN)
         │
         ▼
[ Step 5: Robust Outlier Detection (IQR Rule) ] (Flag impossible transit hours & extreme costs)
         │
         ▼
[ Step 6: Principled Imputation ] (Median for skewed numeric metrics, Mode for transport mode)
         │
         ▼
[ Step 7: Dual-Method Feature Normalization ] (Min-Max Scaling [0, 1] & Z-Score Standardization)
         │
         ▼
[ Step 8: Downstream Business Metrics & Visualizations ] (Export 'clean_logistics_data.csv', 'results.json', & EDA charts)
```

---

## 3. Detailed Data Cleaning Methodology

| Step | Issue Identified | Cleaning Technique Applied | Methodological Justification |
|---|---|---|---|
| **1. Deduplication** | 35 duplicate rows (3.4% of raw records). | `df.drop_duplicates()` | Prevents artificial inflation of freight volumes and prevents double-counted records from biasing imputation medians. |
| **2. Categorical Standardization** | Inconsistent casing (`DELHI`, `delhi `) and trailing spaces. | `.str.strip().str.title()` | Consolidates fragmented categories. Prevents routing and grouping algorithms from splitting identical cities into separate buckets. |
| **3. Domain Constraints** | 6 negative travel distances (`distance_km < 0`). | `df.loc[df['distance_km'] < 0, 'distance_km'] = np.nan` | Distance is non-negative by physical laws; negative values stem from sensor or sign-inversion faults. Converting to `NaN` allows systematic imputation. |
| **4. Outlier Handling** | Extreme delivery times (up to 2,500 hrs) and costs (> Rs 60,000). | Non-parametric **IQR Rule** ($[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$) | The IQR method is chosen over Z-score for outlier identification because Z-score relies on the mean and standard deviation, which are themselves severely distorted by extreme values. Flagged points are masked as `NaN`. |
| **5. Missing Value Imputation** | Null entries across continuous and categorical fields. | **Median Imputation** for continuous skewed features; **Mode Imputation** for categorical (`transport_mode`). | Median is resistant to right-tail skewness and residual extremes. Mode correctly assigns the most frequent operational modality (Road). |
| **6. Normalization** | Features measured in vastly different units (hours, kg, km, INR). | **Min-Max Scaling** ($x \in [0, 1]$) and **Z-Score Standardization** ($\mu=0, \sigma=1$). | Eliminates scale dominance so distance-based (KNN, K-Means) and gradient-based machine learning models treat all logistics features with balanced weighting. |

---

## 4. Key Quantitative Results

| Metric / Attribute | Raw Dataset | Cleaned Dataset | Impact / Improvement |
|---|---|---|---|
| **Total Record Count** | 1,035 rows | 1,000 rows | 35 duplicates completely eliminated |
| **Missing Values (Total)** | 338 nulls across 6 columns | **0** nulls | 100% data completeness achieved |
| **Mean Shipping Duration** | 44.27 hours | **21.23 hours** | -52.0% distortion removed (extreme 2,500h outliers eliminated) |
| **Std Dev Shipping Duration**| 209.91 hours | **8.83 hours** | Drastic variance reduction reflecting realistic freight operations |
| **Mean Freight Cost** | ₹2,104.71 | **₹1,272.27** | -39.5% reduction in artificial budgetary inflation |
| **Mean Cargo Weight** | 1,619.61 kg | **1,266.14 kg** | 38-tonne entry errors removed |
| **Unique Origin Labels** | 16 variants | **8 standard cities** | Standardized to proper geographical entities |
| **Minimum Distance** | -918.1 km (invalid) | **60.0 km** (valid) | Domain boundary integrity enforced |

### Average Cleaned Freight Cost by Transport Mode
* **Road:** ₹1,257.38
* **Rail:** ₹1,279.38
* **Air:** ₹1,288.94
* **Sea:** ₹1,350.39

---

## 5. Reflection: Impact on Logistics Decision-Making

1. **Safety Stock & Capital Efficiency:** In logistics lead-time planning, using the raw uncleaned mean transit time (44.3 hours) instead of the true cleaned operational mean (21.2 hours) would lead inventory managers to hold more than double the required safety buffer, tying up working capital unnecessarily.
2. **Freight Budgeting & Contracting:** Logistics budget forecasts derived from the uncleaned dataset would overestimate annual freight expenses by nearly 40% (₹2,104 vs ₹1,272 per shipment), creating severe misallocations.
3. **Routing & Network Analytics:** Categorical noise (e.g., splitting Delhi into `DELHI`, `delhi `, and `Delhi`) prevents business intelligence systems from aggregating lane-level density and negotiating volume discounts with transport carriers.

---

## 6. How to Run and Test the Project

### Environment Setup
The codebase is fully compatible with **Python 3.8 through 3.13+**:

```bash
# 1. Clone repository (if needed)
git clone https://github.com/Mehtaji2004/logistics-data-preprocessing.git
cd logistics-data-preprocessing

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
# Step 1: Simulate the raw logistics dataset
python generate_data.py

# Step 2: Run the full cleaning, preprocessing, and normalization pipeline
python preprocess.py

# Step 3: Run the automated unit test suite
python -m unittest test_pipeline.py -v
```

### CLI Options
Both scripts support modular CLI arguments:
```bash
python generate_data.py --samples 1000 --seed 42 --output raw_logistics_data.csv
python preprocess.py --input raw_logistics_data.csv --output clean_logistics_data.csv --results results.json
```

---

## 7. Project File Structure

```
├── README.md                                    # Comprehensive project documentation
├── requirements.txt                             # Production dependencies (pandas, numpy, matplotlib)
├── generate_data.py                             # Modular simulation script referencing DataCo benchmarks
├── preprocess.py                                # Modular data cleaning and preprocessing pipeline
├── test_pipeline.py                             # Automated unit test suite (8 test cases, zero dependencies)
├── raw_logistics_data.csv                       # Generated raw shipment records (1,035 rows)
├── clean_logistics_data.csv                     # Cleaned, imputed, and normalized dataset (1,000 rows, 21 cols)
├── results.json                                 # Quantitative audit trail of all before/after pipeline metrics
├── Week2_Logistics_Data_Preprocessing_Report.docx# Submitted formal project report
├── fig_missing_before.png                       # EDA Plot: Raw missing values per column
├── fig_outliers_before.png                      # EDA Plot: Boxplots showing raw outliers
└── fig_hours_before_after.png                   # EDA Plot: shipping_hours distribution before vs after
```

---

## 8. Author & Internship Information
* **Author:** Akshit Mehta
* **Task:** Yuva Intern - Week 2 Data Preprocessing for Logistics Analysis
* **GitHub Repository:** [logistics-data-preprocessing](https://github.com/Mehtaji2004/logistics-data-preprocessing)
