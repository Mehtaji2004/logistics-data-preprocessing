"""Logistics Data Cleaning, Preprocessing, and Quality Verification Pipeline.

Methodological Architecture & Pipeline Design:
----------------------------------------------
The preprocessing pipeline is engineered with a strict, theoretically grounded execution order:

  Step 1: Data Ingestion & Structural Inspection
          - Validate schema, row/column dimensions, null value distribution, and summary statistics.
  Step 2: Deduplication
          - Eliminate duplicate records early to prevent repeated records from artificially
            skewing sample variances, summary metrics, and subsequent imputation values.
  Step 3: Categorical Label Standardization
          - Strip leading/trailing whitespaces and standardize text to Title Case.
            Prevents fragmentation of transport modes and metropolitan lanes in downstream grouping.
  Step 4: Domain Constraint & Invalid Value Remediation
          - Flag impossible physical values (e.g., negative transit distance) and convert to NaN.
  Step 5: Robust Outlier Detection via Interquartile Range (IQR) Rule
          - Formulate outlier boundaries: [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
          - IQR is selected over Z-score for outlier identification because Z-score relies on
            the mean and standard deviation, which are themselves corrupted by severe outliers.
          - Flagged anomalous measurements are masked as NaN for systematic imputation.
  Step 6: Missing Value Imputation
          - Continuous numeric features: Imputed using median (robust against distributional skew).
          - Categorical features: Imputed using mode (most frequent operational class).
  Step 7: Dual-Method Feature Normalization
          - Min-Max Scaling (rescaling to [0, 1]): Preserves relative distances for bounded models.
          - Z-Score Standardization (zero mean, unit variance): Eliminates scale bias for gradient-based
            and distance-based machine learning algorithms.
  Step 8: Quality Metrics Export & Visualization
          - Track before-and-after metrics in 'results.json'.
          - Render high-resolution EDA figures (distributions, box plots, missingness charts).

Author: Akshit Mehta (Yuva Intern Week 2 Logistics Analytics Task)
Compatible with: Python 3.8+, pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# Configure non-interactive Matplotlib backend and safe cache directory
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS: List[str] = [
    "distance_km",
    "weight_kg",
    "shipping_hours",
    "shipping_cost_inr",
    "inventory_units",
]

CATEGORICAL_COLUMNS: List[str] = [
    "origin",
    "destination",
    "transport_mode",
]


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load shipment records from CSV with parsed datetime columns."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Input logistics dataset not found at: '{path}'")
    return pd.read_csv(path, parse_dates=["ship_date"])


def inspect_data(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    """Inspect dataset shape, missing value counts, and descriptive statistics."""
    return {
        "shape": list(df.shape),
        "missing": df.isna().sum().to_dict(),
        "describe": df[num_cols].describe().round(2).to_dict(),
    }


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Identify and remove duplicated shipment records.

    Duplicate entries represent accidental ETL re-ingestion, double-counting shipment
    volumes and distorting operational capacity planning.
    """
    n_dup = int(df.duplicated().sum())
    cleaned_df = df.drop_duplicates().reset_index(drop=True)
    return cleaned_df, n_dup


def standardize_categorical_labels(
    df: pd.DataFrame, cat_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, List[str]]]:
    """Standardize categorical string fields by stripping whitespace and applying Title Case.

    Prevents casing fragmentation (e.g. 'DELHI' vs 'Delhi' vs 'delhi ') from creating
    spurious operational categories in routing algorithms.
    """
    df = df.copy()
    labels_before = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in ["origin", "transport_mode"]
    }

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            # If any string was 'Nan', keep it as true np.nan
            df[col] = df[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan})

    labels_after = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in ["origin", "transport_mode"]
    }
    return df, labels_before, labels_after


def handle_invalid_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Identify domain constraint violations and convert them to NaN for principled imputation.

    In logistics physics, travel distance cannot be strictly negative (< 0).
    Negative distance indicates sign-inversion sensor fault or manual entry error.
    """
    df = df.copy()
    n_neg = int((df["distance_km"] < 0).sum())
    df.loc[df["distance_km"] < 0, "distance_km"] = np.nan
    return df, n_neg


def detect_and_handle_outliers_iqr(
    df: pd.DataFrame,
    target_cols: List[str],
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, List[float]]]:
    """Detect extreme anomalous values using the non-parametric Interquartile Range (IQR) rule.

    Calculates:
      IQR = Q3 - Q1
      Lower Bound = Q1 - (factor * IQR)
      Upper Bound = Q3 + (factor * IQR)

    Outlier observations are converted to NaN so they can be imputed systematically
    without distorting underlying distribution parameters.
    """
    df = df.copy()
    out_counts: Dict[str, int] = {}
    bounds: Dict[str, List[float]] = {}

    for col in target_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = float(q1 - factor * iqr)
        hi = float(q3 + factor * iqr)
        mask = (df[col] < lo) | (df[col] > hi)

        out_counts[col] = int(mask.sum())
        bounds[col] = [round(lo, 2), round(hi, 2)]
        df.loc[mask, col] = np.nan

    return df, out_counts, bounds


def impute_missing_values(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    """Impute missing data using robust statistics.

    - Continuous skewed numeric attributes are imputed via the Median.
    - Categorical attributes are imputed via Mode (most frequent category).
    """
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        if col in df.columns:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series[0])

    return df


def normalize_features(
    df: pd.DataFrame,
    num_cols: List[str],
) -> pd.DataFrame:
    """Apply both Min-Max Scaling and Z-Score Standardization to numeric features.

    Min-Max Scaling:
      x' = (x - min) / (max - min), maps strictly to [0, 1]
    Z-Score Standardization:
      z = (x - mu) / sigma, centers distribution at 0 with unit variance
    """
    df = df.copy()
    for col in num_cols:
        mn = df[col].min()
        mx = df[col].max()
        col_std = df[col].std()
        col_mean = df[col].mean()

        # Min-Max Scaling
        if mx > mn:
            df[f"{col}_minmax"] = ((df[col] - mn) / (mx - mn)).round(4)
        else:
            df[f"{col}_minmax"] = 0.0

        # Z-Score Standardization
        if col_std > 0:
            df[f"{col}_zscore"] = ((df[col] - col_mean) / col_std).round(4)
        else:
            df[f"{col}_zscore"] = 0.0

    return df


def compute_downstream_metrics(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute before-and-after business impact metrics for logistics operations."""
    return {
        "mean_cost_raw": round(float(raw_df["shipping_cost_inr"].mean()), 2),
        "mean_cost_clean": round(float(clean_df["shipping_cost_inr"].mean()), 2),
        "mean_hours_raw": round(float(raw_df["shipping_hours"].mean()), 2),
        "mean_hours_clean": round(float(clean_df["shipping_hours"].mean()), 2),
        "mean_weight_raw": round(float(raw_df["weight_kg"].mean()), 2),
        "mean_weight_clean": round(float(clean_df["weight_kg"].mean()), 2),
        "cost_by_mode_clean": (
            clean_df.groupby("transport_mode")["shipping_cost_inr"]
            .mean()
            .round(2)
            .to_dict()
        ),
    }


def generate_visualizations(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    results: Dict[str, Any],
    output_dir: str | Path = ".",
) -> None:
    """Generate high-resolution EDA figures illustrating cleaning impacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Box plots of raw data showing outliers (fully forward-compatible Matplotlib)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col in zip(axes, ["shipping_hours", "weight_kg", "shipping_cost_inr"]):
        # Default boxplot is vertical; avoids deprecated vert=True argument
        ax.boxplot(raw_df[col].dropna())
        ax.set_title(col)
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Raw Data: Outliers Visible in Box Plots")
    plt.tight_layout()
    plt.savefig(out_path / "fig_outliers_before.png", dpi=150)
    plt.close()

    # 2. Missing values per column in raw data
    fig, ax = plt.subplots(figsize=(7, 3.6))
    missing_series = pd.Series(results["missing_before"])
    missing_series = missing_series[missing_series > 0]
    ax.bar(missing_series.index, missing_series.values, color="#d9622b")
    ax.set_title("Missing Values per Column (Raw Dataset)")
    ax.set_ylabel("Count of Missing Records")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path / "fig_missing_before.png", dpi=150)
    plt.close()

    # 3. Before vs After Distribution (shipping_hours)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(raw_df["shipping_hours"].dropna(), bins=40, color="#d9622b")
    axes[0].set_title("shipping_hours BEFORE Cleaning (Skewed by Outliers)")
    axes[0].set_xlabel("Hours")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(clean_df["shipping_hours"], bins=40, color="#2b8ad9")
    axes[1].set_title("shipping_hours AFTER Cleaning (Operational Distribution)")
    axes[1].set_xlabel("Hours")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path / "fig_hours_before_after.png", dpi=150)
    plt.close()


def run_pipeline(
    input_path: str | Path = "raw_logistics_data.csv",
    output_path: str | Path = "clean_logistics_data.csv",
    results_path: str | Path = "results.json",
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute the complete end-to-end data cleaning and preprocessing pipeline."""
    results: Dict[str, Any] = {}

    # Step 1: Load and Inspect Raw Data
    df_raw = load_data(input_path)
    inspection_raw = inspect_data(df_raw, NUMERIC_COLUMNS)
    results["raw_shape"] = inspection_raw["shape"]
    results["missing_before"] = inspection_raw["missing"]
    results["describe_before"] = inspection_raw["describe"]

    # Step 2: Remove Duplicate Records
    df_clean, n_dup = remove_duplicates(df_raw)
    results["duplicates_removed"] = n_dup

    # Step 3: Standardize Categorical Labels
    df_clean, labels_before, labels_after = standardize_categorical_labels(
        df_clean, CATEGORICAL_COLUMNS
    )
    results["labels_before"] = labels_before
    results["labels_after"] = labels_after

    # Step 4: Fix Invalid Domain Values
    df_clean, n_neg = handle_invalid_values(df_clean)
    results["negative_distance_fixed"] = n_neg

    # Step 5: Detect Outliers via IQR Rule
    outlier_target_cols = ["shipping_hours", "weight_kg", "shipping_cost_inr"]
    df_clean, out_counts, bounds = detect_and_handle_outliers_iqr(
        df_clean, outlier_target_cols
    )
    results["outliers_flagged"] = out_counts
    results["iqr_bounds"] = bounds
    results["missing_after_outlier_step"] = df_clean.isna().sum().to_dict()

    # Step 6: Impute Missing Values
    df_clean = impute_missing_values(df_clean, NUMERIC_COLUMNS, ["transport_mode"])
    results["missing_after"] = df_clean.isna().sum().to_dict()

    # Step 7: Feature Normalization (Min-Max & Z-Score)
    df_clean = normalize_features(df_clean, NUMERIC_COLUMNS)
    results["clean_shape"] = list(df_clean.shape)
    results["describe_after"] = df_clean[NUMERIC_COLUMNS].describe().round(2).to_dict()

    # Worked example row for verification and reporting
    example_row = df_clean.iloc[0]
    results["worked_example"] = {
        "shipment_id": example_row["shipment_id"],
        "weight_kg": float(example_row["weight_kg"]),
        "weight_min": float(df_clean["weight_kg"].min()),
        "weight_max": float(df_clean["weight_kg"].max()),
        "weight_mean": round(float(df_clean["weight_kg"].mean()), 2),
        "weight_std": round(float(df_clean["weight_kg"].std()), 2),
        "weight_minmax": float(example_row["weight_kg_minmax"]),
        "weight_zscore": float(example_row["weight_kg_zscore"]),
    }

    # Step 8: Downstream Business Metrics & Visualizations
    impact_metrics = compute_downstream_metrics(df_raw, df_clean)
    results.update(impact_metrics)

    if generate_plots:
        generate_visualizations(df_raw, df_clean, results, Path(output_path).parent)

    # Persist Clean Dataset and Results Dictionary
    clean_csv_path = Path(output_path)
    df_clean.to_csv(clean_csv_path, index=False)

    results_json_path = Path(results_path)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Pipeline] Successfully processed data -> '{clean_csv_path}'. Shape: {df_clean.shape}")
    print(f"[Pipeline] Results metrics saved to '{results_json_path}'.")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Clean, preprocess, and normalize logistics shipment dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="raw_logistics_data.csv",
        help="Input raw CSV dataset path (default: raw_logistics_data.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clean_logistics_data.csv",
        help="Output clean CSV dataset path (default: clean_logistics_data.csv).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Output results JSON path (default: results.json).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating EDA plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(
        input_path=args.input,
        output_path=args.output,
        results_path=args.results,
        generate_plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ["duplicates_removed", "negative_distance_fixed", "outliers_flagged"]
            },
            indent=2,
        )
    )
