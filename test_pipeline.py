"""Automated Unit & Integration Test Suite for Logistics Data Preprocessing.

Tests ensure algorithmic correctness, numeric precision, and execution compatibility
across Python 3.8+ environments using the standard library `unittest` framework.

Author: Akshit Mehta (Yuva Intern Week 2 Logistics Analytics Task)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from generate_data import simulate_logistics_dataset
from preprocess import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    handle_invalid_values,
    impute_missing_values,
    detect_and_handle_outliers_iqr,
    normalize_features,
    remove_duplicates,
    run_pipeline,
    standardize_categorical_labels,
)


class TestLogisticsPreprocessingPipeline(unittest.TestCase):
    """Validation test suite for data cleaning, anomaly detection, and feature scaling."""

    @classmethod
    def setUpClass(cls) -> None:
        """Generate a test dataset fixture."""
        cls.raw_df = simulate_logistics_dataset(n_samples=1000, seed=42, output_path=None)

    def test_simulation_schema_and_defects(self) -> None:
        """Verify simulation generates correct schema and deliberate quality anomalies."""
        expected_columns = {
            "shipment_id",
            "ship_date",
            "origin",
            "destination",
            "transport_mode",
            "distance_km",
            "weight_kg",
            "shipping_hours",
            "shipping_cost_inr",
            "inventory_units",
            "status",
        }
        self.assertTrue(expected_columns.issubset(set(self.raw_df.columns)))
        self.assertEqual(len(self.raw_df), 1035)  # 1000 + 35 duplicates
        # Verify presence of injected defects
        self.assertGreater(self.raw_df.duplicated().sum(), 0)
        self.assertGreater(self.raw_df.isna().sum().sum(), 0)
        self.assertGreater((self.raw_df["distance_km"] < 0).sum(), 0)

    def test_remove_duplicates(self) -> None:
        """Verify deduplication removes exactly the injected duplicate rows."""
        deduped_df, n_dup = remove_duplicates(self.raw_df)
        self.assertEqual(n_dup, 35)
        self.assertEqual(len(deduped_df), 1000)
        self.assertEqual(int(deduped_df.duplicated().sum()), 0)

    def test_standardize_categorical_labels(self) -> None:
        """Verify label standardisation strips whitespace and applies title casing."""
        sample_df = pd.DataFrame(
            {
                "origin": ["DELHI", "delhi ", "Delhi", " Mumbai "],
                "transport_mode": ["road", "ROAD", "Road", "air"],
            }
        )
        cleaned_df, before, after = standardize_categorical_labels(
            sample_df, ["origin", "transport_mode"]
        )
        self.assertEqual(sorted(cleaned_df["origin"].tolist()), ["Delhi", "Delhi", "Delhi", "Mumbai"])
        self.assertEqual(sorted(cleaned_df["transport_mode"].tolist()), ["Air", "Road", "Road", "Road"])
        self.assertEqual(after["transport_mode"], ["Air", "Road"])

    def test_handle_invalid_values(self) -> None:
        """Verify negative physical distance values are converted to NaN."""
        sample_df = pd.DataFrame({"distance_km": [500.0, -120.5, 850.0, -45.0]})
        cleaned_df, n_neg = handle_invalid_values(sample_df)
        self.assertEqual(n_neg, 2)
        self.assertTrue(np.isnan(cleaned_df.loc[1, "distance_km"]))
        self.assertTrue(np.isnan(cleaned_df.loc[3, "distance_km"]))
        self.assertEqual(cleaned_df.loc[0, "distance_km"], 500.0)

    def test_detect_and_handle_outliers_iqr(self) -> None:
        """Verify IQR outlier detection flags values outside 1.5 * IQR bounds."""
        # 10 normal observations and 2 obvious extreme outliers
        values = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.8, 11.2, 11.8, 12.2, 500.0, -300.0]
        sample_df = pd.DataFrame({"shipping_hours": values})
        cleaned_df, out_counts, bounds = detect_and_handle_outliers_iqr(
            sample_df, ["shipping_hours"], factor=1.5
        )
        self.assertEqual(out_counts["shipping_hours"], 2)
        self.assertTrue(np.isnan(cleaned_df.loc[10, "shipping_hours"]))
        self.assertTrue(np.isnan(cleaned_df.loc[11, "shipping_hours"]))

    def test_impute_missing_values(self) -> None:
        """Verify missing values are completely imputed with median and mode."""
        sample_df = pd.DataFrame(
            {
                "distance_km": [100.0, 200.0, np.nan, 300.0],
                "transport_mode": ["Road", "Road", np.nan, "Rail"],
            }
        )
        imputed_df = impute_missing_values(sample_df, ["distance_km"], ["transport_mode"])
        self.assertEqual(imputed_df.isna().sum().sum(), 0)
        self.assertEqual(imputed_df.loc[2, "distance_km"], 200.0)  # median of [100, 200, 300]
        self.assertEqual(imputed_df.loc[2, "transport_mode"], "Road")  # mode of ['Road', 'Road', 'Rail']

    def test_normalize_features(self) -> None:
        """Verify Min-Max values are in [0, 1] and Z-Score has mean ~ 0 and std ~ 1."""
        sample_df = pd.DataFrame(
            {
                "weight_kg": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )
        norm_df = normalize_features(sample_df, ["weight_kg"])

        # Check Min-Max scaling bounds
        self.assertAlmostEqual(norm_df["weight_kg_minmax"].min(), 0.0, places=3)
        self.assertAlmostEqual(norm_df["weight_kg_minmax"].max(), 1.0, places=3)
        self.assertTrue((norm_df["weight_kg_minmax"] >= 0.0).all())
        self.assertTrue((norm_df["weight_kg_minmax"] <= 1.0).all())

        # Check Z-score standardization distribution properties
        self.assertAlmostEqual(norm_df["weight_kg_zscore"].mean(), 0.0, places=2)
        self.assertAlmostEqual(norm_df["weight_kg_zscore"].std(), 1.0, places=2)

    def test_end_to_end_pipeline(self) -> None:
        """Verify execution of the full pipeline produces expected clean output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_file = tmp_path / "raw.csv"
            clean_file = tmp_path / "clean.csv"
            results_file = tmp_path / "results.json"

            self.raw_df.to_csv(raw_file, index=False)
            res = run_pipeline(
                input_path=raw_file,
                output_path=clean_file,
                results_path=results_file,
                generate_plots=False,
            )

            self.assertTrue(clean_file.is_file())
            self.assertTrue(results_file.is_file())
            self.assertEqual(res["clean_shape"], [1000, 21])
            self.assertEqual(res["duplicates_removed"], 35)
            self.assertEqual(res["negative_distance_fixed"], 6)
            self.assertEqual(sum(res["missing_after"].values()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
