"""Simulate a raw logistics shipment dataset with realistic data-quality problems.

Theoretical Foundation & Reference Benchmark:
---------------------------------------------
This simulation is modeled after publicly available logistics and freight benchmarks,
specifically referencing the schema and operational distributions of the:
  * DataCo Smart Supply Chain Dataset (Kaggle / Mendeley Data)
  * Multimodal Freight Transport & Shipping Records (Road, Rail, Air, Sea)

The generated dataset represents an enterprise shipment log recording 11 operational
attributes across transport, transit duration, cargo weight, freight cost, and warehouse inventory.

Injected Data Quality Defects (Simulating Real-World Operational Issues):
  1. Missing Data (MCAR/MAR): 3% to 8% null values across critical columns.
  2. Duplicate Entries: 35 duplicated records simulating repeated ETL extracts.
  3. Categorical Inconsistency: Arbitrary case casing ('DELHI', 'delhi ') and trailing spaces.
  4. Extreme Outliers: Sensor / manual entry errors (impossible shipping hours up to 2500h,
     unrealistic cargo weights up to 40 tonnes, extreme freight costs).
  5. Domain Constraint Violations: Negative travel distances (-km).

Author: Akshit Mehta (Yuva Intern Week 2 Logistics Analytics Task)
Compatible with: Python 3.8+ (pandas, numpy)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def simulate_logistics_dataset(
    n_samples: int = 1000,
    seed: int = 42,
    output_path: Optional[str | Path] = "raw_logistics_data.csv",
) -> pd.DataFrame:
    """Generate a realistic raw logistics dataset with deliberate quality anomalies.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of unique shipment records to generate prior to duplicate injection.
    seed : int, default=42
        Random seed ensuring deterministic reproducibility.
    output_path : Optional[str | Path], default="raw_logistics_data.csv"
        File path to save the generated CSV. If None, file is not written.

    Returns
    -------
    pd.DataFrame
        Simulated raw logistics DataFrame with realistic quality defects.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    # Reference operational categorical features
    cities = [
        "Delhi",
        "Mumbai",
        "Chennai",
        "Kolkata",
        "Bengaluru",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
    ]
    modes = ["Road", "Rail", "Air", "Sea"]
    status = ["Delivered", "In Transit", "Delayed"]

    # Categorical distributions
    origin = rng.choice(cities, n)
    destination = rng.choice(cities, n)
    mode = rng.choice(modes, n, p=[0.55, 0.25, 0.15, 0.05])

    # Numerical feature distributions
    distance = rng.normal(900, 350, n).clip(60, 2200).round(1)
    speed = {"Road": 50, "Rail": 60, "Air": 600, "Sea": 30}
    shipping_hours = np.array(
        [distance[i] / speed[mode[i]] + rng.normal(6, 2) for i in range(n)]
    ).clip(2).round(1)
    weight = rng.gamma(3.0, 450, n).round(1)  # cargo weight in kg
    cost = (distance * 0.9 + weight * 0.35 + rng.normal(0, 150, n)).clip(100).round(2)
    stock = rng.integers(20, 500, n).astype(float)  # origin warehouse inventory units

    # Temporal features (spanning 9 months of operations)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 270, n), unit="D"
    )

    df = pd.DataFrame(
        {
            "shipment_id": [f"SHP{100000 + i}" for i in range(n)],
            "ship_date": dates,
            "origin": origin,
            "destination": destination,
            "transport_mode": mode,
            "distance_km": distance,
            "weight_kg": weight,
            "shipping_hours": shipping_hours,
            "shipping_cost_inr": cost,
            "inventory_units": stock,
            "status": rng.choice(status, n, p=[0.7, 0.15, 0.15]),
        }
    )

    # -------------------------------------------------------------------------
    # Inject Data Quality Defects
    # -------------------------------------------------------------------------

    # 1. Missing Values (simulate transmission losses / incomplete logging)
    missing_specs = [
        ("distance_km", 0.06),
        ("weight_kg", 0.05),
        ("shipping_hours", 0.08),
        ("shipping_cost_inr", 0.04),
        ("inventory_units", 0.07),
        ("transport_mode", 0.03),
    ]
    for col, frac in missing_specs:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Inconsistent Text Labels (simulate manual operator variations & trailing whitespace)
    idx_origin = rng.choice(n, 90, replace=False)
    df.loc[idx_origin, "origin"] = df.loc[idx_origin, "origin"].str.upper()

    idx_dest = rng.choice(n, 70, replace=False)
    df.loc[idx_dest, "destination"] = df.loc[idx_dest, "destination"].str.lower() + " "

    idx_mode = rng.choice(n, 40, replace=False)
    df.loc[idx_mode, "transport_mode"] = (
        df.loc[idx_mode, "transport_mode"].astype(str).str.lower()
    )

    # 3. Outliers & Invalid Domain Values (sensor calibration & entry errors)
    idx_hours = rng.choice(n, 12, replace=False)
    df.loc[idx_hours, "shipping_hours"] = rng.uniform(900, 2500, 12).round(1)

    idx_weight = rng.choice(n, 10, replace=False)
    df.loc[idx_weight, "weight_kg"] = rng.uniform(15000, 40000, 10).round(1)

    idx_cost = rng.choice(n, 8, replace=False)
    df.loc[idx_cost, "shipping_cost_inr"] = rng.uniform(60000, 150000, 8).round(2)

    idx_dist = rng.choice(n, 6, replace=False)
    df.loc[idx_dist, "distance_km"] = -df.loc[idx_dist, "distance_km"].abs()

    # 4. Duplicate Records (simulating duplicate batch ingestion)
    dups = df.sample(35, random_state=1)
    df = (
        pd.concat([df, dups], ignore_index=True)
        .sample(frac=1, random_state=7)
        .reset_index(drop=True)
    )

    if output_path is not None:
        path = Path(output_path)
        df.to_csv(path, index=False)
        print(f"[Data Generator] Successfully created '{path}' with shape {df.shape}.")

    return df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data simulation."""
    parser = argparse.ArgumentParser(
        description="Generate a simulated raw logistics dataset with realistic defects."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of baseline unique shipment records (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw_logistics_data.csv",
        help="Target output CSV file path (default: raw_logistics_data.csv).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulate_logistics_dataset(
        n_samples=args.samples,
        seed=args.seed,
        output_path=args.output,
    )
