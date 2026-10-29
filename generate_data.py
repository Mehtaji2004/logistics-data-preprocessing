"""Simulate a raw logistics shipment dataset with realistic data-quality problems.

The dataset imitates a transport/shipping log: one row per shipment.
Problems deliberately injected (so the cleaning pipeline has real work to do):
  * missing values in several columns
  * duplicate rows
  * inconsistent text labels (e.g. 'Delhi', 'delhi ', 'DELHI')
  * outliers (impossible delivery times, extreme weights/costs)
  * negative / invalid values
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru", "Hyderabad", "Pune", "Ahmedabad"]
modes = ["Road", "Rail", "Air", "Sea"]
status = ["Delivered", "In Transit", "Delayed"]

origin = rng.choice(cities, N)
destination = rng.choice(cities, N)
mode = rng.choice(modes, N, p=[0.55, 0.25, 0.15, 0.05])

distance = rng.normal(900, 350, N).clip(60, 2200).round(1)
speed = {"Road": 50, "Rail": 60, "Air": 600, "Sea": 30}
shipping_hours = np.array([distance[i] / speed[mode[i]] + rng.normal(6, 2) for i in range(N)]).clip(2).round(1)
weight = rng.gamma(3.0, 450, N).round(1)  # kg
cost = (distance * 0.9 + weight * 0.35 + rng.normal(0, 150, N)).clip(100).round(2)
stock = rng.integers(20, 500, N).astype(float)  # inventory units at origin warehouse

dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 270, N), unit="D")

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000 + i}" for i in range(N)],
    "ship_date": dates,
    "origin": origin,
    "destination": destination,
    "transport_mode": mode,
    "distance_km": distance,
    "weight_kg": weight,
    "shipping_hours": shipping_hours,
    "shipping_cost_inr": cost,
    "inventory_units": stock,
    "status": rng.choice(status, N, p=[0.7, 0.15, 0.15]),
})

# ---- inject problems -------------------------------------------------------
# 1. missing values
for col, frac in [("distance_km", 0.06), ("weight_kg", 0.05), ("shipping_hours", 0.08),
                  ("shipping_cost_inr", 0.04), ("inventory_units", 0.07), ("transport_mode", 0.03)]:
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2. inconsistent labels
idx = rng.choice(N, 90, replace=False)
df.loc[idx, "origin"] = df.loc[idx, "origin"].str.upper()
idx = rng.choice(N, 70, replace=False)
df.loc[idx, "destination"] = df.loc[idx, "destination"].str.lower() + " "
idx = rng.choice(N, 40, replace=False)
df.loc[idx, "transport_mode"] = df.loc[idx, "transport_mode"].astype(str).str.lower()

# 3. outliers and invalid values
idx = rng.choice(N, 12, replace=False)
df.loc[idx, "shipping_hours"] = rng.uniform(900, 2500, 12).round(1)
idx = rng.choice(N, 10, replace=False)
df.loc[idx, "weight_kg"] = rng.uniform(15000, 40000, 10).round(1)
idx = rng.choice(N, 8, replace=False)
df.loc[idx, "shipping_cost_inr"] = rng.uniform(60000, 150000, 8).round(2)
idx = rng.choice(N, 6, replace=False)
df.loc[idx, "distance_km"] = -df.loc[idx, "distance_km"].abs()

# 4. duplicates
dups = df.sample(35, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("raw_logistics_data.csv", index=False)
print(f"Saved raw_logistics_data.csv with shape {df.shape}")
