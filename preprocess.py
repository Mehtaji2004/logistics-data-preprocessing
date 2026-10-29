"""Data cleaning and preprocessing pipeline for logistics shipment data.

Steps: 1) inspect  2) remove duplicates  3) standardise labels  4) fix invalid values
       5) outlier detection (IQR)  6) missing-value imputation  7) normalisation
Every step records before/after numbers in results.json for the report.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = {}
df = pd.read_csv("raw_logistics_data.csv", parse_dates=["ship_date"])
num_cols = ["distance_km", "weight_kg", "shipping_hours", "shipping_cost_inr", "inventory_units"]

# ---- Step 1: inspect -------------------------------------------------------
results["raw_shape"] = list(df.shape)
results["missing_before"] = df.isna().sum().to_dict()
results["describe_before"] = df[num_cols].describe().round(2).to_dict()

# ---- Step 2: duplicates ----------------------------------------------------
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
results["duplicates_removed"] = n_dup

# ---- Step 3: standardise categorical labels --------------------------------
results["labels_before"] = {
    "origin": sorted(df["origin"].dropna().unique().tolist()),
    "transport_mode": sorted(df["transport_mode"].dropna().unique().tolist()),
}
for col in ["origin", "destination", "transport_mode"]:
    df[col] = df[col].str.strip().str.title()
results["labels_after"] = {
    "origin": sorted(df["origin"].dropna().unique().tolist()),
    "transport_mode": sorted(df["transport_mode"].dropna().unique().tolist()),
}

# ---- Step 4: invalid values (negative distance) ----------------------------
n_neg = int((df["distance_km"] < 0).sum())
df.loc[df["distance_km"] < 0, "distance_km"] = np.nan   # treat as missing
results["negative_distance_fixed"] = n_neg

# ---- Step 5: outliers with IQR rule ----------------------------------------
out_counts = {}
bounds = {}
for col in ["shipping_hours", "weight_kg", "shipping_cost_inr"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < lo) | (df[col] > hi)
    out_counts[col] = int(mask.sum())
    bounds[col] = [round(lo, 2), round(hi, 2)]
    df.loc[mask, col] = np.nan                           # replaced later by imputation
results["outliers_flagged"] = out_counts
results["iqr_bounds"] = bounds

# boxplot of the raw outlier columns (before removal) for the report
raw = pd.read_csv("raw_logistics_data.csv")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, col in zip(axes, ["shipping_hours", "weight_kg", "shipping_cost_inr"]):
    ax.boxplot(raw[col].dropna(), vert=True)
    ax.set_title(col)
plt.suptitle("Raw data: outliers visible in box plots")
plt.tight_layout()
plt.savefig("fig_outliers_before.png", dpi=150)
plt.close()

# ---- Step 6: missing values ------------------------------------------------
results["missing_after_outlier_step"] = df.isna().sum().to_dict()
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())           # median: robust to skew
df["transport_mode"] = df["transport_mode"].fillna(df["transport_mode"].mode()[0])
results["missing_after"] = df.isna().sum().to_dict()

# missing-value chart
fig, ax = plt.subplots(figsize=(7, 3.6))
mb = pd.Series(results["missing_before"])
mb = mb[mb > 0]
ax.bar(mb.index, mb.values, color="#d9622b")
ax.set_title("Missing values per column (raw data)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("fig_missing_before.png", dpi=150)
plt.close()

# ---- Step 7: normalisation -------------------------------------------------
for col in num_cols:
    mn, mx = df[col].min(), df[col].max()
    df[col + "_minmax"] = ((df[col] - mn) / (mx - mn)).round(4)
    df[col + "_zscore"] = ((df[col] - df[col].mean()) / df[col].std()).round(4)

results["clean_shape"] = list(df.shape)
results["describe_after"] = df[num_cols].describe().round(2).to_dict()

# a worked example row: raw vs minmax vs zscore
ex = df.iloc[0]
results["worked_example"] = {
    "shipment_id": ex["shipment_id"],
    "weight_kg": float(ex["weight_kg"]),
    "weight_min": float(df["weight_kg"].min()),
    "weight_max": float(df["weight_kg"].max()),
    "weight_mean": round(float(df["weight_kg"].mean()), 2),
    "weight_std": round(float(df["weight_kg"].std()), 2),
    "weight_minmax": float(ex["weight_kg_minmax"]),
    "weight_zscore": float(ex["weight_kg_zscore"]),
}

# before/after distribution chart
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].hist(raw["shipping_hours"].dropna(), bins=40, color="#d9622b")
axes[0].set_title("shipping_hours BEFORE cleaning")
axes[1].hist(df["shipping_hours"], bins=40, color="#2b8ad9")
axes[1].set_title("shipping_hours AFTER cleaning")
plt.tight_layout()
plt.savefig("fig_hours_before_after.png", dpi=150)
plt.close()

# ---- Downstream impact: average cost per mode, raw vs clean ----------------
raw_mode = raw.copy()
raw_mode["transport_mode"] = raw_mode["transport_mode"].str.strip().str.title()
results["mean_cost_raw"] = round(float(raw["shipping_cost_inr"].mean()), 2)
results["mean_cost_clean"] = round(float(df["shipping_cost_inr"].mean()), 2)
results["mean_hours_raw"] = round(float(raw["shipping_hours"].mean()), 2)
results["mean_hours_clean"] = round(float(df["shipping_hours"].mean()), 2)
results["mean_weight_raw"] = round(float(raw["weight_kg"].mean()), 2)
results["mean_weight_clean"] = round(float(df["weight_kg"].mean()), 2)
results["cost_by_mode_clean"] = df.groupby("transport_mode")["shipping_cost_inr"].mean().round(2).to_dict()

df.to_csv("clean_logistics_data.csv", index=False)
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Done. Clean shape:", df.shape)
print(json.dumps({k: results[k] for k in ["duplicates_removed", "negative_distance_fixed", "outliers_flagged"]}, indent=2))
