"""Generate a synthetic customer dataset with a known, planted effect.

Planted ground truth
--------------------
* Retention (retained_90d) depends on segment:
    small_business ~ 60%, mid_market ~ 70%, enterprise ~ 80%
* PLANTED EFFECT: enterprise customers who signed up in Q3 2024 have
  retention reduced by 15 percentage points (80% -> 65%).
* CONFOUNDER: the same group (enterprise, Q3) also places fewer monthly
  orders, so order frequency is associated with the retention drop.
* Deliberate data quality issues, so a profiler has something to find:
    - ~2% missing values in `revenue`
    - ~0.5% extreme revenue outliers
    - ~1% duplicated rows

Usage
-----
    python eval/data_synthesisers/make_customers.py
        # 20,000 rows -> eval/datasets/customers.csv (+ customers.truth.json)
    python eval/data_synthesisers/make_customers.py --rows 5000 --seed 7 \
        --out eval/datasets/small.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Default output: eval/datasets/customers.csv (resolved relative to this file,
# so it works no matter which directory you run the script from)
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets" / "customers.csv"

SEGMENTS = ["small_business", "mid_market", "enterprise"]
SEGMENT_PROBS = [0.60, 0.30, 0.10]

BASE_RETENTION = {"small_business": 0.60, "mid_market": 0.70, "enterprise": 0.80}
MEAN_MONTHLY_ORDERS = {"small_business": 4.0, "mid_market": 8.0, "enterprise": 15.0}
AVG_ORDER_VALUE = {"small_business": 40.0, "mid_market": 90.0, "enterprise": 220.0}

REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["organic", "paid_search", "referral", "email"]

# Planted effect parameters
EFFECT_SEGMENT = "enterprise"
EFFECT_QUARTER = "2024Q3"
RETENTION_DROP = 0.15  # 15 percentage points
ORDER_FACTOR = 0.6  # enterprise Q3 customers order 40% less


def generate(rows: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    # Signup dates spread across all of 2024
    days = rng.integers(0, 366, size=rows)
    signup_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D")

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, rows + 1),
            "signup_date": signup_date,
            "segment": rng.choice(SEGMENTS, size=rows, p=SEGMENT_PROBS),
            "region": rng.choice(REGIONS, size=rows),
            "acquisition_channel": rng.choice(CHANNELS, size=rows, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    df["signup_quarter"] = df["signup_date"].dt.to_period("Q").astype(str)

    in_effect = (df["segment"] == EFFECT_SEGMENT) & (df["signup_quarter"] == EFFECT_QUARTER)

    # Monthly orders (confounder: lower in the affected group)
    mean_orders = df["segment"].map(MEAN_MONTHLY_ORDERS).astype(float)
    mean_orders = np.where(in_effect, mean_orders * ORDER_FACTOR, mean_orders)
    df["monthly_orders"] = rng.poisson(mean_orders)

    # Retention probability with the planted drop
    p_retain = df["segment"].map(BASE_RETENTION).astype(float)
    p_retain = np.where(in_effect, p_retain - RETENTION_DROP, p_retain)
    df["retained_90d"] = (rng.random(rows) < p_retain).astype(int)

    # Revenue = orders * order value * noise
    order_value = df["segment"].map(AVG_ORDER_VALUE).astype(float)
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=rows)
    df["revenue"] = (df["monthly_orders"] * order_value * noise).round(2)

    # --- Data quality issues -------------------------------------------------
    # Extreme outliers (~0.5% of rows)
    outlier_idx = rng.choice(rows, size=max(1, int(rows * 0.005)), replace=False)
    df.loc[outlier_idx, "revenue"] = (df.loc[outlier_idx, "revenue"] * 25 + 5000).round(2)

    # Missing revenue (~2%)
    missing_idx = rng.choice(rows, size=int(rows * 0.02), replace=False)
    df.loc[missing_idx, "revenue"] = np.nan

    # Duplicated rows (~1%)
    dupes = df.sample(n=int(rows * 0.01), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    # Keep date as plain YYYY-MM-DD text in the CSV
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")

    truth = compute_truth(df)
    return df, truth


def compute_truth(df: pd.DataFrame) -> dict:
    """Compute the observed values an analysis tool should reproduce."""
    clean = df.drop_duplicates()
    rate = clean.groupby(["segment", "signup_quarter"])["retained_90d"].mean().round(4)
    return {
        "planted_effect": {
            "segment": EFFECT_SEGMENT,
            "quarter": EFFECT_QUARTER,
            "retention_drop_pp": RETENTION_DROP * 100,
            "expected_retention_in_effect_group": BASE_RETENTION[EFFECT_SEGMENT] - RETENTION_DROP,
        },
        "rows_with_duplicates": len(df),
        "rows_after_dedup": len(clean),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_revenue": int(clean["revenue"].isna().sum()),
        "observed_retention_by_segment_quarter": {
            f"{seg}|{q}": float(v) for (seg, q), v in rate.items()
        },
        "observed_retention_overall": round(float(clean["retained_90d"].mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rows", type=int, default=20_000, help="base number of customers")
    parser.add_argument("--seed", type=int, default=42, help="random seed (same seed = same data)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output CSV path")
    args = parser.parse_args()

    df, truth = generate(args.rows, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2))

    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Wrote ground truth to {truth_path}")
    print("\nObserved retention for the affected segment:")
    for key, val in truth["observed_retention_by_segment_quarter"].items():
        if key.startswith(EFFECT_SEGMENT):
            print(f"  {key}: {val:.1%}")


if __name__ == "__main__":
    main()