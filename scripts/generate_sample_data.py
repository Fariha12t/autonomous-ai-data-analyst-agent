"""Generate a reproducible synthetic sales dataset at data/sample_sales.csv.

Run:  python scripts/generate_sample_data.py
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
N_ORDERS = 3000
START, END = date(2024, 1, 1), date(2025, 12, 31)

REGIONS = {"North": 0.22, "South": 0.28, "East": 0.20, "West": 0.30}
SEGMENTS = {"Consumer": 0.55, "Corporate": 0.30, "Small Business": 0.15}
# category -> (products with base price, profit margin)
CATALOG = {
    "Electronics": ({"Laptop": 820, "Smartphone": 560, "Headphones": 130, "Monitor": 260}, 0.14),
    "Furniture": ({"Office Chair": 190, "Standing Desk": 430, "Bookcase": 150, "Sofa": 610}, 0.10),
    "Office Supplies": ({"Notebook Pack": 14, "Printer Paper": 24, "Stapler": 11, "Desk Organizer": 28}, 0.32),
    "Clothing": ({"Jacket": 95, "Sneakers": 80, "T-Shirt": 22, "Backpack": 48}, 0.22),
}
CATEGORY_WEIGHTS = {"Electronics": 0.25, "Furniture": 0.15, "Office Supplies": 0.40, "Clothing": 0.20}
MONTH_SEASONALITY = {1: 0.85, 2: 0.85, 3: 0.95, 4: 0.95, 5: 1.0, 6: 1.0,
                     7: 0.95, 8: 1.0, 9: 1.05, 10: 1.1, 11: 1.3, 12: 1.4}


def _pick(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def main() -> None:
    rng = random.Random(SEED)
    span = (END - START).days
    # sample dates weighted by month seasonality and a gentle growth trend
    all_days = [START + timedelta(days=i) for i in range(span + 1)]
    day_weights = [
        MONTH_SEASONALITY[d.month] * (1 + 0.35 * (d - START).days / span) for d in all_days
    ]
    dates = sorted(rng.choices(all_days, weights=day_weights, k=N_ORDERS))

    rows = []
    for i, d in enumerate(dates, start=1):
        category = _pick(rng, CATEGORY_WEIGHTS)
        products, margin = CATALOG[category]
        product = rng.choice(list(products))
        unit_price = round(products[product] * rng.uniform(0.92, 1.08), 2)
        quantity = rng.choices([1, 2, 3, 4, 5], weights=[50, 25, 12, 8, 5])[0]
        discount = rng.choices([0, 0.05, 0.10, 0.20], weights=[60, 20, 15, 5])[0]
        revenue = round(unit_price * quantity * (1 - discount), 2)
        # heavier discounts erode margin; add noise
        profit = round(revenue * (margin - discount * 0.8 + rng.uniform(-0.04, 0.04)), 2)
        rows.append({
            "order_id": f"ORD-{i:05d}",
            "order_date": d.isoformat(),
            "region": _pick(rng, REGIONS),
            "customer_segment": _pick(rng, SEGMENTS),
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "profit": profit,
        })

    out = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
