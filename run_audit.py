"""Headless FairLens run: prints every audit table and saves results/.
Use this if the live dashboard fails during the presentation.
    python run_audit.py
"""
import json, os
import pandas as pd
from fairlens_core import run_experiment_a, run_experiment_b

pd.set_option("display.width", 160)
COLS = ["n", "accuracy", "precision", "recall_tpr", "fpr", "fnr", "selection_rate"]
os.makedirs("results", exist_ok=True)
out = {}

A = run_experiment_a()
for k in ("skewed", "representative"):
    print(f"\n=== Experiment A · {k} training (Group B share {A[k]['share_b']:.0%}) ===")
    print((A[k]["report"][COLS].astype(float)).round(3))
    print({g: round(v, 3) for g, v in A[k]["gaps"].items()})
    A[k]["report"][COLS].to_csv(f"results/expA_{k}.csv")
    out[f"A_{k}"] = {"report": A[k]["report"][COLS].astype(float).round(4).to_dict(orient="index"),
                     "gaps": A[k]["gaps"], "share_b": A[k]["share_b"]}

B = run_experiment_b()
for k in ("B1", "B2", "B3"):
    print(f"\n=== Experiment B · {k}: {B[k]['label']} (threshold {B[k]['threshold']}) ===")
    print((B[k]["report"][COLS].astype(float)).round(3))
    print({g: round(v, 3) for g, v in B[k]["gaps"].items()})
    B[k]["report"][COLS].to_csv(f"results/expB_{k}.csv")
    th = B[k]["threshold"]
    out[k] = {"label": B[k]["label"],
              "threshold": th if isinstance(th, dict) else round(float(th), 3),
              "report": B[k]["report"][COLS].astype(float).round(4).to_dict(orient="index"),
              "gaps": B[k]["gaps"]}
print("\nBaseline LR coefficients (standardised):", B["B1"]["coef"])
out["B1"]["coef"] = {k: float(v) for k, v in B["B1"]["coef"].items()}
A["skewed"]["train"].head(200).to_csv("results/sample_training_data_expA.csv", index=False)
json.dump(out, open("results/results.json", "w"), indent=2, default=float)
print("\nSaved results/ (CSV tables + results.json)")
