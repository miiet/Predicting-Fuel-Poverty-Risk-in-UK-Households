from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
metrics_path = PROJECT_ROOT / "outputs" / "metrics" / "lightgbm_no_leakage_metrics.json"
out_csv = PROJECT_ROOT / "outputs" / "metrics" / "final_summary.csv"

with open(metrics_path, "r", encoding="utf-8") as f:
    m = json.load(f)

cr = m["classification_report"]
row = {
    "model": m["model"],
    "threshold": m["threshold"],
    "roc_auc": m["roc_auc"],
    "precision_risk1": cr["1"]["precision"],
    "recall_risk1": cr["1"]["recall"],
    "f1_risk1": cr["1"]["f1-score"],
    "accuracy": cr["accuracy"],
    "n_train": m["n_train"],
    "n_test": m["n_test"],
}

df = pd.DataFrame([row])
df.to_csv(out_csv, index=False)

print("✅ Saved:", out_csv)
print(df.to_string(index=False))