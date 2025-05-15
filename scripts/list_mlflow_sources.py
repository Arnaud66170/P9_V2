import os
import yaml
import pandas as pd

mlruns_path = os.path.abspath("mlruns/0")
runs_data = []

for run_id in os.listdir(mlruns_path):
    run_path = os.path.join(mlruns_path, run_id)
    meta_file = os.path.join(run_path, "meta.yaml")
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
            runs_data.append({
                "run_id": meta.get("run_id", ""),
                "run_name": meta.get("run_name", ""),
                "source_name": meta.get("source_name", "❌ Non spécifié"),
                "artifact_uri": meta.get("artifact_uri", "")
            })

df = pd.DataFrame(runs_data)

# ✅ Export CSV
export_path = os.path.abspath("mlflow_sources_report.csv")
df.to_csv(export_path, index=False)


df = pd.DataFrame(runs_data)
print(df.sort_values(by="run_name"))

print(f"✅ Rapport exporté : {export_path}")
print(df)

# activation du script :
# python scripts/list_mlflow_sources.py