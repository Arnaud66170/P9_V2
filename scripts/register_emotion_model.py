# scripts/register_emotion_model.py

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "Default"
RUN_NAME = "retrospective_emotions_log"
MODEL_NAME = "emotions_classifier"

client = MlflowClient()

# 📌 Récupère l'ID du dernier run avec ce nom
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                          filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
                          order_by=["start_time DESC"],
                          max_results=1)

if not runs:
    raise ValueError(f"❌ Aucun run trouvé avec le nom {RUN_NAME}")

run = runs[0]
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/emotions_model"

# 🧠 Enregistrement dans le Model Registry
print(f"📦 Enregistrement du modèle : {MODEL_NAME}")
result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

# ✅ Promotion en "Production"
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ Modèle {MODEL_NAME} v{result.version} promu en Production.")

# Exécution :
# python scripts/register_emotion_model.py