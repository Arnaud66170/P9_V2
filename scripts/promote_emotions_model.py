# scripts/promote_emotions_model.py

from mlflow import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="emotions_classifier",
    version="3",  # ou le numéro affiché dans ton terminal
    stage="Production",
    archive_existing_versions=True
)

print("✅ Version 3 promue en Production.")
