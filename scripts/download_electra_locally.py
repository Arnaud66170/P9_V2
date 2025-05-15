# scripts/download_electra_locally.py

from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token = os.getenv("HF_TOKEN"))

LOCAL_DIR = "models/hf_assets/electra-small-discriminator"

if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR, exist_ok = True)
    print(f"📁 Dossier créé : {LOCAL_DIR}")
else:
    print(f"✅ Dossier déjà présent : {LOCAL_DIR}")

print("⬇️ Téléchargement du modèle et du tokenizer ELECTRA (google/electra-small-discriminator)...")
model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels = 2)
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

print("💾 Sauvegarde en local...")
model.save_pretrained(LOCAL_DIR)
tokenizer.save_pretrained(LOCAL_DIR)

print("✅ Modèle ELECTRA-Small téléchargé avec succès dans :", LOCAL_DIR)
