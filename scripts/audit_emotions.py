# scripts/audit_emotions.py

import os
import sys
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 📚 Chemins projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from constants.emotions import EMOTION_LABELS

# -------------------------
# 1. Chargement du pipeline
# -------------------------
model_path = os.path.abspath("notebooks/models/emotions/model").replace("\\", "/")
tokenizer_path = os.path.abspath("notebooks/models/emotions/tokenizer").replace("\\", "/")

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

emotion_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    function_to_apply="sigmoid"
)

# -------------------------
# 2. Jeu de test : 50 phrases
# -------------------------
examples = [
    "I'm scared. Why is the turbulence so intense?",
    "Finally landed after 3 hours delay... yay.",
    "Best flight ever! Love the service ❤️",
    "This is the dumbest thing I've experienced.",
    "I’m literally shaking. This pilot is a legend.",
    "Why do I always get the middle seat 😩",
    "Appreciate the smooth boarding process.",
    "Stuck on the tarmac for 45 minutes. Again.",
    "The view above the clouds... breathtaking.",
    "You lost my luggage. Again.",
    "Thank you for the upgrade ✈️🥂",
    "Food was inedible. Like actual garbage.",
    "This crew is amazing, honestly.",
    "What a nightmare. Never flying this airline again.",
    "The baby screamed for 9 hours straight.",
    "Overbooked. Got bumped. No compensation. Rude.",
    "Crying tears of joy... I made it to the wedding!",
    "Captain made a joke mid-flight. Loved it.",
    "The air hostess just insulted me... wow.",
    "So cramped I can't feel my legs.",
    "Such kindness. I felt truly cared for.",
    "Security made me feel like a criminal.",
    "Lost my passport mid-flight. Panic.",
    "My first flight ever. So emotional.",
    "Pilot landed like butter on the runway.",
    "Turbulence had me praying, not gonna lie.",
    "This app is terrible. Check-in failed twice.",
    "Dog next to me farted the whole flight.",
    "Window seat. Clouds. Peace.",
    "I cried when I saw my grandma at arrival.",
    "Delayed 5 times. No info. No help.",
    "Such a joyful surprise: my partner proposed mid-air!",
    "Wasn't expecting champagne and warm cookies 😍",
    "Is it normal the plane makes that noise?",
    "I wanted to scream. The man snored like a buffalo.",
    "No Wi-Fi. No movies. Just pain.",
    "Smiling crew made it bearable. Thanks.",
    "She held my hand during takeoff. So sweet.",
    "The seats were wet. What even is this?",
    "Flight attendant ignored me the whole time.",
    "This guy keeps coughing. I'm paranoid.",
    "In awe of the sunrise from the sky.",
    "Finally, peace and quiet after chaos.",
    "Creeped out. Guy stared at me entire flight.",
    "Cabin smells like soup and feet.",
    "How is this even legal? Absolute mess.",
    "Smiles and champagne at 30,000 feet. Bliss.",
    "The safety video was hilarious 😂",
    "You made my year with this gesture.",
    "Everyone clapped at landing. Goosebumps."
]

# -------------------------
# 3. Audit automatique + seuils
# -------------------------
results = []

for text in examples:
    preds = emotion_pipeline(text)[0]
    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    top_emotions = [
        (
            EMOTION_LABELS[int(p["label"].replace("LABEL_", ""))],
            round(p["score"], 3)
        )
        for p in preds_sorted if p["score"] > 0.25
    ]

    results.append({
        "text": text,
        "top_1": top_emotions[0][0] if len(top_emotions) > 0 else None,
        "score_1": top_emotions[0][1] if len(top_emotions) > 0 else None,
        "top_2": top_emotions[1][0] if len(top_emotions) > 1 else None,
        "score_2": top_emotions[1][1] if len(top_emotions) > 1 else None,
        "raw_labels": ", ".join(f"{label}:{score}" for label, score in top_emotions)
    })

df_preds = pd.DataFrame(results)

# -------------------------
# 4. highlight_errors() intégré
# -------------------------
INTENSE_WORDS = [

    # 🔴 Colère / insultes
    "hate", "idiot", "stupid", "moron", "jerk", "bastard", "dumb", "loser",
    "useless", "freak", "trash", "scumbag", "shit", "fuck", "asshole", "bitch",
    "douche", "crap", "retard", "dick", "sucker", "screw", "fucked",

    # 🔵 Peur / anxiété / panique
    "terrified", "panic", "anxious", "paranoid", "scared", "afraid", "shaking",
    "nervous", "trembling", "suffocating", "hyperventilating",

    # ⚫ Tristesse / désespoir
    "depressed", "miserable", "crying", "sobbing", "worthless", "hopeless",
    "sad", "broken", "destroyed", "ruined", "tired of life", "done with life",

    # 🟠 Dégoût / rejet
    "disgusting", "gross", "nauseating", "vile", "filthy", "revolting",
    "horrible", "horrid", "nasty",

    # ⚠️ Violence / menaces
    "kill", "die", "choke", "smash", "bleed", "murder", "slap", "beat", "explode",

    # 🟡 Dévalorisation / humiliation
    "pathetic", "embarrassing", "shame", "ashamed", "humiliated", "failure",

    # 🧨 Désespoir absolu
    "no one cares", "what's the point", "i want to disappear", "leave me alone",
    "end it all", "it's over", "i give up"
]

def contains_intense_words(text):
    pattern = r"\b(" + "|".join(re.escape(w) for w in INTENSE_WORDS) + r")\b"
    return bool(re.search(pattern, text.lower()))

def highlight_errors(df, threshold=0.25):
    return df[
        ((df["top_1"].isna()) | (df["top_1"] == "neutral") | (df["score_1"] < threshold)) &
        (df["text"].apply(contains_intense_words))
    ]

df_errors = highlight_errors(df_preds)

# -------------------------
# 5. Export
# -------------------------
output_dir = os.path.join("outputs")
os.makedirs(output_dir, exist_ok=True)

df_preds.to_csv(os.path.join(output_dir, "emotion_audit_flight_feedback.csv"), index=False)
df_errors.to_csv(os.path.join(output_dir, "emotion_errors_flagged.csv"), index=False)

print("✅ Audit complet exporté.")
print(f"📝 Prédictions complètes : outputs/emotion_audit_flight_feedback.csv")
print(f"⚠️ Cas suspects détectés : outputs/emotion_errors_flagged.csv")

# -------------------------
# 6. Export log_errors.csv enrichi
# -------------------------

log_error_path = os.path.join(output_dir, "log_errors.csv")

if not df_errors.empty:
    df_errors["intense_detected"] = df_errors["text"].apply(contains_intense_words)
    df_errors["alert_reason"] = "Prediction neutral/empty with intense terms"
    df_errors["raw_summary"] = df_errors.apply(
        lambda row: f"{row['top_1']} ({row['score_1']}) | {row['raw_labels']}", axis=1
    )
    df_errors = df_errors[[
        "text", "top_1", "score_1", "raw_labels", "intense_detected", "alert_reason", "raw_summary"
    ]]

    df_errors.to_csv(log_error_path, index=False, encoding="utf-8")
    print(f"⚠️ Erreurs silencieuses logguées dans : {log_error_path}")
else:
    print("✅ Aucun cas suspect détecté (selon INTENSE_WORDS).")
