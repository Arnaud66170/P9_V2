# huggingface_api/app.py

import gradio as gr
import sys
import os
import re
import random
import threading
import mlflow
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import csv
from pathlib import Path
from src.constants.emotions import EMOTION_LABELS
from src.utils.emotion_utils import filter_predictions
from src.utils.logger import log_user_event
from src.utils.alert_email import send_alert_email
from transformers import pipeline as hf_pipeline, AutoModelForSequenceClassification, AutoTokenizer
import plotly.express as px

# === Chargement du modèle depuis le Model Registry ===
# model = mlflow.transformers.load_model("models:/emotions_classifier/Production")
model_path = "models/electra_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# === Globals ===
HISTORY_LIMIT = 5
history = deque(maxlen=HISTORY_LIMIT)
ALERT_WINDOW_MINUTES = 5
ALERT_COOLDOWN_MINUTES = 10
FEEDBACK_ALERT_THRESHOLD = 3
alert_history = []
FEEDBACK_CSV = os.path.abspath("feedback_log_emotions.csv")
EXPORT_CSV_PATH = os.path.abspath("outputs/emotion_stats_2000.csv")
suspect_text = ""

# === Tweets d'exemples ===
tweet_examples = [
    "I feel amazing today!", "I'm so angry right now.", "Everything is fine, just a little tired.",
    "Why is nobody answering me?", "This is the best day of my life!",
    "Sure, because waiting 3 hours for coffee is totally normal 😒",
    "I miss you... but maybe it's for the best.", "Finally got that promotion 🥳",
    "Just great. Another Monday. Yay.", "Thank you for ruining my day 😊",
    "She smiled, but her eyes were empty.", "I can’t take this anymore.",
    "That’s it. I’m done. Don’t call me.", "Well... that escalated quickly.",
    "I'm so proud of you ❤️", "Guess who forgot their keys again 🙄",
    "I just want to disappear for a while.", "Wow. Thanks a lot. Really helpful. 👏",
    "I can't believe you remembered 🥺", "Oh joy, another pointless meeting...",
    "Let's pretend everything’s okay 🌪️", "Honestly? I don’t even care anymore.",
    "You’ve made my day ☀️", "Love this song. Hits hard today.",
    "So peaceful here. I could stay forever.", "You always know just what to say 😌",
    "I guess it’s whatever now 🤷", "What’s the point of trying?",
    "I laughed so hard I cried 😂", "This meal is a 10/10 👌",
    "Please don’t leave me.", "I knew this would happen.",
    "Don’t talk to me like that again.", "You forgot again. Of course you did.",
    "Oh wow, a surprise. Totally didn’t expect that 🙃",
    "That was unexpected... and kinda sweet.", "Creeped out. That guy followed me home.",
    "Ugh. Can't stand this anymore.", "So tired of pretending I'm fine.",
    "Feeling super grateful today 🙏", "Can’t stop smiling 😁",
    "I'm proud of how far I’ve come.", "Back at it again. Let’s gooo 💪",
    "I’m trying, okay? I really am.", "This is the dumbest thing ever.",
    "You always ruin everything 😡", "Missing the good old days.",
    "Best birthday ever 🎂🎈", "I'm not crying. You are 😢",
    "Guess I should’ve seen it coming.", "No one listens. No one cares."
]

INTENSE_WORDS = [
    "hate", "idiot", "stupid", "moron", "jerk", "bastard", "dumb", "loser",
    "useless", "freak", "trash", "scumbag", "shit", "fuck", "asshole", "bitch",
    "douche", "crap", "retard", "dick", "sucker", "screw", "fucked",
    "terrified", "panic", "anxious", "paranoid", "scared", "afraid", "shaking",
    "nervous", "trembling", "suffocating", "hyperventilating",
    "depressed", "miserable", "crying", "sobbing", "worthless", "hopeless",
    "sad", "broken", "destroyed", "ruined", "tired of life", "done with life",
    "disgusting", "gross", "nauseating", "vile", "filthy", "revolting",
    "horrible", "horrid", "nasty",
    "kill", "die", "choke", "smash", "bleed", "murder", "slap", "beat", "explode",
    "pathetic", "embarrassing", "shame", "ashamed", "humiliated", "failure",
    "no one cares", "what's the point", "i want to disappear", "leave me alone",
    "end it all", "it's over", "i give up"
]

def contains_intense_words(text):
    words = re.findall(r"\w+", text.lower())
    return any(w in INTENSE_WORDS for w in words)

model_path = Path("notebooks/models/emotions/model").resolve().as_posix()
tokenizer_path = Path("notebooks/models/emotions/tokenizer").resolve().as_posix()
local_model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
emotion_pipeline = hf_pipeline("text-classification", model=local_model, tokenizer=tokenizer, top_k=None, function_to_apply="sigmoid")

def predict_emotions(text):
    global suspect_text
    if not text.strip():
        return "⛔️ Texte vide", None, None, ""
    raw_output = emotion_pipeline(text)[0]
    labels = [EMOTION_LABELS[int(e["label"].replace("LABEL_", ""))] for e in raw_output]
    scores = [e["score"] for e in raw_output]
    filtered = filter_predictions(scores, labels, threshold=0.25)
    result = ", ".join(f"{l} ({s:.2f})" for l, s in sorted(filtered, key=lambda x: x[1], reverse=True)) if filtered else "😐 Aucune émotion détectée"
    top = filtered[0] if filtered else ("neutral", 0.0)
    history.appendleft({"text": text, "emotion": top[0], "score": round(top[1]*100, 2)})
    alert_banner = ""
    if contains_intense_words(text):
        suspect_text = text
        alert_banner = "🟥 <strong style='color:red;'>⚠️ Ce tweet contient un mot intense. Consultez l’onglet \\\"Cas suspects\\\" pour audit.</strong>"
    html_output = f"<h2 style='text-align:center;'>🧠 Résultat : {result}</h2>" + (f"<div style='margin-top:10px'>{alert_banner}</div>" if alert_banner else "")
    return html_output, update_pie_chart(), update_history(), text if alert_banner else ""

def audit_text(text):
    output = emotion_pipeline(text)[0]
    labels = [EMOTION_LABELS[int(e["label"].replace("LABEL_", ""))] for e in output]
    scores = [e["score"] for e in output]
    filtered = [(l, s) for l, s in zip(labels, scores) if s >= 0.25]
    filtered.sort(key=lambda x: x[1], reverse=True)
    alert = "⚠️ Attention : émotion absente malgré termes forts\n" if contains_intense_words(text) and (not filtered or filtered[0][0] == "neutral") else ""
    return alert + ", ".join(f"{l} ({s:.2f})" for l, s in filtered) or "😐 Aucune émotion détectée", {l: s for l, s in filtered}

def update_pie_chart():
    emotions = [h['emotion'] for h in history]
    df = pd.DataFrame(emotions, columns=['emotion'])
    if df.empty:
        return None
    counts = df['emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    return px.pie(counts, names='Emotion', values='Count', title='Distribution émotions récentes')

def update_history():
    df = pd.DataFrame(list(history))
    return df.rename(columns={"text": "Tweet", "emotion": "Emotion", "score": "Confiance (%)"}) if not df.empty else pd.DataFrame(columns=["Tweet", "Emotion", "Confiance (%)"])

def save_feedback(tweet, feedback, comment):
    if not history:
        return "❌ Aucun historique disponible.", ""
    last = history[0]
    emotion = last["emotion"]
    confidence = last["score"]
    timestamp = datetime.now()
    row = {
        "tweet": tweet,
        "predicted_emotion": emotion,
        "proba": confidence,
        "user_feedback": feedback,
        "comment": comment,
        "timestamp": timestamp.isoformat()
    }
    try:
        file_exists = os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Erreur CSV : {e}")
    log_user_event("feedback", tweet_text=tweet, predicted_label=emotion, proba=confidence, feedback=feedback, comment=comment)
    if feedback == "👎 No":
        alert_history.append(timestamp)
        now = datetime.now()
        recent_alerts = [t for t in alert_history if now - t < timedelta(minutes=ALERT_WINDOW_MINUTES)]
        alert_history[:] = recent_alerts
        if len(recent_alerts) >= FEEDBACK_ALERT_THRESHOLD:
            if not hasattr(save_feedback, "last_alert") or now - save_feedback.last_alert > timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                threading.Thread(target=send_alert_email, args=(len(recent_alerts),), daemon=True).start()
                save_feedback.last_alert = now
    return "✅ Feedback enregistré.", update_feedback_stats()

def update_feedback_stats():
    if not os.path.exists(FEEDBACK_CSV):
        return "Aucun feedback encore."
    try:
        df = pd.read_csv(FEEDBACK_CSV)
        count_yes = (df['user_feedback'] == '👍 Yes').sum()
        count_no = (df['user_feedback'] == '👎 No').sum()
        return f"👍 Yes: {count_yes} | 👎 No: {count_no} | Total: {len(df)}"
    except:
        return "Erreur lecture stats."

def download_emotion_stats():
    return EXPORT_CSV_PATH if os.path.exists(EXPORT_CSV_PATH) else None

with gr.Blocks() as demo:
    gr.Markdown("## 😶‍🌫️ Analyse multi-label avec ELECTRA fine-tuné")

    with gr.Tabs():
        with gr.Tab("🔍 Analyse des émotions"):
            with gr.Row():
                with gr.Column():
                    input_box = gr.Textbox(label="💬 Texte", lines=3)
                    random_btn = gr.Button("🎲 Tweet aléatoire")
                with gr.Column():
                    analyze_btn = gr.Button("Analyser")
                    sentiment_output = gr.HTML()
            gr.Markdown("### 📊 Répartition des émotions détectées récemment")
            pie_chart = gr.Plot()
            history_display = gr.Dataframe()
            with gr.Accordion("📩 Feedback utilisateur", open=False):
                feedback = gr.Radio(["👍 Yes", "👎 No"], label="Prédiction correcte ?")
                comment = gr.Textbox(label="Commentaire optionnel")
                feedback_btn = gr.Button("✅ Envoyer Feedback")
                feedback_status = gr.Textbox(label="Statut", interactive=False)
                feedback_stats = gr.Textbox(label="Stats feedback", interactive=False)

        with gr.Tab("⚠️ Cas suspects"):
            with gr.Row():
                suspicious_input = gr.Textbox(label="🧪 Texte à auditer", lines=3, value=suspect_text)
                audit_btn = gr.Button("Vérifier")
            audit_alert = gr.Text(label="Diagnostic")
            audit_scores = gr.Label(label="Scores détectés")

        with gr.Tab("📁 Export / Logs"):
            gen_button = gr.Button("📤 Télécharger CSV émotions")
            download_btn = gr.File(label="Fichier CSV", interactive=True)

    analyze_btn.click(fn=predict_emotions, inputs=input_box, outputs=[sentiment_output, pie_chart, history_display, suspicious_input])
    random_btn.click(fn=lambda: random.choice(tweet_examples), outputs=input_box)
    audit_btn.click(fn=audit_text, inputs=suspicious_input, outputs=[audit_alert, audit_scores])
    feedback_btn.click(fn=save_feedback, inputs=[input_box, feedback, comment], outputs=[feedback_status, feedback_stats])
    gen_button.click(fn=download_emotion_stats, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()

# # Lancement de l'interface
# 
# python app.py

