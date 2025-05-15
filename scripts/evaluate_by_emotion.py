# scripts/evaluate_by_emotion.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

emotion_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

print("📦 Chargement modèle + tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained("models/emotions/model")
tokenizer = AutoTokenizer.from_pretrained("models/emotions/tokenizer")

data_path = os.path.join("data", "raw", "goemotions.csv")
df = pd.read_csv(data_path)
df['labels'] = df[emotion_cols].values.tolist()

df_test = df.sample(n=2000, random_state=42).reset_index(drop=True)

dataset = Dataset.from_pandas(df_test[['text']])
dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)
dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

print("🚀 Prédiction en cours...")
outputs = model(**dataset[:])
logits = outputs.logits.detach().numpy()
preds = (logits > 0).astype(int)

y_true = np.array(df_test[emotion_cols].values)

print("📊 F1-score par émotion :")
report = classification_report(y_true, preds, target_names=emotion_cols, zero_division=0)
print(report)
