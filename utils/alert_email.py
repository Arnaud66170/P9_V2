# src/utils/alert_email.py

import requests

def send_alert_email(nb_bad_feedbacks):
    """
    Envoie une alerte via l'API Railway quand trop de feedbacks émotionnels négatifs sont reçus.
    """
    try:
        response = requests.post(
            "https://web-production-662f7.up.railway.app/send-alert",
            json={"nb_feedbacks": nb_bad_feedbacks}
        )
        print(f"✅ Appel API Railway : {response.status_code} | {response.json()}")
    except Exception as e:
        print(f"❌ Échec appel API d’alerte : {e}")

# Option : envoi via SMTP Gmail (commenté)
# import os
# import smtplib
# from email.mime.text import MIMEText

# def send_alert_email(nb_bad_feedbacks):
#     smtp_server = os.getenv("EMAIL_HOST")
#     port = int(os.getenv("EMAIL_PORT", 587))
#     sender = os.getenv("EMAIL_HOST_USER")
#     receiver = os.getenv("EMAIL_RECEIVER")
#     password = os.getenv("EMAIL_HOST_PASSWORD")

#     if not all([smtp_server, port, sender, receiver, password]):
#         print("❌ Erreur : Variables d'environnement manquantes.")
#         return

#     message = MIMEText(f"🚨 Alerte : {nb_bad_feedbacks} feedbacks émotionnels négatifs reçus en <5min.")
#     message['Subject'] = "Alerte Feedback Négatif - Emotions UI"
#     message['From'] = sender
#     message['To'] = receiver

#     try:
#         with smtplib.SMTP(smtp_server, port) as server:
#             server.starttls()
#             server.login(sender, password)
#             server.send_message(message)
#         print("✅ Email d’alerte envoyé.")
#     except Exception as e:
#         print("❌ Erreur lors de l'envoi d'email :", e)
