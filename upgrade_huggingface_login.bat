@echo off
echo 🔁 Activation de l'environnement virtuel...
call .\venv_P9_V2\Scripts\activate

echo ⏫ Mise à jour du package huggingface_hub...
pip install --upgrade huggingface_hub

echo ✅ Terminé. Vous pouvez maintenant lancer :
echo huggingface-cli login
pause