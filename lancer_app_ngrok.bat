@echo off
echo ✅ Activation de l’environnement virtuel...
start cmd /k "cd /d %~dp0 && venv_P9_V2\\Scripts\\activate && python app.py"

timeout /t 5 >nul

echo 🌐 Lancement de Ngrok sur le port 7860...
start cmd /k "ngrok http 7860"
