@echo off
echo 🛑 Fermeture de l'application Gradio + Ngrok...

taskkill /f /im python.exe >nul 2>&1
taskkill /f /im ngrok.exe >nul 2>&1

echo ✅ Processus terminés.
pause
