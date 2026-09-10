@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在启动 Whisper 音频转写应用（首次加载模型需要一点时间）...
"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
pause
