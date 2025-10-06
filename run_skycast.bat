@echo off
echo ========================================
echo    SKYCAST - EJECUTAR PROGRAMA
echo ========================================
echo.

echo Verificando que el entorno este configurado...
if not exist "config.py" (
    echo ERROR: Archivo config.py no encontrado
    echo Ejecuta primero: setup_environment.bat
    pause
    exit /b 1
)

echo Verificando dependencias...
python -c "import tensorflow, openai, flask, pandas, numpy, sklearn, requests" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Dependencias no instaladas
    echo Ejecuta primero: setup_environment.bat
    pause
    exit /b 1
)

echo Verificando API key...
python -c "from config import OPENAI_API_KEY; print('API Key configurada:', 'SI' if OPENAI_API_KEY != 'tu_api_key_aqui' else 'NO - ACTUALIZAR')"
echo.

echo.
echo Iniciando SkyCast API...
echo.
echo ========================================
echo    SKYCAST API ACTIVA
echo ========================================
echo.
echo URL: http://127.0.0.1:5001
echo.
echo Presiona Ctrl+C para detener el servidor
echo.

python backend_lstm_clean_4vars.py
