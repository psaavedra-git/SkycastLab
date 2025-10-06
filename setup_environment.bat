@echo off
echo ========================================
echo    SKYCAST - SETUP DEL ENTORNO
echo ========================================
echo.

echo [1/6] Verificando Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado o no esta en PATH
    echo Descarga Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

echo [2/6] Actualizando pip...
python -m pip install --upgrade pip
echo.

echo [3/6] Instalando dependencias desde requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de dependencias
    echo Verifica tu conexion a internet y vuelve a intentar
    pause
    exit /b 1
)
echo.

echo [4/6] Verificando instalacion de TensorFlow...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: TensorFlow no se instalo correctamente
    echo Intenta: python -m pip install tensorflow --upgrade
    pause
    exit /b 1
)
echo.

echo [5/6] Verificando instalacion de OpenAI...
python -c "import openai; print('OpenAI version:', openai.__version__)"
if %errorlevel% neq 0 (
    echo ERROR: OpenAI no se instalo correctamente
    echo Intenta: python -m pip install openai --upgrade
    pause
    exit /b 1
)
echo.

echo [6/6] Verificando configuracion...
if exist "config.py" (
    echo ✓ Archivo config.py encontrado
) else (
    echo ⚠ Archivo config.py no encontrado - creando template...
    echo # Configuracion para SkyCast API > config.py
    echo OPENAI_API_KEY = "tu_api_key_aqui" >> config.py
    echo MET_USER = "linogarcia_yenso" >> config.py
    echo MET_PASS = "eBCQ7aI6MhpvMg9SCkno" >> config.py
    echo MODEL_NAME = "gpt-4o-mini" >> config.py
    echo MAX_TOKENS = 800 >> config.py
    echo TEMPERATURE = 0.7 >> config.py
    echo ✓ Template config.py creado
)
echo.

echo ========================================
echo    SETUP COMPLETADO EXITOSAMENTE
echo ========================================
echo.
echo Dependencias instaladas:
echo ✓ TensorFlow - Redes neuronales LSTM
echo ✓ OpenAI - ChatGPT GPT-4o-mini
echo ✓ Flask - API web
echo ✓ Pandas - Procesamiento de datos
echo ✓ NumPy - Operaciones numericas
echo ✓ Scikit-learn - Metricas de evaluacion
echo ✓ Requests - Peticiones HTTP
echo.
echo SIGUIENTE PASO: Editar config.py con tu API key de OpenAI
echo.
echo Para ejecutar el programa:
echo    run_skycast.bat
echo.
pause
