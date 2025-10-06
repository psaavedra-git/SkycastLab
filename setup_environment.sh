#!/bin/bash

echo "========================================"
echo "    SKYCAST - SETUP DEL ENTORNO"
echo "========================================"
echo

echo "[1/6] Verificando Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python no está instalado o no está en PATH"
    echo "Instala Python desde: https://www.python.org/downloads/"
    exit 1
fi
echo

echo "[2/6] Actualizando pip..."
python3 -m pip install --upgrade pip
echo

echo "[3/6] Instalando dependencias desde requirements.txt..."
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Falló la instalación de dependencias"
    echo "Verifica tu conexión a internet y vuelve a intentar"
    exit 1
fi
echo

echo "[4/6] Verificando instalación de TensorFlow..."
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
if [ $? -ne 0 ]; then
    echo "ERROR: TensorFlow no se instaló correctamente"
    echo "Intenta: python3 -m pip install tensorflow --upgrade"
    exit 1
fi
echo

echo "[5/6] Verificando instalación de OpenAI..."
python3 -c "import openai; print('OpenAI version:', openai.__version__)"
if [ $? -ne 0 ]; then
    echo "ERROR: OpenAI no se instaló correctamente"
    echo "Intenta: python3 -m pip install openai --upgrade"
    exit 1
fi
echo

echo "[6/6] Verificando configuración..."
if [ -f "config.py" ]; then
    echo "✓ Archivo config.py encontrado"
else
    echo "⚠ Archivo config.py no encontrado - creando template..."
    cat > config.py << EOF
# Configuración para SkyCast API
OPENAI_API_KEY = "tu_api_key_aqui"
MET_USER = "linogarcia_yenso"
MET_PASS = "eBCQ7aI6MhpvMg9SCkno"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 800
TEMPERATURE = 0.7
EOF
    echo "✓ Template config.py creado"
fi
echo

echo "========================================"
echo "    SETUP COMPLETADO EXITOSAMENTE"
echo "========================================"
echo
echo "Dependencias instaladas:"
echo "✓ TensorFlow - Redes neuronales LSTM"
echo "✓ OpenAI - ChatGPT GPT-4o-mini"
echo "✓ Flask - API web"
echo "✓ Pandas - Procesamiento de datos"
echo "✓ NumPy - Operaciones numéricas"
echo "✓ Scikit-learn - Métricas de evaluación"
echo "✓ Requests - Peticiones HTTP"
echo
echo "SIGUIENTE PASO: Editar config.py con tu API key de OpenAI"
echo
echo "Para ejecutar el programa:"
echo "   ./run_skycast.sh"
echo
