#!/bin/bash

echo "========================================"
echo "    SKYCAST - EJECUTAR PROGRAMA"
echo "========================================"
echo

echo "Verificando que el entorno esté configurado..."
if [ ! -f "config.py" ]; then
    echo "ERROR: Archivo config.py no encontrado"
    echo "Ejecuta primero: ./setup_environment.sh"
    exit 1
fi

echo "Verificando dependencias..."
python3 -c "import tensorflow, openai, flask, pandas, numpy, sklearn, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Dependencias no instaladas"
    echo "Ejecuta primero: ./setup_environment.sh"
    exit 1
fi

echo "Verificando API key..."
python3 -c "from config import OPENAI_API_KEY; print('API Key configurada:', 'SI' if OPENAI_API_KEY != 'tu_api_key_aqui' else 'NO - ACTUALIZAR')"
echo

echo
echo "Iniciando SkyCast API..."
echo
echo "========================================"
echo "    SKYCAST API ACTIVA"
echo "========================================"
echo
echo "URL: http://127.0.0.1:5001"
echo
echo "Presiona Ctrl+C para detener el servidor"
echo

python3 backend_lstm_clean_4vars.py
