# 🚀 Guía de Instalación - SkyCast

## 📋 Instalación Paso a Paso

### **Paso 1: Verificar Requisitos**

#### **🪟 Windows:**
- Windows 10/11
- Python 3.8+ (descargar desde python.org)
- 4GB RAM mínimo, 8GB recomendado
- 2GB espacio libre en disco

#### **🐧 Linux:**
- Ubuntu 18.04+ / CentOS 7+ / Debian 10+
- Python 3.8+
- 4GB RAM mínimo, 8GB recomendado
- 2GB espacio libre en disco

#### **🍎 macOS:**
- macOS 10.15+ (Catalina o superior)
- Python 3.8+
- 4GB RAM mínimo, 8GB recomendado
- 2GB espacio libre en disco

### **Paso 2: Instalar Python**

#### **Windows:**
1. Ir a https://www.python.org/downloads/
2. Descargar Python 3.9 o 3.10
3. **IMPORTANTE:** Marcar "Add Python to PATH"
4. Instalar con configuración por defecto

#### **Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### **macOS:**
```bash
# Con Homebrew
brew install python3

# O descargar desde python.org
```

### **Paso 3: Clonar/Descargar SkyCast**

#### **Opción A: Clonar desde Git**
```bash
git clone https://github.com/tu-usuario/skycast.git
cd skycast
```

#### **Opción B: Descargar ZIP**
1. Descargar ZIP del repositorio
2. Extraer en carpeta deseada
3. Abrir terminal en esa carpeta

### **Paso 4: Configurar Entorno**

#### **🪟 Windows:**
```bash
setup_environment.bat
```

#### **🐧 Linux/macOS:**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### **Paso 5: Configurar API Keys**

#### **Editar config.py:**
```python
# Reemplazar con tu API key real
OPENAI_API_KEY = "sk-proj-tu_api_key_aqui"
```

#### **Obtener OpenAI API Key:**
1. Ir a https://platform.openai.com/api-keys
2. Crear cuenta o iniciar sesión
3. Hacer clic en "Create new secret key"
4. Copiar la key y pegarla en config.py

### **Paso 6: Ejecutar SkyCast**

#### **🪟 Windows:**
```bash
run_skycast.bat
```

#### **🐧 Linux/macOS:**
```bash
chmod +x run_skycast.sh
./run_skycast.sh
```

### **Paso 7: Acceder al Sistema**

1. Abrir navegador
2. Ir a: http://127.0.0.1:5001
3. ¡Listo para usar!

## 🔧 Instalación Manual (Alternativa)

### **Si los scripts no funcionan:**

#### **1. Instalar dependencias manualmente:**
```bash
pip install tensorflow>=2.10.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install scikit-learn>=1.0.0
pip install requests>=2.25.0
pip install flask>=2.0.0
pip install flask-cors>=3.0.0
pip install openai>=1.0.0
```

#### **2. Verificar instalación:**
```bash
python -c "import tensorflow, openai, flask, pandas, numpy, sklearn, requests; print('✓ Todas las dependencias instaladas')"
```

#### **3. Ejecutar programa:**
```bash
python backend_lstm_clean_4vars.py
```

## 🐛 Solución de Problemas Comunes

### **Error: "Python no encontrado"**
- **Windows:** Reinstalar Python marcando "Add to PATH"
- **Linux:** `sudo apt install python3`
- **macOS:** `brew install python3`

### **Error: "pip no encontrado"**
```bash
# Windows
python -m ensurepip --upgrade

# Linux
sudo apt install python3-pip

# macOS
python3 -m ensurepip --upgrade
```

### **Error: "TensorFlow no se instala"**
```bash
# Verificar versión de Python
python --version

# Actualizar pip
python -m pip install --upgrade pip

# Instalar TensorFlow específico
pip install tensorflow==2.10.0
```

### **Error: "OpenAI API Key inválida"**
1. Verificar que la key esté correcta en config.py
2. Comprobar cuota en OpenAI Dashboard
3. Verificar que la key tenga permisos para GPT-4o-mini

### **Error: "Puerto 5001 en uso"**
```bash
# Windows
netstat -ano | findstr :5001
taskkill /PID [NUMERO] /F

# Linux/macOS
lsof -ti:5001 | xargs kill -9
```

## 📊 Verificación de Instalación

### **Comando de verificación completa:**
```bash
python -c "
import tensorflow as tf
import openai
import flask
import pandas as pd
import numpy as np
import sklearn
import requests
print('✓ TensorFlow:', tf.__version__)
print('✓ OpenAI:', openai.__version__)
print('✓ Flask:', flask.__version__)
print('✓ Pandas:', pd.__version__)
print('✓ NumPy:', np.__version__)
print('✓ Scikit-learn:', sklearn.__version__)
print('✓ Requests:', requests.__version__)
print('🎉 ¡Todas las dependencias instaladas correctamente!')
"
```

## 🎯 Próximos Pasos

1. **Configurar API Key** de OpenAI
2. **Ejecutar el programa** con los scripts
3. **Acceder a la interfaz** web
4. **Probar pronóstico** para tu ubicación
5. **Explorar recomendaciones** de IA

## 📞 Soporte

Si tienes problemas con la instalación:
1. Revisar esta guía completa
2. Verificar requisitos del sistema
3. Comprobar logs de error
4. Contactar soporte técnico

---

**¡SkyCast está listo para usar!** 🌟
