# 🌤️ SkyCast - Sistema de Pronóstico Meteorológico con IA

## 📋 Descripción

SkyCast es un sistema avanzado de pronóstico meteorológico que utiliza **redes neuronales LSTM** y **ChatGPT GPT-4o-mini** para generar predicciones precisas y recomendaciones inteligentes para agricultura y transporte.

## 🚀 Instalación Rápida

### 🪟 **Windows:**

1. **Configurar entorno:**
   ```bash
   setup_environment.bat
   ```

2. **Editar configuración:**
   - Abrir `config.py`
   - Reemplazar `"tu_api_key_aqui"` con tu API key de OpenAI

3. **Ejecutar programa:**
   ```bash
   run_skycast.bat
   ```

### 🐧 **Linux/Mac:**

1. **Configurar entorno:**
   ```bash
   chmod +x setup_environment.sh
   ./setup_environment.sh
   ```

2. **Editar configuración:**
   - Abrir `config.py`
   - Reemplazar `"tu_api_key_aqui"` con tu API key de OpenAI

3. **Ejecutar programa:**
   ```bash
   chmod +x run_skycast.sh
   ./run_skycast.sh
   ```

## 🔧 Requisitos del Sistema

### **Software Requerido:**
- **Python 3.8+** (recomendado 3.9-3.11)
- **pip** (gestor de paquetes de Python)
- **Git** (opcional, para clonar repositorio)

### **APIs Necesarias:**
- **OpenAI API Key** - Para ChatGPT GPT-4o-mini
- **Meteomatics** - Ya configurado
- **Google Maps** - Ya configurado

## 📦 Dependencias

### **🤖 Inteligencia Artificial:**
- `tensorflow>=2.10.0` - Redes neuronales LSTM
- `scikit-learn>=1.0.0` - Métricas de evaluación
- `openai>=1.0.0` - ChatGPT GPT-4o-mini

### **🌐 Web Framework:**
- `flask>=2.0.0` - API REST
- `flask-cors>=3.0.0` - CORS para peticiones

### **📊 Procesamiento de Datos:**
- `pandas>=1.3.0` - Manipulación de datos
- `numpy>=1.21.0` - Operaciones numéricas
- `requests>=2.25.0` - Peticiones HTTP

## ⚙️ Configuración

### **Archivo `config.py`:**
```python
# Configuración para SkyCast API
OPENAI_API_KEY = "sk-proj-..."  # ⚠️ ACTUALIZAR
MET_USER = "linogarcia_yenso"
MET_PASS = "eBCQ7aI6MhpvMg9SCkno"
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 800
TEMPERATURE = 0.7
```

### **Obtener API Key de OpenAI:**
1. Ir a https://platform.openai.com/api-keys
2. Crear nueva API key
3. Copiar y pegar en `config.py`

## 🎯 Funcionalidades

### **🌤️ Pronóstico Meteorológico:**
- **Temperatura** (°C)
- **Humedad** (%)
- **Velocidad del Viento** (km/h)
- **Precipitaciones** (mm)

### **🤖 Recomendaciones IA:**
- **Análisis agrícola** contextual
- **Sugerencias de transporte** basadas en condiciones
- **Generado por GPT-4o-mini**

### **📈 Métricas de Evaluación:**
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coeficiente de Determinación)

### **🗺️ Interfaz Interactiva:**
- **Mapa de Google Maps** integrado
- **Selección de ubicación** interactiva
- **Resultados en tiempo real**
- **Diseño responsive**

## 🌐 Uso del Sistema

### **Interfaz Web:**
1. Abrir navegador en: http://127.0.0.1:5001
2. Seleccionar ubicación en el mapa
3. Elegir fecha de pronóstico
4. Hacer clic en "🚀 Generar Pronóstico"

### **API Endpoints:**
- **Pronóstico completo:** `GET /api/forecast?lat=-12.04318&lon=-77.02824&forecast_date=2026-02-14`
- **Solo recomendaciones:** `GET /api/recommendations?lat=-12.04318&lon=-77.02824&forecast_date=2026-02-14`
- **Estado de la API:** `GET /api/status`

## 🔧 Solución de Problemas

### **❌ Error: "Python no está instalado"**
```bash
# Windows: Descargar desde python.org
# Linux: sudo apt install python3 python3-pip
# Mac: brew install python3
```

### **❌ Error: "TensorFlow no se instaló"**
```bash
# Verificar versión de Python (3.8-3.11)
python --version
# Actualizar pip
python -m pip install --upgrade pip
# Reinstalar TensorFlow
python -m pip install tensorflow --upgrade
```

### **❌ Error: "OpenAI API Key inválida"**
- Verificar API key en `config.py`
- Comprobar cuota disponible en OpenAI
- Verificar que la key tenga permisos para GPT-4o-mini

### **❌ Error: "Puerto 5001 en uso"**
```bash
# Windows
netstat -ano | findstr :5001
taskkill /PID [NUMERO] /F

# Linux/Mac
lsof -ti:5001 | xargs kill -9
```

## 📊 Arquitectura del Sistema

### **🧠 Modelo LSTM:**
```
Input Layer (LOOKBACK_DAYS timesteps) 
    ↓
LSTM Layer (64 units)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit)
```

### **📈 Pipeline de Datos:**
1. **Descarga** de datos históricos (Meteomatics)
2. **Limpieza** y preprocesamiento
3. **Normalización** (MinMaxScaler)
4. **Ventanas supervisadas** (LOOKBACK_DAYS)
5. **Entrenamiento** del modelo LSTM
6. **Evaluación** con métricas
7. **Pronóstico** iterativo
8. **Generación** de recomendaciones (ChatGPT)

## 🎯 Ejemplos de Uso

### **Pronóstico para Lima, 14 Febrero 2026:**
```
http://127.0.0.1:5001/api/forecast?lat=-12.04318&lon=-77.02824&forecast_date=2026-02-14
```

### **Respuesta JSON:**
```json
{
  "status": "success",
  "location": {"lat": -12.04318, "lon": -77.02824},
  "forecast_date": "2026-02-14",
  "forecasts": {
    "temperature": {"values": [18.5], "unit": "°C"},
    "humidity": {"values": [75.2], "unit": "%"},
    "wind_speed": {"values": [2.1], "unit": "m/s"},
    "precipitation": {"values": [0.0], "unit": "mm/h"}
  },
  "recommendations": {
    "status": "success",
    "recommendations": "AGRICULTURA:\n- Temperatura moderada..."
  },
  "execution_time": {
    "total_seconds": 45.23,
    "start_time": "14:30:15",
    "end_time": "14:31:00"
  }
}
```

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crear rama para nueva funcionalidad
3. Commit los cambios
4. Push a la rama
5. Crear Pull Request

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear issue en GitHub
- Contactar al desarrollador
- Revisar documentación en `/docs`

---

**SkyCast** - Pronóstico meteorológico inteligente con IA 🚀
