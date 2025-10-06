# ========================================
# SKYCAST - PLANTILLA DE CONFIGURACIÓN
# ========================================
# 
# INSTRUCCIONES:
# 1. Copiar este archivo como 'config.py'
# 2. Reemplazar 'tu_api_key_aqui' con tu API key real de OpenAI
# 3. Guardar el archivo
#
# ========================================

# Configuración de OpenAI (ChatGPT)
# Obtener API key desde: https://platform.openai.com/api-keys
OPENAI_API_KEY = "tu_api_key_aqui"  # ⚠️ REEMPLAZAR CON TU API KEY REAL

# Configuración del modelo de IA
MODEL_NAME = "gpt-4o-mini"          # Modelo de ChatGPT a usar
MAX_TOKENS = 800                    # Máximo número de tokens por respuesta
TEMPERATURE = 0.7                   # Creatividad de las respuestas (0.0-1.0)

# Configuración de Meteomatics (Datos meteorológicos)
# Ya configurado - no modificar
MET_USER = "linogarcia_yenso"
MET_PASS = "eBCQ7aI6MhpvMg9SCkno"

# ========================================
# CONFIGURACIÓN AVANZADA (Opcional)
# ========================================

# Configuración del modelo LSTM
LOOKBACK_DAYS = 30                  # Días de historial para entrenar
EPOCHS = 50                         # Número de épocas de entrenamiento
BATCH_SIZE = 32                     # Tamaño del lote
VALIDATION_SPLIT = 0.2              # Porcentaje para validación

# Configuración de la API web
HOST = "127.0.0.1"                 # Dirección IP del servidor
PORT = 5001                        # Puerto del servidor
DEBUG = False                      # Modo debug (True/False)

# Configuración de Google Maps
GOOGLE_MAPS_API_KEY = "AIzaSyArLVHe9xh3ITcIVLz8_ibHpz3w_Oa7HIQ"

# ========================================
# VARIABLES METEOROLÓGICAS
# ========================================
# No modificar - configuración interna del sistema
WEATHER_VARIABLES = {
    'temperature': {
        'param': 't_2m:C',
        'unit': '°C',
        'description': 'Temperatura a 2 metros'
    },
    'humidity': {
        'param': 'relative_humidity_2m:p',
        'unit': '%',
        'description': 'Humedad relativa a 2 metros'
    },
    'wind_speed': {
        'param': 'wind_speed_10m:ms',
        'unit': 'm/s',
        'description': 'Velocidad del viento a 10 metros'
    },
    'precipitation': {
        'param': 'precip_1h:mm',
        'unit': 'mm/h',
        'description': 'Precipitación por hora'
    }
}

# ========================================
# CONFIGURACIÓN DE FECHAS
# ========================================
START_DATE = "2025-01-05"          # Fecha de inicio para datos históricos
END_DATE = "2025-01-15"            # Fecha de fin para datos históricos
TARGET_FREQ = "D"                  # Frecuencia objetivo (D=daily, H=hourly)
MET_INTERVAL = "PT1H"              # Intervalo de Meteomatics (PT1H=1 hora)

# ========================================
# DIRECTORIOS
# ========================================
MODEL_DIR = "models"               # Directorio para guardar modelos
DATA_DIR = "data"                  # Directorio para datos temporales

# ========================================
# LOGGING
# ========================================
LOG_LEVEL = "INFO"                 # Nivel de logging (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "skycast.log"           # Archivo de log (opcional)

# ========================================
# NOTAS IMPORTANTES
# ========================================
#
# 1. API KEY DE OPENAI:
#    - Obtener desde: https://platform.openai.com/api-keys
#    - Mantener segura y no compartir
#    - Verificar cuota disponible
#
# 2. CONFIGURACIÓN DE METEOmatics:
#    - Ya configurada y funcional
#    - No modificar credenciales
#
# 3. GOOGLE MAPS:
#    - API key ya configurada
#    - No requiere cambios
#
# 4. MODELO LSTM:
#    - Configuración optimizada
#    - Modificar solo si es necesario
#
# ========================================
