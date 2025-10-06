# -*- coding: utf-8 -*-

import os
import warnings

# Suprimir warnings de TensorFlow y otras librerías
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Cambiado a 3 para suprimir más warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*HDF5.*')
warnings.filterwarnings('ignore', message='.*absl.*')
warnings.filterwarnings('ignore', message='.*legacy.*')

import tensorflow as tf

tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# Configurar logging de TensorFlow para suprimir warnings
tf.get_logger().setLevel('ERROR')

# Suprimir warnings de absl
import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import io
import math
import json
import time
import base64
import datetime as dt
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import openai

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================
# ======= CONFIG ==========
# =========================

# --- Configuración del programa ---
app = Flask(__name__)
CORS(app)

# --- Variables meteorológicas soportadas ---
WEATHER_VARIABLES = {
    'temperature': {
        'param': 't_2m:C',
        'unit': '°C',
        'description': 'xtTemperatura '
    },
    'humidity': {
        'param': 'relative_humidity_2m:p',
        'unit': '%',
        'description': 'Humedad relativa '
    },
    'wind_speed': {
        'param': 'wind_speed_10m:ms',
        'unit': 'm/s',
        'description': 'Velocidad del viento '
    },
    'precipitation': {
        'param': 'precip_1h:mm',
        'unit': 'mm/h',
        'description': 'Precipitación'
    }
}

# --- Parámetros para Meteomatics ---
MET_USER = "linogarcia_yenso"
MET_PASS = "eBCQ7aI6MhpvMg9SCkno"

# --- Parámetros para OpenAI ---
try:
    from config import OPENAI_API_KEY, MODEL_NAME, MAX_TOKENS, TEMPERATURE
except ImportError:
    OPENAI_API_KEY = "tu_api_key_aqui"  # Fallback si no se puede importar config
    MODEL_NAME = "gpt-4o-mini"
    MAX_TOKENS = 800
    TEMPERATURE = 0.7

# --- Configuración por defecto ---
TARGET_FREQ = "1D"
MET_INTERVAL = "P1D" if TARGET_FREQ.upper() == "1D" else "PT1H"
LOOKBACK_DAYS = 30
FORECAST_DAYS = 30  # Reducido para API más rápida

# --- Entrenamiento ---
EPOCHS = 100  # Reducido para API más rápida
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42
MODEL_DIR = "./models"
OUTPUT_DIR = "./outputs"

# --- Fechas por defecto ---
START_DATE = "2019-10-05"
END_DATE = "2025-10-05"

np.random.seed(RANDOM_SEED)

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_agricultural_transport_recommendations(forecast_data: Dict) -> Dict[str, str]:
    """Genera recomendaciones agrícolas y de transporte usando ChatGPT"""
    try:
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] INICIO - Generando recomendaciones con ChatGPT")
        
        # Configurar OpenAI (nueva sintaxis >=1.0.0)
        # La API key se pasa directamente al cliente
        
        # Extraer datos del pronóstico
        location = forecast_data['location']
        forecast_date = forecast_data['forecast_date']
        forecasts = forecast_data['forecasts']
        
        # Crear resumen del pronóstico
        weather_summary = f"""
        Pronóstico meteorológico para {location['lat']}, {location['lon']} el {forecast_date}:
        
        - Temperatura: {forecasts['temperature']['values'][0]:.1f}°C
        - Humedad: {forecasts['humidity']['values'][0]:.1f}%
        - Velocidad del viento: {forecasts['wind_speed']['values'][0]:.1f} m/s
        - Precipitación: {forecasts['precipitation']['values'][0]:.3f} mm/h
        """
        
        # Prompt para recomendaciones
        prompt = f"""
        Basándote en el siguiente pronóstico meteorológico, proporciona recomendaciones específicas y prácticas:

        {weather_summary}

        Por favor, proporciona recomendaciones en los siguientes formatos:

        AGRICULTURA:
        - Cultivos recomendados para sembrar
        - Actividades agrícolas recomendadas
        - Precauciones necesarias
        - Riego y fertilización

        TRANSPORTE:
        - Condiciones de viaje
        - Precauciones para transporte terrestre
        - Recomendaciones para transporte aéreo
        - Logística y planificación

        Responde en español, de manera concisa y práctica.
        """
        
        # Llamada a ChatGPT (nueva sintaxis OpenAI >=1.0.0)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Eres un experto en meteorología aplicada a agricultura y transporte. Proporcionas recomendaciones prácticas y específicas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        recommendations = response.choices[0].message.content
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] FIN - Recomendaciones generadas en {duration:.2f} segundos")
        
        return {
            'status': 'success',
            'recommendations': recommendations,
            'generation_time': duration
        }
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ERROR - Generación de recomendaciones falló después de {duration:.2f} segundos: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'fallback_recommendations': generate_fallback_recommendations(forecast_data)
        }

def generate_fallback_recommendations(forecast_data: Dict) -> str:
    """Genera recomendaciones básicas sin ChatGPT como respaldo"""
    forecasts = forecast_data['forecasts']
    temp = forecasts['temperature']['values'][0]
    humidity = forecasts['humidity']['values'][0]
    wind = forecasts['wind_speed']['values'][0]
    precip = forecasts['precipitation']['values'][0]
    
    recommendations = "AGRICULTURA:\n"
    
    # Recomendaciones agrícolas básicas
    if temp < 15:
        recommendations += "- Temperatura baja: Proteger cultivos sensibles al frío\n"
    elif temp > 25:
        recommendations += "- Temperatura alta: Aumentar riego y sombra\n"
    
    if humidity > 80:
        recommendations += "- Alta humedad: Prevenir enfermedades fúngicas\n"
    elif humidity < 50:
        recommendations += "- Baja humedad: Incrementar riego\n"
    
    if precip > 0.1:
        recommendations += "- Lluvia esperada: Evitar aplicaciones de fertilizantes\n"
    
    recommendations += "\nTRANSPORTE:\n"
    
    # Recomendaciones de transporte básicas
    if wind > 3:
        recommendations += "- Viento moderado: Precauciones en transporte aéreo\n"
    
    if precip > 0.1:
        recommendations += "- Lluvia esperada: Reducir velocidad en carreteras\n"
    
    if temp < 10:
        recommendations += "- Temperatura baja: Posible formación de hielo\n"
    
    return recommendations

# Funciones de NetCDF eliminadas - ahora solo usamos Meteomatics

def fetch_meteomatics_historical(lat: float, lon: float, start_date: str, end_date: str, 
                                target_freq: str, variable: str) -> pd.Series:
    """Descarga datos históricos de Meteomatics para una variable específica"""
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] INICIO - Descargando datos históricos para {variable}")
    
    start = pd.to_datetime(start_date).tz_localize("UTC")
    end = pd.to_datetime(end_date).tz_localize("UTC")
    
    start_str = start.isoformat().replace("+00:00", "Z")
    end_str = end.isoformat().replace("+00:00", "Z")

    base = "https://api.meteomatics.com"
    param = WEATHER_VARIABLES[variable]['param']
    url = f"{base}/{start_str}--{end_str}:{MET_INTERVAL}/{param}/{lat},{lon}/json"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching {variable}: {url}")

    auth = (MET_USER, MET_PASS)
    r = requests.get(url, auth=auth, timeout=60)

    if r.status_code != 200:
        raise RuntimeError(f"Error Meteomatics {r.status_code}: {r.text}")

    data = r.json()
    try:
        dates = data["data"][0]["coordinates"][0]["dates"]
        df = pd.DataFrame(dates)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        s = df["value"].astype(float)
        s = s.resample(target_freq).mean().dropna()
        s.name = f"{variable}_met_historical"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] FIN - Descarga de {variable} completada en {duration:.2f} segundos")
        return s
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] ERROR - Descarga de {variable} falló después de {duration:.2f} segundos")
        raise RuntimeError(f"Formato inesperado de respuesta Meteomatics para {variable}: {e}")

def fetch_all_weather_data(lat: float, lon: float, start_date: str, end_date: str, 
                          target_freq: str) -> Dict[str, pd.Series]:
    """Descarga datos históricos para todas las variables meteorológicas"""
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] INICIO - Descarga de todas las variables meteorológicas")
    
    weather_data = {}
    
    for variable in WEATHER_VARIABLES.keys():
        try:
            weather_data[variable] = fetch_meteomatics_historical(
                lat, lon, start_date, end_date, target_freq, variable
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] OK - {variable}: {len(weather_data[variable])} observaciones")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR - Error descargando {variable}: {e}")
            raise
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"[{end_time.strftime('%H:%M:%S')}] FIN - Descarga de todas las variables completada en {duration:.2f} segundos")
    return weather_data

def clean_meteomatics_data(s_met: pd.Series, target_freq: str, variable: str) -> pd.Series:
    """Limpia y procesa datos de Meteomatics"""
    # Asegurar que la serie tenga zona horaria UTC
    if s_met.index.tz is None:
        s_met = s_met.tz_localize("UTC")
    elif s_met.index.tz != "UTC":
        s_met = s_met.tz_convert("UTC")
    
    # Interpolar valores faltantes y remuestrear
    s_met = s_met.interpolate(limit_direction="both").ffill().bfill()
    s_met = s_met.resample(target_freq).mean().dropna()
    s_met.name = variable
    return s_met

def clean_all_weather_data(weather_data: Dict[str, pd.Series], target_freq: str) -> Dict[str, pd.Series]:
    """Limpia y procesa todos los datos meteorológicos"""
    cleaned_data = {}
    
    for variable, series in weather_data.items():
        cleaned_data[variable] = clean_meteomatics_data(series, target_freq, variable)
    
    return cleaned_data

def make_supervised(series: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(input_timesteps: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_timesteps, 1), return_sequences=False))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula las métricas de evaluación del modelo"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4)
    }

def iterative_forecast(model, last_window_scaled: np.ndarray, scaler: MinMaxScaler, steps: int) -> pd.Series:
    preds = []
    window = last_window_scaled.copy().reshape(1, -1, 1)

    for _ in range(steps):
        yhat_scaled = model.predict(window, verbose=0).flatten()[0]
        yhat = scaler.inverse_transform(np.array([[yhat_scaled]]))[0, 0]
        preds.append(yhat)

        yhat_scaled_for_window = scaler.transform(np.array([[yhat]]))[0, 0]
        new_window = np.append(window.flatten()[1:], yhat_scaled_for_window)
        window = new_window.reshape(1, -1, 1)

    return pd.Series(preds)

def train_models_for_location(lat: float, lon: float, start_date: str, end_date: str) -> Dict[str, Dict]:
    """Entrena modelos para todas las variables meteorológicas en una ubicación específica"""
    start_time = datetime.now()
    print(f"[{start_time.strftime('%H:%M:%S')}] === INICIO ENTRENAMIENTO MODELOS PARA {lat}, {lon} ===")
    
    # Descargar y limpiar datos
    weather_data = fetch_all_weather_data(lat, lon, start_date, end_date, TARGET_FREQ)
    cleaned_data = clean_all_weather_data(weather_data, TARGET_FREQ)
    
    models_info = {}
    
    for variable, series in cleaned_data.items():
        variable_start_time = datetime.now()
        print(f"[{variable_start_time.strftime('%H:%M:%S')}] [TRAIN] INICIO - Entrenando modelo para {variable}")
        
        try:
            # Preparar datos
            values = series.values.astype("float32").reshape(-1, 1)
            
            # Escalado
            scaler = MinMaxScaler()
            values_scaled = scaler.fit_transform(values).flatten()
            
            # Crear ventanas supervisadas
            X, y = make_supervised(values_scaled, LOOKBACK_DAYS, horizon=1)

            # Construir modelo
            model = build_lstm(LOOKBACK_DAYS)

        # Callbacks
            model_name = f"lstm_{variable}_{lat}_{lon}_{start_date}_{end_date}.h5"
            model_path = os.path.join(MODEL_DIR, model_name)
            cbs = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
                ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
            ]
            
            # Entrenar
            history = model.fit(
                X, y,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                shuffle=True,
                verbose=0,  # Sin salida para API
                callbacks=cbs
            )
            
            # Calcular métricas de evaluación
            y_pred = model.predict(X, verbose=0).flatten()
            y_true_original = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            metrics = calculate_metrics(y_true_original, y_pred_original)
            
            # Guardar información del modelo
            models_info[variable] = {
                'model_path': model_path,
                'scaler': scaler,
                'last_window': values_scaled[-LOOKBACK_DAYS:],
                'variable_info': WEATHER_VARIABLES[variable],
                'metrics': metrics,
                'training_completed': True
            }
            
            variable_end_time = datetime.now()
            variable_duration = (variable_end_time - variable_start_time).total_seconds()
            print(f"[{variable_end_time.strftime('%H:%M:%S')}] [TRAIN] FIN - Modelo {variable} entrenado y guardado en {variable_duration:.2f} segundos")
            
        except Exception as e:
            variable_end_time = datetime.now()
            variable_duration = (variable_end_time - variable_start_time).total_seconds()
            print(f"[{variable_end_time.strftime('%H:%M:%S')}] [TRAIN] ERROR - Error entrenando modelo {variable} después de {variable_duration:.2f} segundos: {e}")
            models_info[variable] = {
                'error': str(e),
                'training_completed': False
            }
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print(f"[{end_time.strftime('%H:%M:%S')}] === FIN ENTRENAMIENTO MODELOS - Tiempo total: {total_duration:.2f} segundos ===")
    return models_info

def generate_forecast(lat: float, lon: float, forecast_date: str, days_ahead: int = 7) -> Dict:
    """Genera pronóstico para una ubicación y fecha específica"""
    try:
        process_start_time = datetime.now()
        print(f"[{process_start_time.strftime('%H:%M:%S')}] === INICIO PROCESO COMPLETO DE PRONÓSTICO ===")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Entrenando modelos para la ubicación...")
        
        # Entrenar modelos directamente
        models_info = train_models_for_location(lat, lon, START_DATE, END_DATE)
        
        # Generar pronósticos
        forecast_start_time = datetime.now()
        print(f"[{forecast_start_time.strftime('%H:%M:%S')}] INICIO - Generación de pronósticos")
        forecast_date = pd.to_datetime(forecast_date)
        forecasts = {}
        
        for variable, info in models_info.items():
            if info.get('training_completed', False) and 'error' not in info:
                try:
                    # Usar el modelo entrenado
                    model = load_model(info['model_path'], compile=False)
                    scaler = info['scaler']
                    last_window = info['last_window']
                    
                    # Generar pronóstico
                    preds = iterative_forecast(model, last_window, scaler, days_ahead)
                    
                    # Crear fechas
                    start_date = forecast_date
                    dates = pd.date_range(start_date, periods=days_ahead, freq="D")
                    
                    # Formatear resultado
                    forecasts[variable] = {
                        'values': preds.tolist(),
                        'dates': [d.strftime('%Y-%m-%d') for d in dates],
                        'unit': WEATHER_VARIABLES[variable]['unit'],
                        'description': WEATHER_VARIABLES[variable]['description'],
                        'metrics': info.get('metrics', {}),
                        'status': 'success'
                    }
                    
                except Exception as e:
                    print(f"Error generando pronóstico para {variable}: {e}")
                    forecasts[variable] = {
                        'error': str(e),
                        'status': 'error'
                    }
            else:
                error_msg = info.get('error', 'Modelo no disponible')
                print(f"Error con modelo {variable}: {error_msg}")
                forecasts[variable] = {
                    'error': error_msg,
                    'status': 'error'
                }
        
        forecast_end_time = datetime.now()
        forecast_duration = (forecast_end_time - forecast_start_time).total_seconds()
        print(f"[{forecast_end_time.strftime('%H:%M:%S')}] FIN - Generación de pronósticos completada en {forecast_duration:.2f} segundos")
        
        # Generar recomendaciones con ChatGPT (ACTIVADO)
        recommendations_start_time = datetime.now()
        print(f"[{recommendations_start_time.strftime('%H:%M:%S')}] INICIO - Generando recomendaciones con ChatGPT (GPT-4o-mini)")
        
        # Crear datos del pronóstico para las recomendaciones
        forecast_data = {
            'location': {'lat': lat, 'lon': lon},
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'forecasts': forecasts
        }
        
        # Usar ChatGPT para generar recomendaciones
        recommendations = generate_agricultural_transport_recommendations(forecast_data)
        
        recommendations_end_time = datetime.now()
        recommendations_duration = (recommendations_end_time - recommendations_start_time).total_seconds()
        print(f"[{recommendations_end_time.strftime('%H:%M:%S')}] FIN - Recomendaciones ChatGPT generadas en {recommendations_duration:.2f} segundos")
        
        process_end_time = datetime.now()
        total_process_duration = (process_end_time - process_start_time).total_seconds()
        print(f"[{process_end_time.strftime('%H:%M:%S')}] === FIN PROCESO COMPLETO - TIEMPO TOTAL: {total_process_duration:.2f} segundos ===")
        
        return {
            'location': {'lat': lat, 'lon': lon},
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'days_ahead': days_ahead,
            'forecasts': forecasts,
            'recommendations': recommendations,
            'status': 'success',
            'execution_time': {
                'total_seconds': total_process_duration,
                'start_time': process_start_time.strftime('%H:%M:%S'),
                'end_time': process_end_time.strftime('%H:%M:%S')
            }
        }
        
    except Exception as e:
        process_end_time = datetime.now()
        total_process_duration = (process_end_time - process_start_time).total_seconds()
        print(f"[{process_end_time.strftime('%H:%M:%S')}] ERROR GENERAL después de {total_process_duration:.2f} segundos: {e}")
        return {
            'error': str(e),
            'status': 'error',
            'execution_time': {
                'total_seconds': total_process_duration,
                'start_time': process_start_time.strftime('%H:%M:%S'),
                'end_time': process_end_time.strftime('%H:%M:%S')
            }
        }

def save_forecast_to_csv(forecast_result: Dict, output_file: str = None) -> str:
    """Guarda el pronóstico de una fecha específica en un archivo CSV"""
    if forecast_result['status'] != 'success':
        raise Exception(f"Error en el pronóstico: {forecast_result.get('error', 'Error desconocido')}")
    
    # Crear DataFrame con los datos del pronóstico
    forecast_date = forecast_result['forecast_date']
    forecasts = forecast_result['forecasts']
    
    # Obtener el valor del primer día (fecha específica)
    csv_data = {
        'fecha': [forecast_date],
        'latitud': [forecast_result['location']['lat']],
        'longitud': [forecast_result['location']['lon']],
        'temperatura_c': [None],
        'humedad_porcentaje': [None],
        'velocidad_viento_ms': [None],
        'precipitacion_mmh': [None],
        'temp_mse': [None],
        'temp_rmse': [None],
        'temp_mae': [None],
        'temp_r2': [None],
        'hum_mse': [None],
        'hum_rmse': [None],
        'hum_mae': [None],
        'hum_r2': [None],
        'wind_mse': [None],
        'wind_rmse': [None],
        'wind_mae': [None],
        'wind_r2': [None],
        'precip_mse': [None],
        'precip_rmse': [None],
        'precip_mae': [None],
        'precip_r2': [None]
    }
    
    # Llenar los valores de las variables meteorológicas y métricas
    for variable, forecast in forecasts.items():
        if forecast['status'] == 'success' and len(forecast['values']) > 0:
            value = forecast['values'][0]  # Primer valor (fecha específica)
            metrics = forecast.get('metrics', {})
            
            if variable == 'temperature':
                csv_data['temperatura_c'][0] = round(value, 2)
                csv_data['temp_mse'][0] = metrics.get('MSE', None)
                csv_data['temp_rmse'][0] = metrics.get('RMSE', None)
                csv_data['temp_mae'][0] = metrics.get('MAE', None)
                csv_data['temp_r2'][0] = metrics.get('R2', None)
            elif variable == 'humidity':
                csv_data['humedad_porcentaje'][0] = round(value, 2)
                csv_data['hum_mse'][0] = metrics.get('MSE', None)
                csv_data['hum_rmse'][0] = metrics.get('RMSE', None)
                csv_data['hum_mae'][0] = metrics.get('MAE', None)
                csv_data['hum_r2'][0] = metrics.get('R2', None)
            elif variable == 'wind_speed':
                csv_data['velocidad_viento_ms'][0] = round(value, 2)
                csv_data['wind_mse'][0] = metrics.get('MSE', None)
                csv_data['wind_rmse'][0] = metrics.get('RMSE', None)
                csv_data['wind_mae'][0] = metrics.get('MAE', None)
                csv_data['wind_r2'][0] = metrics.get('R2', None)
            elif variable == 'precipitation':
                csv_data['precipitacion_mmh'][0] = round(value, 2)
                csv_data['precip_mse'][0] = metrics.get('MSE', None)
                csv_data['precip_rmse'][0] = metrics.get('RMSE', None)
                csv_data['precip_mae'][0] = metrics.get('MAE', None)
                csv_data['precip_r2'][0] = metrics.get('R2', None)
    
    # Crear DataFrame
    df = pd.DataFrame(csv_data)
    
    # Generar nombre del archivo si no se proporciona
    if output_file is None:
        lat = forecast_result['location']['lat']
        lon = forecast_result['location']['lon']
        date_str = forecast_date.replace('-', '')
        output_file = f"pronostico_{lat}_{lon}_{date_str}.csv"
    
    # Guardar CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    return output_file

def parse_arguments():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='SkyCast - Pronóstico Meteorológico con LSTM')
    parser.add_argument('--lat', type=float, required=True, help='Latitud de la ubicación (-90 a 90)')
    parser.add_argument('--lon', type=float, required=True, help='Longitud de la ubicación (-180 a 180)')
    parser.add_argument('--fecha', type=str, required=True, help='Fecha de pronóstico (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Nombre del archivo CSV de salida (opcional)')
    
    return parser.parse_args()
    
def validate_arguments(args):
    """Valida los argumentos de entrada"""
    if not (-90 <= args.lat <= 90):
        raise ValueError('Latitud debe estar entre -90 y 90')
    if not (-180 <= args.lon <= 180):
        raise ValueError('Longitud debe estar entre -180 y 180')
    
    try:
        pd.to_datetime(args.fecha)
    except:
        raise ValueError('Fecha debe estar en formato YYYY-MM-DD')
    
# =========================
# ======= API ROUTES ======
# =========================

@app.route('/')
def home():
    """Página principal con interfaz web"""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SkyCast - Pronóstico Meteorológico</title>
        <script async defer src="https://maps.googleapis.com/maps/api/js?key=AIzaSyArLVHe9xh3ITcIVLz8_ibHpz3w_Oa7HIQ&callback=initMap"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                min-height: 100vh;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .form-container { 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                margin-bottom: 30px; 
                backdrop-filter: blur(10px); 
                border: 1px solid rgba(255,255,255,0.2);
            }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { 
                width: 100%; 
                padding: 12px; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px; 
                background: rgba(255,255,255,0.9);
                color: #333;
                box-sizing: border-box;
            }
            button { 
                background: #4CAF50; 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px; 
                font-weight: bold;
                transition: background 0.3s;
            }
            button:hover { background: #45a049; }
            .results { 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                margin-top: 20px; 
                backdrop-filter: blur(10px); 
                border: 1px solid rgba(255,255,255,0.2);
            }
            .weather-grid {
                display: flex;
                gap: 15px;
                margin: 20px 0;
                flex-wrap: wrap;
            }

            .weather-card { 
                background: #ffffff; 
                flex: 1;
                min-width: 200px;
                padding: 25px; 
                border-radius: 20px; 
                border: 2px solid #e0e0e0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                position: relative;
                min-height: 180px;
                text-align: left;
            }

            .weather-card.temperature {
                background: #fdf2f8;
                border-color: #e91e63;
            }

            .weather-card.humidity {
                background: #f0f9ff;
                border-color: #2196f3;
            }

            .weather-card.precipitation {
                background: #f5f5f5;
                border-color: #757575;
            }

            .weather-card.wind {
                background: #e3f2fd;
                border-color: #1976d2;
            }

            .weather-title {
                font-size: 0.9rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 15px;
                letter-spacing: 0.5px;
            }

            .weather-title.temperature {
                color: #000000;
            }

            .weather-title.humidity {
                color: #000000;
            }

            .weather-title.precipitation {
                color: #000000;
            }

            .weather-title.wind {
                color: #000000;
            }

            .weather-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 20px;
                line-height: 1;
            }

            .weather-value.temperature {
                color: #000000;
            }

            .weather-value.humidity {
                color: #000000;
            }

            .weather-value.precipitation {
                color: #000000;
            }

            .weather-value.wind {
                color: #000000;
            }

            .weather-icon {
                position: absolute;
                top: 20px;
                right: 20px;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
            }

            .weather-icon.temperature {
                background: #fdf2f8;
                color: #000000;
            }

            .weather-icon.humidity {
                background: #fdf2f8;
                color: #000000;
            }

            .weather-icon.precipitation {
                background: #fdf2f8;
                color: #000000;
            }

            .weather-icon.wind {
                background: #fdf2f8;
                color: #000000;
            }

            @media (max-width: 768px) {
                .weather-grid {
                    flex-direction: column;
                }
                .weather-card {
                    min-width: auto;
                }
            }
            .loading { text-align: center; font-size: 18px; }
            .error { 
                background: rgba(255,0,0,0.3); 
                padding: 15px; 
                border-radius: 8px; 
                margin: 10px 0; 
                border: 1px solid rgba(255,0,0,0.5);
            }
            .metrics { 
                background: rgba(255,255,255,0.1); 
                padding: 15px; 
                border-radius: 8px; 
                margin-top: 10px; 
            }
            .metric-item { 
                display: inline-block; 
                margin: 5px 15px 5px 0; 
                font-size: 14px; 
            }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            @media (max-width: 768px) {
                .grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌤️ SkyCast</h1>
                <p>Pronóstico Meteorológico Inteligente con LSTM</p>
            </div>
            
            <div class="form-container">
                <h3>📍 Selecciona Ubicación</h3>
                
                <!-- Campo de búsqueda de ubicación -->
                <div class="form-group" style="margin-bottom: 20px;">
                    <label for="locationSearch">Buscar ubicación:</label>
                    <div style="display: flex; gap: 10px;">
                        <input type="text" id="locationSearch" placeholder="Ej: Lima, Perú o Miraflores, Lima" 
                               style="flex: 1; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px;">
                        <button type="button" id="searchBtn" style="padding: 12px 20px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer;">
                            🔍 Buscar
                        </button>
                    </div>
                </div>
                
                <!-- Mapa -->
                <div id="map" style="width: 100%; height: 400px; margin-bottom: 20px; border-radius: 10px;"></div>
                
                <!-- Información de ubicación seleccionada -->
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <strong>Ubicación seleccionada:</strong>
                    <div id="selectedLocation">Haz clic en el mapa o busca una ubicación</div>
                </div>
                
                <form id="forecastForm">
                    <input type="hidden" id="lat" name="lat" value="-12.04318">
                    <input type="hidden" id="lon" name="lon" value="-77.02824">
                    
                    <div style="display: flex; gap: 15px; align-items: end;">
                        <div class="form-group" style="flex: 1; margin-bottom: 0;">
                            <label for="forecast_date">Fecha de Pronóstico:</label>
                            <input type="date" id="forecast_date" name="forecast_date" required style="width: 100%; height: 48px; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; box-sizing: border-box;">
                        </div>
                        <button type="submit" id="submitBtn" disabled style="height: 48px; padding: 12px 25px; background: #667eea; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; white-space: nowrap; box-sizing: border-box;">
                            🚀 Generar Pronóstico
                        </button>
                    </div>
                </form>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <h3>Resultados del Pronóstico</h3>
                <div id="resultsContent"></div>
            </div>
        </div>

        <script>
            // Variables globales para el mapa
            let map;
            let marker;
            let selectedLocation = { lat: -12.04318, lng: -77.02824 }; // Lima por defecto
            let geocoder;
            let currentAddress = "Lima, Perú"; // Dirección por defecto
            
            // Establecer fecha por defecto como hoy
            document.getElementById('forecast_date').value = new Date().toISOString().split('T')[0];
            
            // Inicializar campo de búsqueda
            document.getElementById('locationSearch').value = currentAddress;
            
            // Inicializar el mapa
            function initMap() {
                map = new google.maps.Map(document.getElementById('map'), {
                    center: selectedLocation,
                    zoom: 10,
                    mapTypeId: google.maps.MapTypeId.ROADMAP,
                    styles: [
                        {
                            "featureType": "all",
                            "elementType": "geometry.fill",
                            "stylers": [{"weight": "2.00"}]
                        },
                        {
                            "featureType": "all",
                            "elementType": "geometry.stroke",
                            "stylers": [{"color": "#9c9c9c"}]
                        },
                        {
                            "featureType": "all",
                            "elementType": "labels.text",
                            "stylers": [{"visibility": "on"}]
                        }
                    ]
                });
                
                // Inicializar geocoder
                geocoder = new google.maps.Geocoder();
                
                // Crear marcador inicial
                marker = new google.maps.Marker({
                    position: selectedLocation,
                    map: map,
                    title: 'Ubicación seleccionada',
                    draggable: true
                });
                
                // Actualizar ubicación al hacer clic en el mapa
                map.addListener('click', function(event) {
                    selectedLocation = {
                        lat: event.latLng.lat(),
                        lng: event.latLng.lng()
                    };
                    
                    marker.setPosition(selectedLocation);
                    updateLocationInfo();
                });
                
                // Actualizar ubicación al arrastrar el marcador
                marker.addListener('dragend', function(event) {
                    selectedLocation = {
                        lat: event.latLng.lat(),
                        lng: event.latLng.lng()
                    };
                    updateLocationInfo();
                });
                
                // Inicializar información de ubicación
                updateLocationInfo();
                
                // Configurar botón de búsqueda
                document.getElementById('searchBtn').addEventListener('click', searchLocation);
                
                // Configurar búsqueda con Enter
                document.getElementById('locationSearch').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        searchLocation();
                    }
                });
            }
            
            // Actualizar información de ubicación
            function updateLocationInfo() {
                document.getElementById('lat').value = selectedLocation.lat.toFixed(6);
                document.getElementById('lon').value = selectedLocation.lng.toFixed(6);
                
                // Usar geocodificación inversa para obtener dirección
                geocoder.geocode({ location: selectedLocation }, (results, status) => {
                    if (status === 'OK' && results[0]) {
                        const address = results[0].formatted_address;
                        currentAddress = address;
                        document.getElementById('locationSearch').value = address;
                        document.getElementById('selectedLocation').innerHTML = 
                            `<strong>Coordenadas:</strong> ${selectedLocation.lat.toFixed(6)}, ${selectedLocation.lng.toFixed(6)}<br>
                             <strong>Dirección:</strong> ${address}`;
                    } else {
                        document.getElementById('selectedLocation').innerHTML = 
                            `<strong>Coordenadas:</strong> ${selectedLocation.lat.toFixed(6)}, ${selectedLocation.lng.toFixed(6)}`;
                    }
                });
                
                // Habilitar botón de envío
                document.getElementById('submitBtn').disabled = false;
            }
            
            // Función para buscar ubicación
            function searchLocation() {
                const address = document.getElementById('locationSearch').value;
                if (address.trim() === '') {
                    alert('Por favor ingresa una ubicación');
                    return;
                }
                
                geocoder.geocode({ address: address }, (results, status) => {
                    if (status === 'OK' && results[0]) {
                        const location = results[0].geometry.location;
                        selectedLocation = {
                            lat: location.lat(),
                            lng: location.lng()
                        };
                        
                        // Actualizar mapa y marcador
                        map.setCenter(selectedLocation);
                        map.setZoom(15);
                        marker.setPosition(selectedLocation);
                        
                        // Actualizar información
                        updateLocationInfo();
                    } else {
                        alert('No se pudo encontrar la ubicación: ' + address);
                    }
                });
            }
            
            document.getElementById('forecastForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const params = new URLSearchParams();
                
                for (let [key, value] of formData.entries()) {
                    params.append(key, value);
                }
                
                const resultsDiv = document.getElementById('results');
                const resultsContent = document.getElementById('resultsContent');
                
                resultsDiv.style.display = 'block';
                resultsContent.innerHTML = '<div class="loading">⏳ Generando pronóstico... Esto puede tomar varios minutos.</div>';
                
                try {
                    const response = await fetch('/api/forecast?' + params.toString());
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displayResults(data);
                    } else {
                        resultsContent.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                    }
                } catch (error) {
                    resultsContent.innerHTML = `<div class="error">❌ Error de conexión: ${error.message}</div>`;
                }
            });
            
            function displayResults(data) {
                let html = '<div style="display: flex; gap: 15px; width: 100%;">';
                
                const variables = ['temperature', 'humidity', 'wind_speed', 'precipitation'];
                const variableNames = {
                    'temperature': '🌡️ Temperatura',
                    'humidity': '💧 Humedad',
                    'wind_speed': '💨 Velocidad del Viento',
                    'precipitation': '🌧️ Precipitación'
                };
                const variableUnits = {
                    'temperature': '°C',
                    'humidity': '%',
                    'wind_speed': 'm/s',
                    'precipitation': 'mm/h'
                };
                
                variables.forEach(variable => {
                    const forecast = data.forecasts[variable];
                    if (forecast.status === 'success') {
                        const value = forecast.values[0];
                        
                        // Mapear variables a nombres en español
                        const spanishNames = {
                            'temperature': 'TEMPERATURA',
                            'humidity': 'HUMEDAD',
                            'wind_speed': 'VIENTO',
                            'precipitation': 'PRECIPITACIONES'
                        };

                        // Mapear unidades
                        const spanishUnits = {
                            'temperature': '°C',
                            'humidity': '%',
                            'wind_speed': 'km/h',
                            'precipitation': 'mm'
                        };

                        // Convertir velocidad del viento de m/s a km/h
                        let displayValue = value;
                        if (variable === 'wind_speed') {
                            displayValue = (value * 3.6).toFixed(1); // m/s a km/h
                        } else if (variable === 'precipitation') {
                            displayValue = value.toFixed(1); // mm/h a mm
                        } else {
                            displayValue = value.toFixed(0); // Enteros para temp y humedad
                        }
                        
                        html += `
                            <div style="background: #ffffff; padding: 20px; flex: 1; border-radius: 10px; border: 1px solid #ddd; text-align: center; min-width: 0;">
                                <div style="font-size: 1rem; font-weight: 600; color: #000; margin-bottom: 10px; text-transform: uppercase;">
                                    ${spanishNames[variable]}
                                </div>
                                <div style="font-size: 2rem; font-weight: bold; color: #000;">
                                    ${displayValue} ${spanishUnits[variable]}
                                </div>
                            </div>
                        `;
                    } else {
                        html += `
                            <div style="background: #ffffff; padding: 20px; flex: 1; border-radius: 10px; border: 1px solid #ddd; text-align: center; min-width: 0;">
                                <div style="font-size: 1rem; font-weight: 600; color: #000; margin-bottom: 10px; text-transform: uppercase;">
                                    ${spanishNames[variable]}
                                </div>
                                <div style="color: #ff0000; font-size: 1.1rem;">❌ Error: ${forecast.error}</div>
                            </div>
                        `;
                    }
                });
                
                html += '</div>';
                
                // Agregar recomendaciones si están disponibles
                if (data.recommendations) {
                    const recommendations = data.recommendations;
                    if (recommendations.status === 'success') {
                        const isGPTActive = !recommendations.note || !recommendations.note.includes('desactivado');
                        const title = isGPTActive ? '🤖 Recomendaciones IA (ChatGPT)' : '📋 Recomendaciones Básicas';
                        const subtitle = isGPTActive ? 
                            `Generado en ${recommendations.generation_time || 'N/A'} segundos` :
                            'Recomendaciones básicas - ChatGPT desactivado temporalmente';
                        
                        html += `
                            <div class="weather-card" style="margin-top: 30px; text-align: left; padding: 25px;">
                                <h4 style="font-size: 1.4rem; margin-bottom: 20px; color: #333;">${title}</h4>
                                <div style="white-space: pre-line; font-size: 16px; line-height: 1.8; color: #000;">
                                    ${recommendations.recommendations}
                                </div>
                                <div style="margin-top: 15px; font-size: 13px; color: #666;">
                                    ${subtitle}
                                </div>
                            </div>
                        `;
                    } else if (recommendations.fallback_recommendations) {
                        html += `
                            <div class="weather-card" style="margin-top: 30px; text-align: left; padding: 25px;">
                                <h4 style="font-size: 1.4rem; margin-bottom: 20px; color: #333;">📋 Recomendaciones Básicas</h4>
                                <div style="white-space: pre-line; font-size: 16px; line-height: 1.8; color: #000;">
                                    ${recommendations.fallback_recommendations}
                                </div>
                                <div style="margin-top: 15px; font-size: 13px; color: #666;">
                                    (Recomendaciones básicas - ChatGPT no disponible)
                                </div>
                            </div>
                        `;
                    }
                }
                
                // Agregar información de parámetros al final
                const execTime = data.execution_time || {};
                html += `
                    <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
                        <h4 style="font-size: 1.2rem; margin-bottom: 15px; color: #fff;">📊 Parámetros del Pronóstico</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; font-size: 14px; color: #e0e0e0;">
                            <div><strong>📍 Ubicación:</strong> ${data.location.lat.toFixed(6)}, ${data.location.lon.toFixed(6)}</div>
                            <div><strong>📅 Fecha:</strong> ${data.forecast_date}</div>
                            <div><strong>⏱️ Tiempo total:</strong> ${execTime.total_seconds ? execTime.total_seconds.toFixed(2) + ' segundos' : 'N/A'}</div>
                            <div><strong>🕐 Inicio:</strong> ${execTime.start_time || 'N/A'}</div>
                            <div><strong>🕐 Fin:</strong> ${execTime.end_time || 'N/A'}</div>
                            <div><strong>🤖 Modelo:</strong> LSTM con 4 variables</div>
                        </div>
                    </div>
                `;
                
                resultsContent.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """)

@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    """Endpoint API para generar pronósticos"""
    api_start_time = datetime.now()
    print(f"[{api_start_time.strftime('%H:%M:%S')}] === INICIO PETICIÓN API ===")
    
    try:
        # Obtener parámetros
        lat = float(request.args.get('lat', -12.04318))
        lon = float(request.args.get('lon', -77.02824))
        forecast_date = request.args.get('forecast_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        days_ahead = 1  # Siempre 1 día para la fecha específica seleccionada
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Parámetros recibidos: lat={lat}, lon={lon}, fecha={forecast_date}")
        
        # Validar parámetros
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Latitud debe estar entre -90 y 90', 'status': 'error'})
        if not (-180 <= lon <= 180):
            return jsonify({'error': 'Longitud debe estar entre -180 y 180', 'status': 'error'})
        
        # Generar pronóstico
        result = generate_forecast(lat, lon, forecast_date, days_ahead)
        
        api_end_time = datetime.now()
        api_duration = (api_end_time - api_start_time).total_seconds()
        print(f"[{api_end_time.strftime('%H:%M:%S')}] === FIN PETICIÓN API - Tiempo total: {api_duration:.2f} segundos ===")
        
        return jsonify(result)
        
    except ValueError as e:
        api_end_time = datetime.now()
        api_duration = (api_end_time - api_start_time).total_seconds()
        print(f"[{api_end_time.strftime('%H:%M:%S')}] ERROR PARÁMETROS después de {api_duration:.2f} segundos: {e}")
        return jsonify({'error': f'Error en parámetros: {str(e)}', 'status': 'error'})
    except Exception as e:
        api_end_time = datetime.now()
        api_duration = (api_end_time - api_start_time).total_seconds()
        print(f"[{api_end_time.strftime('%H:%M:%S')}] ERROR INTERNO después de {api_duration:.2f} segundos: {e}")
        return jsonify({'error': f'Error interno: {str(e)}', 'status': 'error'})

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """Endpoint para generar solo recomendaciones basadas en pronóstico existente"""
    api_start_time = datetime.now()
    print(f"[{api_start_time.strftime('%H:%M:%S')}] === INICIO PETICIÓN RECOMENDACIONES ===")
    
    try:
        # Obtener parámetros
        lat = float(request.args.get('lat', -12.04318))
        lon = float(request.args.get('lon', -77.02824))
        forecast_date = request.args.get('forecast_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        days_ahead = 1  # Siempre 1 día para la fecha específica seleccionada
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Generando recomendaciones para: lat={lat}, lon={lon}, fecha={forecast_date}")
        
        # Generar pronóstico completo primero
        forecast_result = generate_forecast(lat, lon, forecast_date, days_ahead)
        
        if forecast_result['status'] == 'success':
            recommendations = forecast_result.get('recommendations', {})
            api_end_time = datetime.now()
            api_duration = (api_end_time - api_start_time).total_seconds()
            print(f"[{api_end_time.strftime('%H:%M:%S')}] === FIN PETICIÓN RECOMENDACIONES - Tiempo total: {api_duration:.2f} segundos ===")
            
            return jsonify({
                'status': 'success',
                'location': forecast_result['location'],
                'forecast_date': forecast_result['forecast_date'],
                'recommendations': recommendations,
                'execution_time': {
                    'total_seconds': api_duration,
                    'start_time': api_start_time.strftime('%H:%M:%S'),
                    'end_time': api_end_time.strftime('%H:%M:%S')
                }
            })
        else:
            return jsonify(forecast_result)
            
    except Exception as e:
        api_end_time = datetime.now()
        api_duration = (api_end_time - api_start_time).total_seconds()
        print(f"[{api_end_time.strftime('%H:%M:%S')}] ERROR RECOMENDACIONES después de {api_duration:.2f} segundos: {e}")
        return jsonify({'error': f'Error generando recomendaciones: {str(e)}', 'status': 'error'})

@app.route('/api/status', methods=['GET'])
def api_status():
    """Endpoint para verificar el estado de la API"""
    return jsonify({
        'status': 'active',
        'version': '1.0',
        'supported_variables': list(WEATHER_VARIABLES.keys()),
        'endpoints': {
            'forecast': '/api/forecast',
            'recommendations': '/api/recommendations',
            'status': '/api/status'
        },
        'message': 'SkyCast API funcionando correctamente'
    })

def main():
    """Función principal que soporta tanto CLI como API web"""
    import sys
    
    # Verificar si se están pasando argumentos de línea de comandos
    if len(sys.argv) > 1:
        # Modo CLI
        print("=== SKYCAST - PRONÓSTICO METEOROLÓGICO (CLI) ===")
        
        try:
            # Parsear argumentos
            args = parse_arguments()
            
            # Validar argumentos
            validate_arguments(args)
            
            print(f"Parámetros de entrada:")
            print(f"  Latitud: {args.lat}")
            print(f"  Longitud: {args.lon}")
            print(f"  Fecha de pronóstico: {args.fecha}")
            print(f"  Archivo de salida: {args.output or 'automático'}")
        
            # Crear directorios necesarios
            ensure_dirs()
            print("OK - Directorios creados")
            
            # Generar pronóstico
            cli_start_time = datetime.now()
            print(f"\n[{cli_start_time.strftime('%H:%M:%S')}] === GENERANDO PRONÓSTICO (CLI) ===")
            print("Esto puede tomar varios minutos la primera vez...")
            
            forecast_result = generate_forecast(args.lat, args.lon, args.fecha, days_ahead=1)
            
            if forecast_result['status'] == 'success':
                cli_end_time = datetime.now()
                cli_duration = (cli_end_time - cli_start_time).total_seconds()
                print(f"[{cli_end_time.strftime('%H:%M:%S')}] OK - Pronóstico generado exitosamente")
                
                # Guardar en CSV
                output_file = save_forecast_to_csv(forecast_result, args.output)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] OK - Resultado guardado en: {output_file}")
                
                # Mostrar resumen
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === RESUMEN DEL PRONÓSTICO ===")
                print(f"Ubicación: {args.lat}, {args.lon}")
                print(f"Fecha: {args.fecha}")
                
                # Mostrar tiempo de ejecución
                exec_time = forecast_result.get('execution_time', {})
                if exec_time:
                    print(f"Tiempo de ejecución: {exec_time.get('total_seconds', cli_duration):.2f} segundos")
                    print(f"Inicio: {exec_time.get('start_time', cli_start_time.strftime('%H:%M:%S'))}")
                    print(f"Fin: {exec_time.get('end_time', cli_end_time.strftime('%H:%M:%S'))}")
                
                forecasts = forecast_result['forecasts']
                for variable, forecast in forecasts.items():
                    if forecast['status'] == 'success':
                        value = forecast['values'][0]
                        unit = forecast['unit']
                        description = forecast['description']
                        metrics = forecast.get('metrics', {})
                        
                        print(f"  {description}: {value:.2f} {unit}")
                        if metrics:
                            print(f"    MSE: {metrics.get('MSE', 'N/A')}")
                            print(f"    RMSE: {metrics.get('RMSE', 'N/A')}")
                            print(f"    MAE: {metrics.get('MAE', 'N/A')}")
                            print(f"    R²: {metrics.get('R2', 'N/A')}")
                        print()
                    else:
                        print(f"  {variable}: Error - {forecast.get('error', 'Error desconocido')}")
            
            else:
                cli_end_time = datetime.now()
                cli_duration = (cli_end_time - cli_start_time).total_seconds()
                print(f"[{cli_end_time.strftime('%H:%M:%S')}] ERROR - Error generando pronóstico después de {cli_duration:.2f} segundos: {forecast_result.get('error', 'Error desconocido')}")
                sys.exit(1)
                
        except ValueError as e:
            print(f"ERROR - Error en parámetros: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR - Error interno: {e}")
            sys.exit(1)
    
    else:
        # Modo API Web
        print("=== INICIANDO SKYCAST API WEB ===")
        ensure_dirs()
        print("OK - Directorios creados")
        print("Iniciando servidor Flask...")
        print("API disponible en: http://127.0.0.1:5001")
        print("Interfaz web: http://127.0.0.1:5001")
        print("Endpoint API: http://127.0.0.1:5001/api/forecast")
        print("Estado API: http://127.0.0.1:5001/api/status")
        print("Presiona Ctrl+C para detener el servidor")
        
        # Ejecutar la aplicación Flask
        app.run(host='127.0.0.1', port=5001, debug=True)

if __name__ == "__main__":
    main()
