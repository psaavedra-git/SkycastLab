# ⚡ Inicio Rápido - SkyCast

## 🚀 Instalación en 3 Pasos

### **1️⃣ Configurar Entorno**
```bash
# Windows
setup_environment.bat

# Linux/Mac
chmod +x setup_environment.sh && ./setup_environment.sh
```

### **2️⃣ Configurar API Key**
- Abrir `config.py`
- Reemplazar `"tu_api_key_aqui"` con tu API key de OpenAI
- Obtener API key: https://platform.openai.com/api-keys

### **3️⃣ Ejecutar Programa**
```bash
# Windows
run_skycast.bat

# Linux/Mac
chmod +x run_skycast.sh && ./run_skycast.sh
```

## 🌐 Acceder al Sistema
- **URL:** http://127.0.0.1:5001
- **Seleccionar ubicación** en el mapa
- **Generar pronóstico** meteorológico

## ✅ Verificación Rápida
```bash
# Verificar que todo funciona
python -c "import tensorflow, openai, flask; print('✓ Sistema listo')"
```

## 🆘 Si Algo Sale Mal
1. **Verificar Python:** `python --version` (debe ser 3.8+)
2. **Reinstalar dependencias:** Ejecutar setup nuevamente
3. **Verificar API key:** Debe ser diferente a "tu_api_key_aqui"
4. **Revisar puerto:** Verificar que 5001 esté libre

---
**¡Listo en menos de 5 minutos!** ⚡
