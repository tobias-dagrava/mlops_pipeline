# ==============================================
# src/model_deploy.py - API de FastAPI para el Modelo
# ==============================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import os

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="Servicio web para predicción de mora en créditos usando Random Forest",
    version="1.0.0"
)

# 2. EL TRUCO DE RUTAS: Subimos un nivel para encontrar la carpeta 'models' en la raíz
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Corregido el typo (best_model_rf.pkl)
model_path = os.path.join(ROOT_DIR, 'models', 'best_model_rf.pkl')

try:
    # Cargar el pipeline que contiene el preprocesador y el modelo
    model_pipeline = joblib.load(model_path)
    print(f"✅ Modelo cargado correctamente desde: {model_path}")
except Exception as e:
    model_pipeline = None
    print(f"⚠️ Error al cargar el modelo: {e}")

# 3. Definir el esquema de datos que espera la API (Soporta Batch)
class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

# 4. Endpoints
@app.get("/")
def home():
    return {"mensaje": "La API de Riesgo Crediticio está funcionando 🚀"}

@app.post("/predict")
def predict_batch(request: PredictionRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="El modelo no está disponible en el servidor.")
    
    try:
        # Convertir el JSON recibido a un DataFrame de Pandas
        df_input = pd.DataFrame(request.data)
        
        # El pipeline de Sklearn se encarga de las transformaciones y la predicción
        predicciones = model_pipeline.predict(df_input)
        probabilidades = model_pipeline.predict_proba(df_input)[:, 1]  
        
        # Formatear la respuesta
        resultados = []
        for i in range(len(predicciones)):
            clase = int(predicciones[i])
            proba = float(probabilidades[i])
            
            resultados.append({
                "registro_id": i + 1,
                "prediccion": clase,
                "probabilidad_pago": round(proba, 4),
                "alerta_riesgo": "ALTO RIESGO (Posible Mora)" if clase == 0 else "BAJO RIESGO (Pago a tiempo)"
            })
            
        return {"predicciones": resultados}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento de datos: {str(e)}")