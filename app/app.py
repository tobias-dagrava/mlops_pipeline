# ==============================================
# app/app.py - Dashboard de Monitoreo MLOps
# ==============================================
import streamlit as st

# 1. Configuración de página
st.set_page_config(page_title="Monitoreo de Data Drift", page_icon="📊", layout="wide")

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. EL TRUCO DEFINITIVO DE RUTAS
# Obtenemos la raíz y la carpeta src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# Le decimos a Python que busque módulos en ambas carpetas
sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

# 3. Importamos (ahora no fallará ni acá ni adentro de los otros scripts)
from ft_engineering import make_train_test_data
from model_monitoring import detect_drift

# --- INTERFAZ DEL DASHBOARD ---
st.title("📊 Panel de Monitoreo: Riesgo Crediticio")
st.markdown("""
Este dashboard analiza en tiempo real si las características de los **nuevos solicitantes de crédito** han cambiado respecto a la **población histórica** con la que se entrenó el modelo.
""")

@st.cache_data
def load_and_analyze():
    X_train, X_test, _, _, _ = make_train_test_data()
    df_report = detect_drift(X_train, X_test)
    return df_report, X_train, X_test

with st.spinner('Procesando datos (KS, PSI, Chi2)...'):
    df_report, X_train, X_test = load_and_analyze()

st.header("1. Resumen de Estabilidad del Sistema")

alertas_rojas = len(df_report[df_report['Estado'].str.contains("🔴")])
advertencias = len(df_report[df_report['Estado'].str.contains("🟡")])
estables = len(df_report[df_report['Estado'].str.contains("🟢")])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Variables Analizadas", len(df_report))
col2.metric("🟢 Estables", estables)
col3.metric("🟡 Advertencias", advertencias)
col4.metric("🔴 Alertas Críticas", alertas_rojas)

if alertas_rojas > 0:
    st.error(f"**🚨 ALERTA:** Hay {alertas_rojas} variables con cambios críticos en su distribución (PSI > 0.2 o Cambio Proporcional). Se sugiere **Re-entrenamiento (Retraining)**.")
elif advertencias > 0:
    st.warning(f"**⚠️ ADVERTENCIA:** Hay {advertencias} variables con drift leve. Mantener en observación.")
else:
    st.success("**✅ SISTEMA SALUDABLE:** No se detectan desviaciones significativas en la población.")

st.header("2. Detalle de Métricas por Variable")

def color_status(val):
    if '🔴' in str(val): return 'background-color: #ffcccc; color: #990000; font-weight: bold'
    if '🟡' in str(val): return 'background-color: #fff4cc; color: #997300; font-weight: bold'
    if '🟢' in str(val): return 'background-color: #ccffcc; color: #006600'
    return ''

try:
    st.dataframe(df_report.style.map(color_status, subset=['Estado']), use_container_width=True)
except AttributeError:
    st.dataframe(df_report.style.applymap(color_status, subset=['Estado']), use_container_width=True)

st.header("3. Análisis Visual de Desplazamiento")

num_vars = df_report[df_report['Tipo'] == 'Numérica']['Variable'].tolist()
selected_var = st.selectbox("Seleccionar Variable Numérica:", num_vars)

if selected_var:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(X_train[selected_var].dropna(), label='Histórico (Train)', fill=True, color='royalblue', alpha=0.4, ax=ax)
    sns.kdeplot(X_test[selected_var].dropna(), label='Actual (Test)', fill=True, color='crimson', alpha=0.4, ax=ax)
    ax.set_title(f'Comparación de Distribución: {selected_var}')
    ax.set_ylabel('Densidad')
    ax.legend()
    st.pyplot(fig)