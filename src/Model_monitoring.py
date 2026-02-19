# ==============================================
# model_monitoring.py - Detección de Data Drift
# ==============================================

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Importamos tu función para obtener Train (Referencia) y Test (Actual)
try:
    from ft_engineering import make_train_test_data
except ImportError:
    print("⚠️ Error: No se encuentra 'ft_engineering.py'.")

# ==============================================
# 1. Funciones Matemáticas de Drift
# ==============================================

def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI).
    Compara la distribución de una variable numérica en 10 rangos (deciles).
    - PSI < 0.1 : Sin cambios
    - 0.1 <= PSI <= 0.2 : Cambio leve (Monitorear)
    - PSI > 0.2 : Cambio significativo (Alerta de Retraining)
    """
    # Manejo de nulos y asegurar que sean arrays de numpy
    expected = np.array(expected)[~np.isnan(expected)]
    actual = np.array(actual)[~np.isnan(actual)]
    
    # Si hay muy pocos datos, evitamos errores
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Crear los cortes (bins) basados en la data esperada (Train)
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf  # Asegurar captura del mínimo
    breakpoints[-1] = np.inf  # Asegurar captura del máximo

    # Contar cuántos caen en cada balde
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Reemplazar ceros con un número muy pequeño para evitar división por cero o log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Fórmula PSI: sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return psi_value

def calculate_js_divergence(expected, actual, bins=10):
    """
    Divergencia de Jensen-Shannon. 
    Mide similitud entre dos distribuciones de probabilidad. Rango [0, 1].
    Mismo concepto que KS pero usando teoría de la información.
    """
    expected = np.array(expected)[~np.isnan(expected)]
    actual = np.array(actual)[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    hist_range = (min(np.min(expected), np.min(actual)), max(np.max(expected), np.max(actual)))
    
    # Crear histogramas comparables
    p, _ = np.histogram(expected, bins=bins, range=hist_range, density=True)
    q, _ = np.histogram(actual, bins=bins, range=hist_range, density=True)
    
    # Convertir a probabilidades (suman 1)
    p = p / p.sum() + 1e-10
    q = q / q.sum() + 1e-10
    
    return jensenshannon(p, q)

# ==============================================
# 2. Orquestador de Detección de Drift
# ==============================================

def detect_drift(df_reference, df_current):
    """
    Analiza todas las columnas buscando Drift.
    Devuelve un DataFrame con el reporte de métricas.
    """
    drift_report = []
    
    # Identificar numéricas y categóricas
    num_cols = df_reference.select_dtypes(include=[np.number]).columns
    cat_cols = df_reference.select_dtypes(exclude=[np.number]).columns

    # Analizar variables numéricas
    for col in num_cols:
        if col in df_current.columns:
            ref_data = df_reference[col].dropna()
            curr_data = df_current[col].dropna()
            
            # 1. Kolmogorov-Smirnov (p-value < 0.05 indica drift estadístico)
            ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
            
            # 2. Population Stability Index (PSI)
            psi_val = calculate_psi(ref_data, curr_data)
            
            # 3. Jensen-Shannon
            js_val = calculate_js_divergence(ref_data, curr_data)
            
            # Definir estado según PSI
            if psi_val > 0.2:
                status = "🔴 Alerta (Alto Drift)"
            elif psi_val > 0.1:
                status = "🟡 Advertencia (Drift Leve)"
            else:
                status = "🟢 Estable"

            drift_report.append({
                'Variable': col,
                'Tipo': 'Numérica',
                'KS_p_value': round(p_value, 4),
                'PSI': round(psi_val, 4),
                'JS_Divergence': round(js_val, 4),
                'Chi2_p_value': None,
                'Estado': status
            })

    # Analizar variables categóricas
    for col in cat_cols:
        if col in df_current.columns:
            ref_data = df_reference[col].dropna()
            curr_data = df_current[col].dropna()
            
            # Alinear categorías
            all_cats = list(set(ref_data.unique()) | set(curr_data.unique()))
            ref_counts = ref_data.value_counts().reindex(all_cats, fill_value=0).values
            curr_counts = curr_data.value_counts().reindex(all_cats, fill_value=0).values
            
            # Evitar ceros absolutos para Chi2
            ref_counts = np.where(ref_counts == 0, 1e-5, ref_counts)
            curr_counts = np.where(curr_counts == 0, 1e-5, curr_counts)
            
            # 4. Chi-Cuadrado
            chi2_stat, p_value = stats.chisquare(f_obs=curr_counts, f_exp=ref_counts * (curr_counts.sum() / ref_counts.sum()))
            
            # Estado para categoricas (p-value < 0.05 significa que cambiaron las proporciones)
            status = "🔴 Alerta (Cambio Proporción)" if p_value < 0.05 else "🟢 Estable"

            drift_report.append({
                'Variable': col,
                'Tipo': 'Categórica',
                'KS_p_value': None,
                'PSI': None,
                'JS_Divergence': None,
                'Chi2_p_value': round(p_value, 4),
                'Estado': status
            })

    return pd.DataFrame(drift_report)

# ==============================================
# 3. Prueba Rápida (Main)
# ==============================================
if __name__ == "__main__":
    print("Iniciando análisis de Data Drift (Train vs Test)...")
    
    # 1. Cargamos datos históricos (Train) y datos nuevos (Test)
    # Como tu TimeSplit cortó por fecha, Test son los datos más recientes. ¡Perfecto para simular producción!
    X_train, X_test, y_train, y_test, _ = make_train_test_data()
    
    # 2. Generar Reporte
    df_report = detect_drift(X_train, X_test)
    
    print("\n📊 REPORTE DE DATA DRIFT:")
    print(df_report.to_string(index=False))
    
    # Resumen de alertas
    alertas = df_report[df_report['Estado'].str.contains("🔴|🟡")]
    print(f"\n⚠️ Total de variables requiriendo atención: {len(alertas)}")
    if len(alertas) > 0:
        print(alertas[['Variable', 'Estado']].to_string(index=False))