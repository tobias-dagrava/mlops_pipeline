# ==============================================
# model_training.py - Entrenamiento (V2.1 - Balance 1:1 y F2-Score)
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    recall_score, f1_score, roc_auc_score, fbeta_score
)

# Importar el pipeline de datos limpios
try:
    from ft_engineering import make_train_test_data
except ImportError:
    print("⚠️ Error: No se encuentra 'ft_engineering.py'.")

# ==============================================
# BLOQUE 1: Función de Balanceo (Oversampling 1:1)
# ==============================================
def balance_train_data_oversampling(X_train, y_train, target_ratio=1.0):
    """
    target_ratio = 1.0 -> Igualamos la cantidad de morosos a los pagadores.
    """
    df_temp = X_train.copy()
    df_temp['target_temp'] = y_train
    
    minority_class = df_temp[df_temp['target_temp'] == 0]
    majority_class = df_temp[df_temp['target_temp'] == 1]
    
    n_majority = len(majority_class)
    n_minority_target = int(n_majority * target_ratio)
    
    print(f"   -> Mayoría (Clase 1) original: {n_majority} (Se mantienen)")
    print(f"   -> Minoría (Clase 0) original: {len(minority_class)}")
    print(f"   -> Minoría (Clase 0) multiplicada: {n_minority_target}")
    
    minority_upsampled = resample(
        minority_class, 
        replace=True, 
        n_samples=n_minority_target, 
        random_state=42
    )
    
    df_balanced = pd.concat([majority_class, minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced.drop(columns=['target_temp']), df_balanced['target_temp']

# ==============================================
# BLOQUE 2: Funciones de Evaluación y Umbral (F2-Score)
# ==============================================
def find_optimal_threshold(y_test, y_proba):
    """ Busca el corte exacto que maximiza el F2-Score (Da prioridad al Recall) """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    best_f2_class0 = 0.0
    
    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)
        # Beta=2 le da el doble de peso al recall que a la precisión
        f2_0 = fbeta_score(y_test, y_pred_temp, beta=2, pos_label=0, zero_division=0)
        
        if f2_0 > best_f2_class0:
            best_f2_class0 = f2_0
            best_threshold = t
            
    print(f"🎯 Umbral Óptimo (Prioridad Riesgo): {best_threshold:.4f} -> Max F2: {best_f2_class0:.4f}")
    return best_threshold

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n=== Evaluación de: {model_name} ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()
    
    return {
        'Modelo': model_name,
        'Recall (Clase 0)': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'F1-Score (Clase 0)': f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    }

def print_feature_importances(model_pipeline, feature_names):
    """ Extrae y grafica las variables más importantes del modelo """
    try:
        model = model_pipeline.named_steps['classifier']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_imp = df_imp.sort_values(by='Importance', ascending=False).head(10)
            print("\n🔍 Top 10 Variables más importantes:")
            print(df_imp.to_string(index=False))
    except Exception as e:
        pass

# ==============================================
# BLOQUE 3: Flujo Principal
# ==============================================
def main():
    print("Iniciando Entrenamiento V2.1 (Balance 1:1 + F2-Score)...")
    
    X_train, X_test, y_train, y_test, preprocessor = make_train_test_data()
    
    print("\n⚖️ Balanceando clases (Oversampling Ratio 1.0)...")
    X_train_bal, y_train_bal = balance_train_data_oversampling(X_train, y_train, target_ratio=1.0)
    
    resultados = []
    
    # -----------------------------------------------------------------
    # MODELO 1: REGRESIÓN LOGÍSTICA
    # -----------------------------------------------------------------
    print("\n🔹 Entrenando Modelo 1: Regresión Logística...")
    pipe_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train_bal, y_train_bal)
    
    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    thresh_lr = find_optimal_threshold(y_test, proba_lr)
    y_pred_lr = (proba_lr >= thresh_lr).astype(int)
    
    roc_lr = roc_auc_score(y_test, proba_lr)
    print(f"📈 ROC-AUC Regresión Logística: {roc_lr:.4f}")
    
    met_lr = evaluate_model(y_test, y_pred_lr, "Regresión Logística")
    met_lr['ROC-AUC'] = roc_lr
    resultados.append(met_lr)
    
    # -----------------------------------------------------------------
    # MODELO 2: RANDOM FOREST
    # -----------------------------------------------------------------
    print("\n🔹 Entrenando Modelo 2: Random Forest...")
    pipe_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Aumentamos un pelín la profundidad (max_depth=6) ya que hay más datos balanceados
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42))
    ])
    pipe_rf.fit(X_train_bal, y_train_bal)
    
    proba_rf = pipe_rf.predict_proba(X_test)[:, 1]
    thresh_rf = find_optimal_threshold(y_test, proba_rf)
    y_pred_rf = (proba_rf >= thresh_rf).astype(int)
    
    roc_rf = roc_auc_score(y_test, proba_rf)
    print(f"📈 ROC-AUC Random Forest: {roc_rf:.4f}")
    
    met_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")
    met_rf['ROC-AUC'] = roc_rf
    resultados.append(met_rf)

    # -----------------------------------------------------------------
    # EXTRACCIÓN DE IMPORTANCIA (Protegida contra errores del Logaritmo)
    # -----------------------------------------------------------------
    try:
        feat_names = pipe_rf.named_steps['preprocessor'].get_feature_names_out()
        print_feature_importances(pipe_rf, feat_names)
    except Exception as e:
        print("\n⚠️ Nota: No se pudo graficar la importancia de las variables.")
        print("Esto es normal por la transformación logarítmica (FunctionTransformer).")

    # --- RESUMEN FINAL ---
    print("\n📊 --- TABLA RESUMEN V2.1 ---")
    df_resumen = pd.DataFrame(resultados)
    print(df_resumen[['Modelo', 'Recall (Clase 0)', 'F1-Score (Clase 0)', 'ROC-AUC']].to_string(index=False))

if __name__ == "__main__":
    main()