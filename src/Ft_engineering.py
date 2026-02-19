# ==============================================
# ft_engineering.py - Ingeniería de características
# ==============================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

# Configuración de rutas
DATA_PATH = Path(__file__).resolve().parent.parent / "Base_de_datos.csv"

# ==============================================
# 1. Carga y Limpieza Básica
# ==============================================
def load_data():
    if not DATA_PATH.exists():
        local_path = Path("Base_de_datos.csv")
        if local_path.exists():
            path_to_use = local_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo en: {DATA_PATH} ni en local.")
    else:
        path_to_use = DATA_PATH
    
    print(f"Cargando datos desde: {path_to_use}")
    df = pd.read_csv(path_to_use, encoding='latin-1', sep=',')
    
    # 1. Limpieza de nulos "basura"
    nulos_reales = ['', ' ', '-', 'nan', 'NaN', 'NAN', 'null', 'None', 'N/A']
    df.replace(nulos_reales, np.nan, inplace=True)
    
    # 2. CORRECCIÓN DE TIPOS (Vital)
    if 'tipo_credito' in df.columns:
        df['tipo_credito'] = df['tipo_credito'].astype(str)

    # 3. Limpieza de 'tendencia_ingresos'
    if 'tendencia_ingresos' in df.columns:
        validos = ['Creciente', 'Decreciente', 'Estable']
        df.loc[~df['tendencia_ingresos'].isin(validos), 'tendencia_ingresos'] = np.nan

    # 4. Limpieza de Negativos (Puntajes)
    if 'puntaje_datacredito' in df.columns:
        df['puntaje_datacredito'] = df['puntaje_datacredito'].clip(lower=0)

    # 5. Convertir fecha (Ajustado para evitar el warning de pandas)
    if 'fecha_prestamo' in df.columns:
        df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'], errors='coerce', format='mixed')
        df = df.dropna(subset=['fecha_prestamo'])
        
    # 6. Convertir Target
    if 'Pago_atiempo' in df.columns:
        df = df.dropna(subset=['Pago_atiempo'])
        df['Pago_atiempo'] = df['Pago_atiempo'].astype(int)

    return df

# ==============================================
# 2. Feature Engineering
# ==============================================
def create_safe_features(df):
    df = df.copy()
    
    # --- FECHAS ---
    if 'fecha_prestamo' in df.columns:
        df['mes_solicitud'] = df['fecha_prestamo'].dt.month
        df['dia_semana_solicitud'] = df['fecha_prestamo'].dt.dayofweek.astype(str) 

    # --- RATIOS FINANCIEROS (Usamos +1 para evitar div/0) ---
    if 'cuota_pactada' in df.columns and 'salario_cliente' in df.columns:
        df['ratio_cuota_salario'] = df['cuota_pactada'] / (df['salario_cliente'] + 1)
        
    if 'cuota_pactada' in df.columns and 'total_otros_prestamos' in df.columns and 'salario_cliente' in df.columns:
        df['ratio_endeudamiento_total'] = (df['cuota_pactada'] + df['total_otros_prestamos']) / (df['salario_cliente'] + 1)

    if 'capital_prestado' in df.columns and 'salario_cliente' in df.columns:
        df['ratio_monto_ingreso'] = df['capital_prestado'] / (df['salario_cliente'] + 1)

    if 'edad_cliente' in df.columns:
        df['es_joven'] = (df['edad_cliente'] < 25).astype(int)
        df['edad_cliente'] = df['edad_cliente'].clip(upper=90)

    # --- ENDEUDAMIENTO REAL ---
    if 'cuota_pactada' in df.columns and 'total_otros_prestamos' in df.columns:
        df["carga_financiera_total"] = df["cuota_pactada"] + df["total_otros_prestamos"]
        
        if 'salario_cliente' in df.columns:
            df["ratio_endeudamiento_real"] = df["carga_financiera_total"] / (df["salario_cliente"] + 1)
            df["ratio_endeudamiento_real"] = df["ratio_endeudamiento_real"].clip(0, 20)

    # --- BEHAVIORAL FEATURES ---
    cols_sectores = ["creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal"]
    if all(c in df.columns for c in cols_sectores):
        df["total_entidades"] = df[cols_sectores].sum(axis=1)
        if "cant_creditosvigentes" in df.columns:
            df["creditos_por_entidad"] = df["cant_creditosvigentes"] / (df["total_entidades"] + 1)

    # Ratio de Consultas
    if "huella_consulta" in df.columns and "cant_creditosvigentes" in df.columns:
        df["ratio_consultas_creditos"] = df["huella_consulta"] / (df["cant_creditosvigentes"] + 1)

    return df

# ==============================================
# 3. Pipeline & Split
# ==============================================
def make_train_test_data(target_col='Pago_atiempo', split_ratio=0.8):
    print("--- INICIANDO PIPELINE V5 (LOGARITMOS) ---")
    
    # 1. Carga
    df = load_data()
    df = create_safe_features(df)
    
    # 2. Ordenamiento Temporal
    if 'fecha_prestamo' in df.columns:
        df = df.sort_values('fecha_prestamo')
    
    # 3. Eliminación de Leakage
    leakage_cols = [
        'saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal',
        'dias_mora', 'ratio_mora_capital', 
        'puntaje', 'score_promedio', 
        'fecha_prestamo', target_col
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    # 4. Split
    split_idx = int(len(df) * split_ratio)
    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    print(f"✔ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 5. Definición del Pipeline
    skewed_cols = [
        'salario_cliente', 'total_otros_prestamos', 'cuota_pactada', 
        'capital_prestado', 'carga_financiera_total'
    ]
    skewed_features = [c for c in skewed_cols if c in X_train.columns]
    
    skewed_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        # ¡AQUÍ ESTÁ LA CORRECCIÓN! Agregado feature_names_out='one-to-one'
        ('log', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')), 
        ('scaler', RobustScaler())
    ])
    
    other_numeric = [c for c in X_train.select_dtypes(include=['number']).columns if c not in skewed_features]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', skewed_transformer, skewed_features),
            ('num', numeric_transformer, other_numeric),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object', 'category', 'bool']))
        ],
        verbose_feature_names_out=False
    )
    
    preprocessor.fit(X_train)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    try:
        X_tr, X_te, y_tr, y_te, pipe = make_train_test_data()
        print("✔ Feature Engineering completado.")
        
        feats = pipe.get_feature_names_out()
        tendencia_nulos = [f for f in feats if 'tendencia_ingresos_Desconocido' in f]
        if tendencia_nulos:
            print("✔ Éxito: Se creó la categoría 'tendencia_ingresos_Desconocido' para los nulos.")
        else:
            print("⚠️ Ojo: No se encontró la categoría de nulos (puede que no hubieran en Train).")
            
    except Exception as e:
        print(f"❌ Error: {e}")