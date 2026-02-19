Pipeline de MLOps: De la Exploracion de Datos al Despliegue en Produccion
1. Resumen del Proyecto y Objetivo de Negocio
Este proyecto desarrolla un sistema integral para la gestion del riesgo crediticio. El objetivo es predecir la probabilidad de incumplimiento de pago de clientes potenciales. La prioridad de negocio es minimizar las perdidas por creditos no recuperados, por lo cual el exito se mide a traves del F2-Score, priorizando la identificacion de morosos (Recall) sobre la precision general.

2. Carga de Datos y Limpieza Inicial
El proceso comienza con la ingestion de datos desde un archivo fuente en formato CSV. Durante esta etapa se realizaron las siguientes acciones:

Manejo de codificacion (latin-1) y delimitadores especificos.

Estandarizacion de valores nulos: Se identificaron cadenas de texto como "None", "nan", "null" o espacios vacios para tratarlos uniformemente como NaN de NumPy.

Conversion de tipos: Transformacion de columnas de fecha a objetos datetime y tipificacion correcta de variables categoricas y numericas.

3. Analisis Exploratorio de Datos (EDA)
Se realizo un analisis profundo para entender la naturaleza de la informacion:

Analisis de distribucion: Se detecto un sesgo extremo en variables como salario y capital prestado, con presencia de valores atípicos (outliers) significativos.

Analisis de correlacion: Se identificaron relaciones fuertes entre el monto del prestamo y la carga financiera.

Identificacion de desbalance: Se confirmo que solo el 5% de la muestra pertenece a la clase de mora (clase 0), lo que condiciono la estrategia de entrenamiento.

Hallazgo de Data Leakage: Se detectaron variables que contenian informacion futura al otorgamiento del credito, lo cual invalidaria el modelo en un escenario real.

4. Ingenieria de Caracteristicas (Feature Engineering)
En este modulo (src/ft_engineering.py) se aplico la logica de negocio para transformar los datos crudos:

Creacion de variables: Se derivaron ratios como la cuota/salario, el ratio de endeudamiento real y la huella de consulta crediticia por entidad.

Tratamiento de Outliers: Implementacion de transformaciones logaritmicas (log1p) para "aplastar" las escalas de variables monetarias y mejorar la convergencia de los modelos.

Pipeline de Preprocesamiento: Uso de ColumnTransformer para aplicar RobustScaler a numericas y OneHotEncoding a categoricas, asegurando que el modelo sea agnostico a la escala de los datos.

Prevencion de Fuga de Datos: Eliminacion sistematica de variables como saldo_mora y puntaje actual, garantizando un entrenamiento etico y tecnicamente correcto.

5. Entrenamiento y Evaluacion del Modelo
Se implemento un flujo de entrenamiento robusto (src/model_training_evaluation.py):

Balanceo de Datos: Uso de Oversampling para igualar la representacion de la clase minoritaria a una proporcion 1:1.

Modelado: Comparacion entre Regresion Logistica y Random Forest Classifier.

Optimizacion de Umbral: En lugar de utilizar el umbral por defecto (0.5), se busco el punto de corte optimo que maximizara el F2-Score, adaptando el modelo a la tolerancia al riesgo de la entidad financiera.

Persistencia: El modelo final, que incluye todo el pipeline de transformacion, se exporta como un objeto serializado (.pkl).

6. Monitoreo y Deteccion de Data Drift
El sistema incluye un motor de vigilancia (src/model_monitoring.py) para detectar cuando los nuevos solicitantes de credito difieren estadisticamente de los datos de entrenamiento:

Metricas implementadas: Population Stability Index (PSI) para estabilidad general, Test de Kolmogorov-Smirnov para distribuciones numericas y Chi-Cuadrado para proporciones categoricas.

Visualizacion: Un dashboard en Streamlit (app/app.py) presenta un reporte visual tipo semaforo para alertar sobre la necesidad de re-entrenamiento del modelo.

7. Despliegue: API y Contenedorizacion
Para la disponibilizacion del modelo se siguieron estandares de produccion:

Backend: Se desarrollo una API con FastAPI (src/model_deploy.py) que permite realizar predicciones individuales o por lotes (batch) enviando datos en formato JSON.

Docker: El proyecto se empaqueta en una imagen de Docker basada en Python 3.12-slim. Esto garantiza que el entorno de ejecucion sea identico sin importar la infraestructura donde se despliegue.

Instrucciones para Reproducir el Proyecto
Requisitos Previos
Python 3.12+

Docker Desktop (para la opcion de despliegue)

Instalacion Local
Clonar el repositorio y acceder a la carpeta.

Crear y activar un entorno virtual.

Instalar dependencias:
pip install -r requirements.txt

Ejecucion del Dashboard de Monitoreo
streamlit run app/app.py

Construccion y Ejecucion del Contenedor Docker (API)
Construir la imagen:
docker build -t api_riesgo_crediticio .

Ejecutar el contenedor:
docker run -p 8000:8000 api_riesgo_crediticio

Acceder a la documentacion de la API en: http://localhost:8000/docs

Conclusiones Técnicas
La integracion de tecnicas de preprocesamiento avanzado, la gestion rigurosa de la fuga de datos y la automatizacion del monitoreo estadistico conforman un pipeline de MLOps completo. Este sistema no solo predice el riesgo, sino que ofrece transparencia sobre su propio desempeño y estabilidad a lo largo del tiempo