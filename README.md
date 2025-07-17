# Kilimo ML Test

## Resumen del Proyecto

Este proyecto implementa un sistema de predicción de rendimiento de cultivos utilizando técnicas de aprendizaje automático. El sistema analiza datos históricos de diferentes países, incluyendo información sobre precipitaciones, uso de pesticidas y temperaturas promedio, para predecir el rendimiento de cultivos en hectogramos por hectárea.

### Componentes Principales

- **Pipeline de Datos**: Descarga y procesa datos de Kaggle que incluyen información sobre pesticidas, precipitaciones, temperaturas y rendimientos históricos de cultivos.
- **Modelos de ML**: Implementa dos modelos de regresión (Regresión Lineal y Random Forest) para predecir el rendimiento de cultivos.
- **API REST**: Proporciona un endpoint para realizar predicciones en tiempo real basadas en los modelos entrenados.

### Funcionalidades

- Descarga y procesamiento automático de datos agrícolas
- Entrenamiento de modelos con optimización de hiperparámetros
- Evaluación comparativa de modelos usando métricas como MSE, RMSE, MAE y R²
- API para predicciones con validación de datos de entrada
- Visualizaciones de correlaciones entre variables y comparativas de rendimiento de modelos

### Tecnologías Utilizadas

- Python
- FastAPI
- Pandas y DuckDB para procesamiento de datos
- Scikit-learn para modelos de aprendizaje automático
- Docker para contenerización

### Ejecución del Pipeline

Para ejecutar el pipeline de datos y entrenamiento de modelos, utilice el siguiente comando:

```
docker compose up --build pipelines
```

Este comando construirá la imagen Docker necesaria y ejecutará el pipeline de datos que descarga y procesa la información de cultivos desde Kaggle.

Para entrenar los modelos después de ejecutar el pipeline de datos, utilice:

```
docker compose up --build ml_training
```

### Artefactos Generados

El proceso de entrenamiento genera varios artefactos en la carpeta `.models/` que incluyen:

1. **Modelos Entrenados**:
   - `random_forest_model.pkl`: Modelo Random Forest entrenado para predecir rendimientos de cultivos.
   - `linear_regression_model.pkl`: Modelo de Regresión Lineal entrenado para predecir rendimientos de cultivos.
   - `label_encoder.pkl`: Codificador para transformar variables categóricas (países y elementos) en valores numéricos.

2. **Métricas y Visualizaciones**:
   - `model_evaluation_results.csv`: Archivo CSV con métricas de evaluación (MSE, RMSE, MAE, R²) para ambos modelos.
   - `correlation_heatmap.png`: Mapa de calor que muestra las correlaciones entre las diferentes variables del conjunto de datos.
   - `model_comparison_mse.png`: Gráfico comparativo del Error Cuadrático Medio (MSE) de ambos modelos.
   - `model_comparison_rmse.png`: Gráfico comparativo del Error Cuadrático Medio de la Raíz (RMSE) de ambos modelos.
   - `model_comparison_mae.png`: Gráfico comparativo del Error Absoluto Medio (MAE) de ambos modelos.
   - `model_comparison_r^2_score.png`: Gráfico comparativo del coeficiente de determinación (R²) de ambos modelos.

El pipeline de datos genera varios artefactos en la carpeta `.data/` que incluyen:

3. **Datos Originales**:
   - `pesticides.csv`: Datos sobre el uso de pesticidas por país y año.
   - `rainfall.csv`: Datos sobre precipitaciones promedio por país y año.
   - `temp.csv`: Datos sobre temperaturas promedio por país y año.
   - `yield.csv`: Datos sobre rendimientos de cultivos por país, año y tipo de cultivo.

4. **Datos Procesados**:
   - `crop_yield_processed.parquet`: Archivo Parquet que contiene el conjunto de datos combinado y procesado, listo para el entrenamiento de modelos. Este archivo incluye todas las características (precipitaciones, pesticidas, temperaturas) y la variable objetivo (rendimiento en hg/ha).

### Ejecución de la API

Para ejecutar la API REST que permite realizar predicciones, utilice el siguiente comando:

```
docker compose up --build api
```

Este comando construirá la imagen Docker necesaria y ejecutará la API en el puerto 8000. Puede acceder a la API a través de http://localhost:8000.

#### Documentación Swagger

FastAPI incluye documentación interactiva generada automáticamente. Para acceder a ella, visite:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Estas interfaces le permitirán explorar todos los endpoints disponibles, ver los esquemas de datos requeridos y probar la API directamente desde el navegador.

#### Endpoint de Predicción

El endpoint principal de la API es `/predict`, que permite predecir el rendimiento de cultivos basado en diferentes parámetros:

- **Método**: POST
- **URL**: http://localhost:8000/predict
- **Payload**: Requiere un objeto JSON con los siguientes campos:
  - `model`: Modelo a utilizar (puede ser "random_forest" o "linear_regression")
  - `country`: País para el cual se realiza la predicción
  - `year`: Año para el cual se realiza la predicción
  - `element`: Tipo de cultivo
  - `average_rain_fall_mm_per_year`: Precipitación promedio anual en mm
  - `pesticide_tonnes`: Uso de pesticidas en toneladas
  - `avg_temp_in_celsius`: Temperatura promedio en grados Celsius

El endpoint procesa estos datos utilizando el modelo especificado y devuelve un objeto JSON que incluye todos los parámetros de entrada más el campo `hg_ha_yield`, que representa el rendimiento predicho en hectogramos por hectárea.

### Nota sobre la Documentación

Esta documentación fue escrita utilizando agentes de IA para facilitar la explicación del proyecto y su estructura. Sin embargo, es importante destacar que todo el código del proyecto fue generado por humanos. Los agentes de IA solo se utilizaron como herramienta para crear esta documentación clara y detallada, mientras que la implementación técnica, incluyendo el pipeline de datos, los modelos de machine learning y la API REST, fue desarrollada íntegramente por programadores humanos.
