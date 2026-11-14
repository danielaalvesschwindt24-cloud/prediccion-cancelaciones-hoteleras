# Predicción de cancelaciones hoteleras

**Trabajo práctico integrador – Implementación de Modelos de Aprendizaje Automático**

Especialización en Análisis y Gestión de Datos en Organizaciones – UBA.

## Objetivo

Desarrollar un modelo de aprendizaje automático que permita identificar reservas hoteleras con mayor probabilidad de cancelación, facilitando la toma de decisiones preventivas.

## Metodología

Se utilizó el dataset *Hotel Booking Demand*, con 119.390 registros.

El proyecto comprende exploración y preparación de datos, entrenamiento y comparación de modelos, optimización de hiperparámetros, despliegue por lotes, monitoreo e interpretación de resultados.

Se compararon modelos de regresión logística y Random Forest, seleccionando este último.

## Resultados

El modelo Random Forest seleccionado obtuvo en evaluación:

- Accuracy: 77,71 %
- Precision: 72,91 %
- Recall: 62,52 %
- F1-score: 67,32 %
- ROC-AUC: 83,51 %

En la simulación de producción se observó una disminución del rendimiento, analizada mediante métricas de clasificación y cambios en las distribuciones de variables (PSI).

## Archivos del proyecto

- `TP_Integrador_Implementacion_GRUPAL.ipynb`: desarrollo y evaluación del modelo.
- `predecir_lotes.py`: script independiente para generar predicciones por lotes.
- `requirements.txt`: dependencias del proyecto.
- `datos/`: datos de ejemplo para ejecutar predicciones.
- `resultados/prueba_script/`: predicciones y bitácora de ejecución.

## Modelo entrenado

El modelo se encuentra disponible en la sección [Releases – v1.0.0](https://github.com/danielaalvesschwindt24-cloud/prediccion-cancelaciones-hoteleras/releases/tag/v1.0.0).

Descargar `hotel_cancelaciones_v1.joblib` y colocarlo en la carpeta `modelo/`.

El archivo incluye el pipeline de preprocesamiento, el modelo Random Forest entrenado y el umbral de clasificación de 0,45.

## Despliegue

Se implementó un sistema de predicción por lotes mediante un script de Python, sin necesidad de ejecutar el notebook de entrenamiento.

El script genera un archivo CSV con las predicciones y una bitácora con información de las ejecuciones.

## Monitoreo

Se comparó el desempeño de evaluación con un conjunto temporal reservado para simular producción. Se analizaron las variaciones de rendimiento y los cambios en la distribución de variables mediante PSI, estableciendo criterios para activar alertas y evaluar un nuevo entrenamiento.
