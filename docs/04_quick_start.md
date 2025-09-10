# Guía de Inicio Rápido

## 🚀 Tu Primer Modelo de ML en 10 Minutos

Esta guía te llevará paso a paso para crear tu primer modelo de Machine Learning utilizando el Sistema Multi-Agent AutoML. Al final tendrás un modelo entrenado, predicciones y visualizaciones profesionales.

## ✅ Pre-requisitos

Antes de comenzar, asegúrate de que:
- [ ] El sistema está instalado correctamente (ver [Instalación](03_installation.md))
- [ ] Los servicios están ejecutándose (`python start.py`)
- [ ] La interfaz web es accesible en `http://localhost:8006`
- [ ] Tienes un archivo CSV con datos para analizar

## 📊 Preparar Datos de Ejemplo

Si no tienes datos propios, puedes usar nuestro dataset de ejemplo:

### **Crear Dataset de Ventas**
```python
# crear_datos_ejemplo.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generar datos de ventas simulados
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

# Datos con tendencia y estacionalidad
base_sales = 1000
trend = np.linspace(0, 200, 365)
seasonal = 100 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 50, 365)
sales = base_sales + trend + seasonal + noise

# Crear DataFrame
data = {
    'fecha': dates,
    'ventas': sales.round(2),
    'mes': [d.month for d in dates],
    'dia_semana': [d.weekday() for d in dates],
    'promocion': np.random.choice([0, 1], 365, p=[0.8, 0.2])
}

df = pd.DataFrame(data)
df.to_csv('ventas_ejemplo.csv', index=False)
print("✅ Archivo 'ventas_ejemplo.csv' creado")
print(f"📊 Dataset: {len(df)} filas, {len(df.columns)} columnas")
print(df.head())
```

```bash
# Ejecutar el script
python crear_datos_ejemplo.py
```

## 🖥️ Tutorial Paso a Paso

### **Paso 1: Acceder al Sistema**

1. **Abrir navegador** y ir a `http://localhost:8006`
2. **Verificar estado**: Deberías ver el dashboard principal
3. **Comprobar agentes**: Los 7 agentes deben aparecer como "Ready"

### **Paso 2: Cargar Dataset**

1. **Hacer clic** en el botón "📁 Upload Dataset"
2. **Seleccionar** tu archivo `ventas_ejemplo.csv`
3. **Esperar confirmación**: El archivo se carga automáticamente
4. **Verificar**: Deberías ver detalles del archivo en pantalla

### **Paso 3: Definir Objetivo**

1. **Localizar** el campo "User Objective"
2. **Escribir objetivo**: 
   ```
   Predice las ventas futuras para los próximos 30 días basándose en los datos históricos
   ```
3. **Nombrar pipeline**: `prediccion_ventas_2024`

### **Paso 4: Iniciar Pipeline**

1. **Hacer clic** en "🚀 Start ML Pipeline"
2. **Observar progreso**: El sistema mostrará el estado en tiempo real
3. **Ver logs**: Expandir las secciones de logs para ver detalles

### **Paso 5: Monitorear Progreso**

El sistema ejecutará automáticamente estas fases:

#### **Fase 1: Análisis de Datos (1-2 minutos)**
```
🔍 DataProcessorAgent iniciando...
✅ Detectado CSV con separador ','
✅ Encontradas 365 filas, 5 columnas
✅ Columna objetivo sugerida: 'ventas'
✅ Análisis estadístico completado
```

#### **Fase 2: Entrenamiento de Modelos (5-15 minutos)**
```
🧠 ModelBuilderAgent iniciando...
✅ Código Python generado para H2O AutoML
⚡ CodeExecutorAgent ejecutando en Docker...
🔬 H2O AutoML entrenando múltiples modelos...
✅ Mejor modelo: GBM con RMSE: 45.23
🔍 AnalystAgent validando resultados...
✅ Modelo aprobado para producción
```

#### **Fase 3: Predicciones (2-3 minutos)**
```
🎯 PredictionAgent iniciando...
✅ Modelo cargado correctamente
✅ Generando predicciones para 30 días
✅ Archivo de predicciones creado
```

#### **Fase 4: Visualizaciones (1-2 minutos)**
```
📈 VisualizationAgent iniciando...
✅ Gráfico de tendencias generado
✅ Visualización de predicciones completada
✅ Archivos PNG guardados
```

### **Paso 6: Explorar Resultados**

Una vez completado el pipeline, podrás:

1. **Ver métricas del modelo**:
   - Precisión (RMSE, MAE, R²)
   - Importancia de características
   - Validación cruzada

2. **Descargar predicciones**:
   - Archivo CSV con predicciones futuras
   - Intervalos de confianza
   - Datos históricos incluidos

3. **Ver visualizaciones**:
   - Gráfico de tendencia histórica
   - Predicciones futuras
   - Bandas de confianza

## 📋 Ejemplo de Resultados

### **Métricas del Modelo**
```json
{
  "model_performance": {
    "rmse": 45.23,
    "mae": 35.87,
    "r2": 0.89,
    "mean_residual_deviance": 2045.11
  },
  "feature_importance": {
    "fecha": 0.45,
    "mes": 0.25,
    "promocion": 0.20,
    "dia_semana": 0.10
  }
}
```

### **Predicciones (muestra)**
```csv
fecha,ventas_predichas,limite_inferior,limite_superior
2024-01-01,1234.56,1189.23,1279.89
2024-01-02,1245.78,1200.45,1291.11
2024-01-03,1256.90,1211.57,1302.23
...
```

### **Archivos Generados**
```
results/
├── pipeline_abc123/
│   ├── model_performance.json
│   ├── predictions.csv
│   ├── feature_importance.json
│   └── visualizations/
│       ├── historical_trend.png
│       ├── predictions_chart.png
│       └── residuals_plot.png
```

## 🔍 Interpretación de Resultados

### **Métricas de Rendimiento**

**RMSE (Root Mean Square Error)**: 45.23
- ✅ **Bueno**: Error promedio de ~45 unidades de ventas
- 📊 **Contexto**: En ventas promedio de 1200, error del ~3.8%

**R² (Coeficiente de Determinación)**: 0.89
- ✅ **Excelente**: El modelo explica el 89% de la variabilidad
- 📈 **Interpretación**: Muy buena capacidad predictiva

**MAE (Mean Absolute Error)**: 35.87
- ✅ **Bueno**: Error absoluto promedio de ~36 unidades
- 📊 **Contexto**: Más robusto a outliers que RMSE

### **Importancia de Características**

1. **fecha (45%)**: Factor temporal más importante
2. **mes (25%)**: Estacionalidad mensual significativa  
3. **promocion (20%)**: Impacto considerable de promociones
4. **dia_semana (10%)**: Variación semanal menor

### **Calidad de Predicciones**

- **Intervalos de confianza**: Bandas del 95% incluidas
- **Tendencia**: Modelo captura tendencia ascendente
- **Estacionalidad**: Patrones estacionales preservados

## 🎯 Casos de Uso Adicionales

### **Modificar el Objetivo**

Puedes cambiar el objetivo para diferentes análisis:

```
# Clasificación de clientes
"Clasifica a los clientes en segmentos de alto, medio y bajo valor"

# Detección de anomalías
"Detecta ventas anómalas que podrían indicar fraude o errores"

# Optimización de inventario
"Predice la demanda por producto para optimizar inventario"

# Análisis de churn
"Predice qué clientes tienen alta probabilidad de abandonar"
```

### **Diferentes Tipos de Datos**

El sistema maneja diversos formatos:

```python
# Series temporales
fecha, valor, categoria

# Datos transaccionales  
cliente_id, producto, cantidad, precio, fecha

# Datos de comportamiento
usuario, accion, timestamp, dispositivo

# Datos financieros
fecha, precio_apertura, precio_cierre, volumen
```

## ⚡ Consejos para Mejores Resultados

### **Preparación de Datos**
- ✅ **Consistencia**: Formatos de fecha uniformes
- ✅ **Completitud**: Mínimo de outliers o datos faltantes
- ✅ **Relevancia**: Incluir variables predictoras importantes
- ✅ **Volumen**: Al menos 100 observaciones para resultados confiables

### **Definición de Objetivos**
- ✅ **Específico**: "Predice ventas diarias" vs "Analiza ventas"
- ✅ **Medible**: Definir qué constituye éxito
- ✅ **Temporal**: Especificar horizonte de predicción
- ✅ **Contextual**: Incluir información de dominio relevante

### **Interpretación de Resultados**
- ✅ **Validación**: Comparar predicciones con conocimiento del negocio
- ✅ **Intervalos**: Considerar incertidumbre en las predicciones
- ✅ **Tendencias**: Evaluar si las tendencias son realistas
- ✅ **Outliers**: Investigar predicciones extremas

## 🔄 Siguiente Pasos

¡Felicitaciones! Has creado tu primer modelo de Machine Learning automatizado. 

### **Explorar Más**
1. 📖 **[Tutorial Detallado](tutorials/step_by_step_tutorial.md)**: Guía completa con más ejemplos
2. 🤖 **[Documentación de Agentes](agents/)**: Entender cómo funciona cada agente
3. 🔧 **[API Reference](api/api_reference.md)**: Integrar el sistema en tus aplicaciones
4. 📊 **[Casos de Uso](tutorials/use_cases.md)**: Ejemplos para tu industria

### **Experimentar**
- Probar con tus propios datasets
- Modificar objetivos y comparar resultados
- Explorar diferentes tipos de problemas de ML
- Integrar el sistema en workflows existentes

---

**¡Has dado el primer paso en la automatización de Machine Learning! 🎉🤖**