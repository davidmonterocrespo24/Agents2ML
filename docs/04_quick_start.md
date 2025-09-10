# Quick Start Guide

## 🚀 Your First ML Model in 10 Minutes

This guide will take you step by step to create your first Machine Learning model using the Multi-Agent AutoML System. By the end, you'll have a trained model, predictions, and professional visualizations.

## ✅ Prerequisites

Before starting, make sure that:
- [ ] The system is installed correctly (see [Installation](03_installation.md))
- [ ] Services are running (`python start.py`)
- [ ] Web interface is accessible at `http://localhost:8006`
- [ ] You have a CSV file with data to analyze

## 📊 Prepare Sample Data

If you don't have your own data, you can use our example dataset:

### **Create Sales Dataset**
```python
# create_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate simulated sales data
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

# Data with trend and seasonality
base_sales = 1000
trend = np.linspace(0, 200, 365)
seasonal = 100 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 50, 365)
sales = base_sales + trend + seasonal + noise

# Create DataFrame
data = {
    'date': dates,
    'sales': sales.round(2),
    'month': [d.month for d in dates],
    'day_of_week': [d.weekday() for d in dates],
    'promotion': np.random.choice([0, 1], 365, p=[0.8, 0.2])
}

df = pd.DataFrame(data)
df.to_csv('sales_example.csv', index=False)
print("✅ File 'sales_example.csv' created")
print(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
print(df.head())
```

```bash
# Execute the script
python create_sample_data.py
```

## 🖥️ Step-by-Step Tutorial

### **Step 1: Access System**

1. **Open browser** and go to `http://localhost:8006`
2. **Verify status**: You should see the main dashboard
3. **Check agents**: All 7 agents should appear as "Ready"

### **Step 2: Upload Dataset**

1. **Click** the "📁 Upload Dataset" button
2. **Select** your `sales_example.csv` file
3. **Wait for confirmation**: The file uploads automatically
4. **Verify**: You should see file details on screen

### **Step 3: Define Objective**

1. **Locate** the "User Objective" field
2. **Write objective**: 
   ```
   Predict future sales for the next 30 days based on historical data
   ```
3. **Name pipeline**: `sales_prediction_2024`

### **Step 4: Start Pipeline**

1. **Click** "🚀 Start ML Pipeline"
2. **Observe progress**: System will show real-time status
3. **View logs**: Expand log sections to see details

### **Step 5: Monitor Progress**

The system will automatically execute these phases:

#### **Phase 1: Data Analysis (1-2 minutes)**
```
🔍 DataProcessorAgent starting...
✅ Detected CSV with separator ','
✅ Found 365 rows, 5 columns
✅ Suggested target column: 'sales'
✅ Statistical analysis completed
```

#### **Phase 2: Model Training (5-15 minutes)**
```
🧠 ModelBuilderAgent starting...
✅ Python code generated for H2O AutoML
⚡ CodeExecutorAgent executing in Docker...
🔬 H2O AutoML training multiple models...
✅ Best model: GBM with RMSE: 45.23
🔍 AnalystAgent validating results...
✅ Model approved for production
```

#### **Phase 3: Predictions (2-3 minutes)**
```
🎯 PredictionAgent starting...
✅ Model loaded correctly
✅ Generating predictions for 30 days
✅ Prediction file created
```

#### **Phase 4: Visualizations (1-2 minutes)**
```
📈 VisualizationAgent starting...
✅ Trend chart generated
✅ Prediction visualization completed
✅ PNG files saved
```

### **Step 6: Explore Results**

Once the pipeline is completed, you can:

1. **View model metrics**:
   - Accuracy (RMSE, MAE, R²)
   - Feature importance
   - Cross-validation

2. **Download predictions**:
   - CSV file with future predictions
   - Confidence intervals
   - Historical data included

3. **View visualizations**:
   - Historical trend chart
   - Future predictions
   - Confidence bands

## 📋 Example Results

### **Model Metrics**
```json
{
  "model_performance": {
    "rmse": 45.23,
    "mae": 35.87,
    "r2": 0.89,
    "mean_residual_deviance": 2045.11
  },
  "feature_importance": {
    "date": 0.45,
    "month": 0.25,
    "promotion": 0.20,
    "day_of_week": 0.10
  }
}
```

### **Predictions (sample)**
```csv
date,predicted_sales,lower_limit,upper_limit
2024-01-01,1234.56,1189.23,1279.89
2024-01-02,1245.78,1200.45,1291.11
2024-01-03,1256.90,1211.57,1302.23
...
```

### **Generated Files**
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

## 🔍 Interpreting Results

### **Performance Metrics**

**RMSE (Root Mean Square Error)**: 45.23
- ✅ **Good**: Average error of ~45 sales units
- 📊 **Context**: In average sales of 1200, error of ~3.8%

**R² (Coefficient of Determination)**: 0.89
- ✅ **Excellent**: Model explains 89% of variability
- 📈 **Interpretation**: Very good predictive capability

**MAE (Mean Absolute Error)**: 35.87
- ✅ **Good**: Average absolute error of ~36 units
- 📊 **Context**: More robust to outliers than RMSE

### **Feature Importance**

1. **date (45%)**: Most important temporal factor
2. **month (25%)**: Significant monthly seasonality  
3. **promotion (20%)**: Considerable impact of promotions
4. **day_of_week (10%)**: Minor weekly variation

### **Prediction Quality**

- **Confidence intervals**: 95% bands included
- **Trend**: Model captures upward trend
- **Seasonality**: Seasonal patterns preserved

## 🎯 Additional Use Cases

### **Modify Objective**

You can change the objective for different analyses:

```
# Customer classification
"Classify customers into high, medium and low value segments"

# Anomaly detection
"Detect anomalous sales that could indicate fraud or errors"

# Inventory optimization
"Predict demand by product to optimize inventory"

# Churn analysis
"Predict which customers have high probability of churning"
```

### **Different Data Types**

The system handles various formats:

```python
# Time series
date, value, category

# Transactional data  
customer_id, product, quantity, price, date

# Behavioral data
user, action, timestamp, device

# Financial data
date, opening_price, closing_price, volume
```

## ⚡ Tips for Better Results

### **Data Preparation**
- ✅ **Consistency**: Uniform date formats
- ✅ **Completeness**: Minimum outliers or missing data
- ✅ **Relevance**: Include important predictor variables
- ✅ **Volume**: At least 100 observations for reliable results

### **Objective Definition**
- ✅ **Specific**: "Predict daily sales" vs "Analyze sales"
- ✅ **Measurable**: Define what constitutes success
- ✅ **Temporal**: Specify prediction horizon
- ✅ **Contextual**: Include relevant domain information

### **Result Interpretation**
- ✅ **Validation**: Compare predictions with business knowledge
- ✅ **Intervals**: Consider uncertainty in predictions
- ✅ **Trends**: Evaluate if trends are realistic
- ✅ **Outliers**: Investigate extreme predictions

## 🔄 Next Steps

Congratulations! You've created your first automated Machine Learning model. 

### **Explore More**
1. 📖 **[Detailed Tutorial](tutorials/step_by_step_tutorial.md)**: Complete guide with more examples
2. 🤖 **[Agents Documentation](agents/)**: Understand how each agent works
3. 🔧 **[API Reference](api/api_reference.md)**: Integrate the system into your applications
4. 📊 **[Use Cases](tutorials/use_cases.md)**: Examples for your industry

### **Experiment**
- Try with your own datasets
- Modify objectives and compare results
- Explore different types of ML problems
- Integrate the system into existing workflows

---

**You've taken the first step in Machine Learning automation! 🎉🤖**