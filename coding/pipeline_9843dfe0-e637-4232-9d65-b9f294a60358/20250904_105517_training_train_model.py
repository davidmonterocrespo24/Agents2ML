# TRAINING SCRIPT
# Generated on: 2025-09-04T10:55:17.115721
# Pipeline: pipeline_9843dfe0-e637-4232-9d65-b9f294a60358
# Filename: train_model.py
# Arguments: 
# Script Type: training

import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import numpy as np
from datetime import datetime
import os

# Initialize H2O
h2o.init()

# Define Spanish month abbreviations to numeric mapping
spanish_months = {
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
}

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    try:
        day, month_abbr, year = date_str.split()
        month_abbr = month_abbr.lower().rstrip('.')
        month = spanish_months.get(month_abbr)
        if month:
            return datetime(int(year), month, int(day))
        return None
    except:
        return None

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv('9843dfe0-e637-4232-9d65-b9f294a60358_ventas.csv', 
                 sep=';', 
                 header=None,  # No headers
                 decimal=',',  # Comma as decimal separator
                 encoding='utf-8')

# Name the columns properly
df.columns = ['fecha', 'monto_total']

print(f"Dataset shape: {df.shape}")
print(f"First few rows:\n{df.head()}")

# Parse Spanish dates
print("Parsing dates...")
df['fecha_parsed'] = df['fecha'].apply(parse_spanish_date)

# Check for failed date parsing
failed_dates = df[df['fecha_parsed'].isnull()]
if len(failed_dates) > 0:
    print(f"Warning: {len(failed_dates)} dates could not be parsed")
    print(failed_dates.head())

# Drop rows with invalid dates
df = df.dropna(subset=['fecha_parsed'])

# Sort by date for time series
df = df.sort_values('fecha_parsed')

# Create time-based features
df['year'] = df['fecha_parsed'].dt.year
df['month'] = df['fecha_parsed'].dt.month
df['day'] = df['fecha_parsed'].dt.day
df['day_of_week'] = df['fecha_parsed'].dt.dayofweek
df['day_of_year'] = df['fecha_parsed'].dt.dayofyear

print(f"Final dataset shape after cleaning: {df.shape}")

# Convert to H2O frame
h2o_df = h2o.H2OFrame(df)

# Define predictors and target
predictors = ['year', 'month', 'day', 'day_of_week', 'day_of_year']
target = 'monto_total'

print(f"Predictors: {predictors}")
print(f"Target: {target}")

# Split data (time-series aware split - use last 20% for validation)
total_rows = h2o_df.nrows
train_size = int(total_rows * 0.8)

train = h2o_df[:train_size, :]
test = h2o_df[train_size:, :]

print(f"Training set: {train.nrows} rows")
print(f"Test set: {test.nrows} rows")

# Run AutoML
print("Starting H2O AutoML...")
aml = H2OAutoML(max_models=10, 
                seed=42,
                max_runtime_secs=300,  # 5 minutes
                sort_metric='RMSE')

aml.train(x=predictors, 
          y=target, 
          training_frame=train,
          validation_frame=test)

# Display leaderboard
lb = aml.leaderboard
print("Leaderboard:")
print(lb.head())

# Get best model
best_model = aml.leader
print(f"Best model: {best_model.model_id}")

# Save the model
model_path = h2o.save_model(model=best_model, path="./", force=True)
print(f"Model saved to: {model_path}")

# Make predictions on test set
predictions = best_model.predict(test)
print("Predictions on test set:")
print(predictions.head())

# Calculate performance
performance = best_model.model_performance(test_data=test)
print(f"Model performance on test set:")
print(f"RMSE: {performance.rmse()}")
print(f"MSE: {performance.mse()}")
print(f"MAE: {performance.mae()}")
print(f"R²: {performance.r2()}")

# Save predictions for analysis
predictions_df = predictions.as_data_frame()
predictions_df.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to test_predictions.csv")

# Shutdown H2O
h2o.cluster().shutdown()

print("SUCCESS: Model training completed successfully")
