# TRAINING SCRIPT
# Generated on: 2025-09-04T15:14:19.725484
# Pipeline: pipeline_ed12db74-4167-4756-a811-8fddfa33753b
# Filename: model_training.py
# Arguments: 
# Script Type: training

#!/usr/bin/env python3
# model_h2o_training.py

import argparse
import json
import sys
import time
import re
import pandas as pd
import numpy as np

# Spanish month mapping for date parsing
SPANISH_MONTH_MAP = {
    'ene': 1, 'jan': 1, 'enero': 1,
    'feb': 2, 'febrero': 2,
    'mar': 3, 'marzo': 3,
    'abr': 4, 'abril': 4, 'apr': 4,
    'may': 5, 'mayo': 5,
    'jun': 6, 'junio': 6,
    'jul': 7, 'julio': 7,
    'ago': 8, 'agosto': 8, 'aug': 8,
    'sep': 9, 'septiembre': 9,
    'oct': 10, 'octubre': 10,
    'nov': 11, 'noviembre': 11,
    'dic': 12, 'diciembre': 12
}

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    try:
        if pd.isna(date_str) or not isinstance(date_str, str):
            return pd.NaT
        
        date_str = date_str.strip().lower()
        
        match = re.match(r'(\d{1,2})\s+([a-z]+)\.?\s+(\d{4})', date_str)
        if not match:
            return pd.NaT
            
        day, month_str, year = match.groups()
        month = SPANISH_MONTH_MAP.get(month_str.lower())
        if not month:
            return pd.NaT
            
        return pd.Timestamp(int(year), month, int(day))
    except:
        return pd.NaT

def parse_numeric_with_comma(value):
    """Parse European numeric format with comma as decimal separator"""
    try:
        if pd.isna(value):
            return np.nan
            
        if isinstance(value, str):
            value = value.replace(' ', '').replace(',', '.')
            if value.count('.') > 1:
                parts = value.split('.')
                integer_part = ''.join(parts[:-1])
                decimal_part = parts[-1]
                value = f"{integer_part}.{decimal_part}"
            return float(value)
        return float(value)
    except:
        return np.nan

def read_and_preprocess_data(file_path):
    """Read and preprocess the sales data"""
    try:
        # Read the CSV file with semicolon separator and no header
        df = pd.read_csv(file_path, sep=';', header=None, encoding='utf-8')
        
        # Assign proper column names
        df.columns = ['fecha', 'monto_total']
        
        # Parse Spanish dates
        df['fecha'] = df['fecha'].apply(parse_spanish_date)
        
        # Parse numeric values with comma decimal separator
        df['monto_total'] = df['monto_total'].apply(parse_numeric_with_comma)
        
        # Create time-based features
        df['year'] = df['fecha'].dt.year
        df['month'] = df['fecha'].dt.month
        df['day'] = df['fecha'].dt.day
        df['dayofweek'] = df['fecha'].dt.dayofweek
        df['is_weekend'] = df['fecha'].dt.dayofweek.isin([5,6]).astype(int)
        df['dayofyear'] = df['fecha'].dt.dayofyear
        
        # Drop the original date column
        df = df.drop('fecha', axis=1)
        
        return df
        
    except Exception as e:
        print(f"ERROR_START:Data preprocessing failed: {str(e)}:ERROR_END")
        raise

def main():
    try:
        # Read and preprocess the data
        file_path = "./ed12db74-4167-4756-a811-8fddfa33753b_ventas.csv"
        df = read_and_preprocess_data(file_path)
        
        # Display data summary
        summary = {
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "target_stats": {
                "min": float(df["monto_total"].min()),
                "max": float(df["monto_total"].max()),
                "mean": float(df["monto_total"].mean()),
                "std": float(df["monto_total"].std())
            }
        }
        
        print("DATA_SUMMARY_START:" + json.dumps(summary) + ":DATA_SUMMARY_END")
        
        # Initialize H2O
        import h2o
        from h2o.automl import H2OAutoML
        
        h2o.init()
        
        # Convert to H2O frame
        hf = h2o.H2OFrame(df)
        
        # Define target and features
        y = "monto_total"
        x = [col for col in hf.columns if col != y]
        
        # Split data
        splits = hf.split_frame(ratios=[0.8], seed=42)
        train = splits[0]
        test = splits[1]
        
        # Train AutoML
        aml = H2OAutoML(max_models=10, max_runtime_secs=300, seed=42)
        aml.train(x=x, y=y, training_frame=train)
        
        # Get leader model
        leader = aml.leader
        
        # Save model
        model_path = h2o.save_model(model=leader, path="./", force=True)
        
        # Get metrics
        perf = leader.model_performance(test)
        metrics = {
            "rmse": perf.rmse(),
            "mae": perf.mae(),
            "r2": perf.r2(),
            "mse": perf.mse()
        }
        
        result = {
            "model_path": model_path,
            "metrics": metrics,
            "leader_model": leader.model_id
        }
        
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        print("METRICS_START:" + json.dumps(result) + ":METRICS_END")
        
        h2o.cluster().shutdown()
        return 0
        
    except Exception as e:
        print(f"ERROR_START:{str(e)}:ERROR_END")
        return 1

if __name__ == "__main__":
    sys.exit(main())
