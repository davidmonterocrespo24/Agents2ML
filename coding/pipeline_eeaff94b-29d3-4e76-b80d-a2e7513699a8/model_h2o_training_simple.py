#!/usr/bin/env python3
# model_h2o_training_simple.py

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np

# Check if H2O is available
try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("ERROR_START:H2O library is not available. Please install h2o package:ERROR_END")
    sys.exit(1)

def read_and_preprocess_data(file_path):
    """Read and preprocess the sales data with Spanish format"""
    try:
        # Read CSV with semicolon separator and no header
        df = pd.read_csv(file_path, sep=';', header=None, names=['fecha', 'monto_total'])
        
        # Convert Spanish month names to English
        month_mapping = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
            'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        
        def convert_spanish_date(date_str):
            import re
            for esp, eng in month_mapping.items():
                date_str = re.sub(rf'\b{esp}\b', eng, date_str, flags=re.IGNORECASE)
            return date_str
        
        # Apply date conversion
        df['fecha'] = df['fecha'].apply(convert_spanish_date)
        
        # Parse dates with proper format
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d %b. %Y', errors='coerce')
        
        # Convert numeric values (handle comma as decimal separator)
        df['monto_total'] = df['monto_total'].astype(str).str.replace('.', '', regex=False)
        df['monto_total'] = df['monto_total'].str.replace(',', '.', regex=False).astype(float)
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to read and preprocess data: {str(e)}")

def create_time_series_features(df):
    """Create time series features from date column"""
    # Basic time features
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    df['day'] = df['fecha'].dt.day
    df['day_of_week'] = df['fecha'].dt.dayofweek
    df['day_of_year'] = df['fecha'].dt.dayofyear
    df['week_of_year'] = df['fecha'].dt.isocalendar().week
    df['is_weekend'] = df['fecha'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_month_start'] = df['fecha'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['fecha'].dt.is_month_end.astype(int)
    df['quarter'] = df['fecha'].dt.quarter
    
    # Lag features for time series
    df['lag_1'] = df['monto_total'].shift(1)
    df['lag_7'] = df['monto_total'].shift(7)
    df['lag_30'] = df['monto_total'].shift(30)
    
    # Rolling statistics
    df['rolling_mean_7'] = df['monto_total'].rolling(window=7).mean()
    df['rolling_std_7'] = df['monto_total'].rolling(window=7).std()
    df['rolling_mean_30'] = df['monto_total'].rolling(window=30).mean()
    
    # Remove rows with NaN values from lag features
    df = df.dropna()
    
    return df

def main():
    parser = argparse.ArgumentParser(description='H2O AutoML training for sales forecasting')
    parser.add_argument('--file', required=True, help='Path to input CSV file')
    parser.add_argument('--target', default='monto_total', help='Target column name')
    parser.add_argument('--output-dir', default='./', help='Directory to save the model')
    parser.add_argument('--max-models', type=int, default=10, help='Maximum number of models')
    parser.add_argument('--max-runtime-secs', type=int, default=300, help='Max runtime in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Read and preprocess data
        print("LOG_START:Reading and preprocessing data:LOG_END")
        df = read_and_preprocess_data(args.file)
        
        # Create time series features
        print("LOG_START:Creating time series features:LOG_END")
        df = create_time_series_features(df)
        
        # Data summary
        summary = {
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "date_range": {
                "start": df['fecha'].min().strftime('%Y-%m-%d'),
                "end": df['fecha'].max().strftime('%Y-%m-%d')
            },
            "target_stats": {
                "mean": float(df[args.target].mean()),
                "std": float(df[args.target].std()),
                "min": float(df[args.target].min()),
                "max": float(df[args.target].max())
            }
        }
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summary, indent=2))
        print("LOG_END:DATA_SUMMARY")
        
        # Initialize H2O
        print("LOG_START:Initializing H2O:LOG_END")
        h2o.init(max_mem_size='4G', nthreads=-1)
        
        # Convert to H2OFrame
        hf = h2o.H2OFrame(df)
        
        # Define features and target
        x = [col for col in hf.columns if col not in [args.target, 'fecha']]
        y = args.target
        
        # Split data
        splits = hf.split_frame(ratios=[0.8, 0.1], seed=args.seed)
        train = splits[0]
        valid = splits[1]
        test = splits[2] if len(splits) > 2 else None
        
        # Train AutoML
        print("LOG_START:Training AutoML models:LOG_END")
        aml = H2OAutoML(
            max_models=args.max_models,
            max_runtime_secs=args.max_runtime_secs,
            seed=args.seed,
            stopping_metric="RMSE"
        )
        
        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        
        # Get leader model
        leader = aml.leader
        
        # Save model
        model_path = h2o.save_model(model=leader, path=args.output_dir, force=True)
        
        # Calculate metrics
        if test:
            perf = leader.model_performance(test)
            metrics = {
                "rmse": float(perf.rmse()),
                "mae": float(perf.mae()),
                "r2": float(perf.r2()),
                "mse": float(perf.mse())
            }
        else:
            perf = leader.model_performance(valid)
            metrics = {
                "rmse": float(perf.rmse()),
                "mae": float(perf.mae()),
                "r2": float(perf.r2()),
                "mse": float(perf.mse())
            }
        
        # Final results
        total_time = time.time() - start_time
        result = {
            "model_path": model_path,
            "metrics": metrics,
            "training_time_secs": total_time,
            "rows_trained": int(hf.nrow),
            "leader_model": leader.model_id
        }
        
        # Structured outputs
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        print("METRICS_START:" + json.dumps(result, indent=2) + ":METRICS_END")
        
        print("LOG_START:Training completed successfully:LOG_END")
        return 0
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"ERROR_START:{error_msg}:ERROR_END")
        return 1
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except:
            pass

if __name__ == '__main__':
    sys.exit(main())