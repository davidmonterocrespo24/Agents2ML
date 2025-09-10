# TRAINING SCRIPT
# Generated on: 2025-09-08T15:13:26.960211
# Pipeline: pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc
# Filename: model_h2o_training.py
# Arguments: --file 3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv --output-dir . --max-rows 50000 --max-mem-size 2G
# Script Type: training

#!/usr/bin/env python3
# model_h2o_training.py

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OIsolationForestEstimator

def try_read_csv(path, sep=None):
    """Read CSV with multiple encoding fallbacks"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding)
            return df, ',', encoding
        except (UnicodeDecodeError, Exception):
            continue
    
    try:
        df = pd.read_csv(path, sep=sep, engine='python', encoding='utf-8')
        return df, ',', 'utf-8'
    except Exception as e:
        raise Exception(f"Failed to read CSV: {str(e)}")

def basic_impute(df):
    """Simple imputation for missing values"""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('UNKNOWN')
    return df

def summarize_df(df):
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_nulls": {col: int(df[col].isna().sum()) for col in df.columns}
    }

def main():
    parser = argparse.ArgumentParser(description='H2O Isolation Forest Training')
    parser.add_argument('--file', required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', default='./', help='Output directory for model')
    parser.add_argument('--max-rows', type=int, default=50000, help='Maximum rows to use for training')
    parser.add_argument('--ntrees', type=int, default=100, help='Number of trees in Isolation Forest')
    parser.add_argument('--sample-rate', type=float, default=0.8, help='Sample rate for each tree')
    parser.add_argument('--max-depth', type=int, default=8, help='Maximum tree depth')
    parser.add_argument('--max-mem-size', default='4G', help='H2O memory allocation')
    parser.add_argument('--nthreads', type=int, default=-1, help='Number of threads (-1 for all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    start_time = time.time()
    try:
        # Read and preprocess data
        print("LOG_START:LOADING_DATA")
        df, sep, encoding = try_read_csv(args.file)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print("LOG_END:LOADING_DATA")
        
        # Basic data summary
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summarize_df(df), default=str))
        print("LOG_END:DATA_SUMMARY")
        
        # Basic imputation
        df = basic_impute(df)
        
        # Sample if dataset is too large
        if len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=args.seed)
            print(f"LOG_START:SAMPLING\nSampled to {len(df)} rows\nLOG_END:SAMPLING")
        
        # Initialize H2O
        print("LOG_START:INITIALIZING_H2O")
        h2o.init(max_mem_size=args.max_mem_size, nthreads=args.nthreads)
        print("LOG_END:INITIALIZING_H2O")
        
        # Convert to H2OFrame
        print("LOG_START:CONVERTING_TO_H2O")
        hf = h2o.H2OFrame(df)
        print(f"Converted to H2OFrame with {hf.nrow} rows, {len(hf.columns)} columns")
        print("LOG_END:CONVERTING_TO_H2O")
        
        # Convert string columns to factors
        print("LOG_START:CONVERTING_FACTORS")
        for col in hf.columns:
            col_type = str(hf[col].types[0])
            if col_type == 'string':
                hf[col] = hf[col].asfactor()
                print(f"Converted {col} to factor")
        print("LOG_END:CONVERTING_FACTORS")
        
        # Train Isolation Forest for anomaly detection
        print("LOG_START:TRAINING_STARTED")
        print("Using Isolation Forest for unsupervised anomaly detection")
        print(f"Training on {hf.nrow} rows with {len(hf.columns)} features")
        print("LOG_END:TRAINING_STARTED")
        
        model = H2OIsolationForestEstimator(
            ntrees=args.ntrees,
            sample_rate=args.sample_rate,
            max_depth=args.max_depth,
            seed=args.seed
        )
        
        model.train(x=hf.columns, training_frame=hf)
        
        print("LOG_START:TRAINING_COMPLETED")
        print("Isolation Forest training completed successfully")
        print("LOG_END:TRAINING_COMPLETED")
        
        # Save model
        print("LOG_START:SAVING_MODEL")
        model_path = h2o.save_model(model=model, path=args.output_dir, force=True)
        print(f"Model saved to: {model_path}")
        print("LOG_END:SAVING_MODEL")
        
        # Generate predictions for metrics
        print("LOG_START:GENERATING_METRICS")
        predictions = model.predict(hf)
        anomaly_scores = predictions['predict'].as_data_frame()['predict'].values
        
        metrics = {
            "mean_anomaly_score": float(np.mean(anomaly_scores)),
            "std_anomaly_score": float(np.std(anomaly_scores)),
            "min_anomaly_score": float(np.min(anomaly_scores)),
            "max_anomaly_score": float(np.max(anomaly_scores)),
            "anomaly_threshold_5pct": float(np.percentile(anomaly_scores, 5)),
            "anomaly_percentage_5pct": float(np.mean(anomaly_scores <= np.percentile(anomaly_scores, 5)) * 100)
        }
        
        result = {
            "model_path": model_path,
            "metrics": metrics,
            "rows_trained": int(hf.nrow),
            "training_time_secs": time.time() - start_time,
            "h2o_version": h2o.__version__
        }
        
        print("METRICS_START:" + json.dumps(result, default=str) + ":METRICS_END")
        print("LOG_END:GENERATING_METRICS")
        
        # Structured output for pipeline
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        
        return 0
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('"', "'")
        print(f"ERROR_START:{error_msg}:ERROR_END")
        return 1
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except:
            pass

if __name__ == '__main__':
    sys.exit(main())
