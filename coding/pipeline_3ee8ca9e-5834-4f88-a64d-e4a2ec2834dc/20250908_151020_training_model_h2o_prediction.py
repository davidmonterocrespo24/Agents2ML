# TRAINING SCRIPT
# Generated on: 2025-09-08T15:10:20.650494
# Pipeline: pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc
# Filename: model_h2o_prediction.py
# Arguments: --model-dir . --data-file 3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv --output-file predictions.csv --h2o-mem 2G
# Script Type: training

#!/usr/bin/env python3
# model_h2o_prediction.py

import argparse
import json
import os
import sys
import glob
import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OIsolationForestEstimator

def find_model_files(model_path):
    """Find actual model files in the directory"""
    if os.path.isdir(model_path):
        # Look for model files in the directory
        model_files = []
        for ext in ['*.bin', '*.zip', '*.mojo']:
            model_files.extend(glob.glob(os.path.join(model_path, ext)))
        return model_files
    elif os.path.isfile(model_path):
        return [model_path]
    return []

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

def main():
    parser = argparse.ArgumentParser(description='H2O Isolation Forest Prediction')
    parser.add_argument('--model-dir', required=True, help='Directory containing the H2O model')
    parser.add_argument('--data-file', required=True, help='CSV file to make predictions on')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    
    args = parser.parse_args()
    
    try:
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        
        print("LOG_START:SEARCHING_MODEL")
        print(f"Looking for model files in: {args.model_dir}")
        
        # Find model files
        model_files = find_model_files(args.model_dir)
        if not model_files:
            print(f"ERROR_START:No model files found in {args.model_dir}:ERROR_END")
            return 1
        
        print(f"Found model files: {model_files}")
        print("LOG_END:SEARCHING_MODEL")
        
        # Try to load the model
        model = None
        for model_file in model_files:
            try:
                print(f"LOG_START:LOADING_ATTEMPT\nTrying to load: {model_file}\nLOG_END:LOADING_ATTEMPT")
                model = h2o.load_model(model_file)
                print(f"LOG_START:MODEL_LOADED\nSuccessfully loaded model from {model_file}\nLOG_END:MODEL_LOADED")
                break
            except Exception as e:
                print(f"LOG_START:LOAD_FAILED\nFailed to load {model_file}: {str(e)}\nLOG_END:LOAD_FAILED")
                continue
        
        if model is None:
            print("ERROR_START:Failed to load any model files:ERROR_END")
            return 1
        
        # Load and preprocess the data
        print("LOG_START:LOADING_DATA")
        print(f"Loading data from: {args.data_file}")
        data_df, sep, enc = try_read_csv(args.data_file)
        data_df = basic_impute(data_df)
        print(f"Loaded {len(data_df)} rows, {len(data_df.columns)} columns")
        print("LOG_END:LOADING_DATA")
        
        # Convert to H2OFrame
        h2o_data = h2o.H2OFrame(data_df)
        
        # Make predictions
        print("LOG_START:MAKING_PREDICTIONS")
        predictions = model.predict(h2o_data)
        print("LOG_END:MAKING_PREDICTIONS")
        
        # Convert to pandas and combine with original data
        pred_df = predictions.as_data_frame()
        result_df = pd.concat([data_df.reset_index(drop=True), pred_df], axis=1)
        
        # Save predictions
        result_df.to_csv(args.output_file, index=False)
        
        print("LOG_START:PREDICTION_SUMMARY")
        print(f"Model: {os.path.basename(model_files[0])}")
        print(f"Input data: {args.data_file}")
        print(f"Rows predicted: {len(result_df)}")
        print(f"Output file: {args.output_file}")
        print(f"File size: {os.path.getsize(args.output_file)} bytes")
        
        # Calculate anomaly statistics
        if 'predict' in pred_df.columns:
            anomaly_scores = pred_df['predict']
            threshold = np.percentile(anomaly_scores, 5)
            anomalies = anomaly_scores <= threshold
            print(f"Anomaly threshold (5th percentile): {threshold:.4f}")
            print(f"Potential anomalies: {anomalies.sum()} ({anomalies.mean()*100:.2f}%)")
        
        print("LOG_END:PREDICTION_SUMMARY")
        
        # Structured output for pipeline
        print(f"PREDICTIONS_FILE_START:{os.path.abspath(args.output_file)}:PREDICTIONS_FILE_END")
        
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
