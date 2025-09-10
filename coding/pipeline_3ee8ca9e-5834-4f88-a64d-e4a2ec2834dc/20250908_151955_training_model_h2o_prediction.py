# TRAINING SCRIPT
# Generated on: 2025-09-08T15:19:55.063738
# Pipeline: pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc
# Filename: model_h2o_prediction.py
# Arguments: --model-path /workspace/pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc/IsolationForest_model_python_1757355387626_1 --pipeline-dir . --data-file 3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv --output-file predictions.csv --h2o-mem 2G
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

def change_to_pipeline_dir(pipeline_dir):
    """Change to the pipeline directory"""
    os.chdir(pipeline_dir)
    return os.getcwd()

def try_read_csv(path, sep=None):
    """Read CSV with multiple encoding fallbacks"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine='python')
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
    parser.add_argument('--model-path', required=True, help='Path to the H2O model directory')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--data-file', default='3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv', help='CSV file to make predictions on')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    original_dir = os.getcwd()
    pipeline_dir = change_to_pipeline_dir(args.pipeline_dir)
    
    try:
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        
        print("LOG_START:LOADING_MODEL")
        print(f"Loading model from: {args.model_path}")
        
        # Check if model path exists
        if not os.path.exists(args.model_path):
            print(f"ERROR_START:Model path does not exist: {args.model_path}:ERROR_END")
            return 1
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
            print(f"Successfully loaded model: {type(model).__name__}")
            print("LOG_END:LOADING_MODEL")
        except Exception as e:
            print(f"ERROR_START:Failed to load H2O model: {str(e)}:ERROR_END")
            return 1
        
        # Load and preprocess the data
        print("LOG_START:LOADING_DATA")
        print(f"Loading data from: {args.data_file}")
        
        if not os.path.exists(args.data_file):
            print(f"ERROR_START:Data file does not exist: {args.data_file}:ERROR_END")
            return 1
        
        data_df, sep, enc = try_read_csv(args.data_file)
        data_df = basic_impute(data_df)
        print(f"Loaded {len(data_df)} rows, {len(data_df.columns)} columns")
        print("LOG_END:LOADING_DATA")
        
        # Convert to H2OFrame
        print("LOG_START:CONVERTING_TO_H2O")
        h2o_data = h2o.H2OFrame(data_df)
        print(f"Converted to H2OFrame with {h2o_data.nrow} rows, {len(h2o_data.columns)} columns")
        print("LOG_END:CONVERTING_TO_H2O")
        
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
        print(f"Model: {os.path.basename(args.model_path)}")
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
        output_path = os.path.join(pipeline_dir, args.output_file)
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        return 0
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ').replace('"', "'")
        print(f"ERROR_START:{error_msg}:ERROR_END")
        return 1
    finally:
        # Return to original directory and shutdown H2O
        os.chdir(original_dir)
        try:
            h2o.cluster().shutdown(prompt=False)
        except:
            pass

if __name__ == '__main__':
    sys.exit(main())
