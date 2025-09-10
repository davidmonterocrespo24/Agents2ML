# TRAINING SCRIPT
# Generated on: 2025-09-08T15:07:36.706198
# Pipeline: pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc
# Filename: model_h2o_prediction.py
# Arguments: --model-path ./IsolationForest_model --pipeline-dir . --output-file predictions.csv --h2o-mem 2G
# Script Type: training

#!/usr/bin/env python3
# model_h2o_prediction.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import h2o
from datetime import datetime, timedelta

def change_to_pipeline_dir(pipeline_dir):
    """Change to the pipeline directory and return the original directory"""
    original_dir = os.getcwd()
    os.chdir(pipeline_dir)
    return original_dir

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

def generate_future_dates(horizon, freq, date_column=None):
    """Generate future dates for prediction"""
    end_date = datetime.now()
    
    if freq == 'D':
        dates = pd.date_range(end=end_date, periods=horizon, freq='D')
    elif freq == 'W':
        dates = pd.date_range(end=end_date, periods=horizon, freq='W')
    elif freq == 'M':
        dates = pd.date_range(end=end_date, periods=horizon, freq='M')
    else:
        dates = pd.date_range(end=end_date, periods=horizon, freq='D')
    
    return pd.DataFrame({date_column: dates}) if date_column else pd.DataFrame(index=dates)

def apply_feature_spec(df, feature_spec):
    """Apply feature engineering based on specification"""
    if not feature_spec:
        return df
    
    try:
        spec = json.loads(feature_spec)
        
        # Apply date features if specified
        if 'date_columns' in spec:
            for date_col in spec['date_columns']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df[f"{date_col}_year"] = df[date_col].dt.year
                    df[f"{date_col}_month"] = df[date_col].dt.month
                    df[f"{date_col}_day"] = df[date_col].dt.day
                    df[f"{date_col}_dayofweek"] = df[date_col].dt.dayofweek
        
        return df
    except Exception as e:
        print(f"LOG_START:FEATURE_SPEC_WARNING\nFailed to apply feature spec: {str(e)}\nLOG_END:FEATURE_SPEC_WARNING")
        return df

def validate_columns(h2o_frame, model):
    """Validate that required columns exist in the prediction data"""
    try:
        model_features = model._model_json['output']['names']
        frame_features = h2o_frame.columns
        
        missing_features = set(model_features) - set(frame_features)
        if missing_features:
            return False, f"Missing features in prediction data: {missing_features}"
        
        return True, "All features present"
    except Exception as e:
        return False, f"Error validating columns: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
    parser.add_argument('--model-path', required=True, help='Path to the saved H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--future-dates-file', help='CSV/JSON file with future dates for prediction')
    parser.add_argument('--horizon', type=int, default=30, help='Number of periods to predict')
    parser.add_argument('--freq', default='D', help='Frequency of predictions (D, W, M)')
    parser.add_argument('--date-column', help='Name of date column for time series')
    parser.add_argument('--feature-spec', help='JSON string specifying feature engineering')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--max-rows-sample', type=int, default=10000, help='Max rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    original_dir = change_to_pipeline_dir(args.pipeline_dir)
    
    try:
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        
        print("LOG_START:LOADING_MODEL")
        print(f"Loading model from: {args.model_path}")
        print("LOG_END:LOADING_MODEL")
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
        except Exception as e:
            error_msg = f"Failed to load H2O model: {str(e)}. This might be a MOJO model which requires different loading approach."
            print(f"ERROR_START:{error_msg}:ERROR_END")
            return 1
        
        print("LOG_START:MODEL_LOADED")
        print(f"Model type: {type(model).__name__}")
        print(f"Model features: {model._model_json['output']['names'] if hasattr(model, '_model_json') else 'Unknown'}")
        print("LOG_END:MODEL_LOADED")
        
        # Prepare prediction data
        if args.future_dates_file:
            # Load from file
            if args.future_dates_file.endswith('.csv'):
                pred_data, sep, enc = try_read_csv(args.future_dates_file)
            elif args.future_dates_file.endswith('.json'):
                pred_data = pd.read_json(args.future_dates_file)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        else:
            # Generate future dates
            pred_data = generate_future_dates(args.horizon, args.freq, args.date_column)
        
        # Apply feature specification if provided
        pred_data = apply_feature_spec(pred_data, args.feature_spec)
        
        # Sample if too large
        if len(pred_data) > args.max_rows_sample:
            pred_data = pred_data.sample(n=args.max_rows_sample, random_state=args.seed)
        
        # Convert to H2OFrame
        h2o_pred_data = h2o.H2OFrame(pred_data)
        
        # Validate columns
        is_valid, validation_msg = validate_columns(h2o_pred_data, model)
        if not is_valid:
            print(f"ERROR_START:{validation_msg}:ERROR_END")
            return 1
        
        print("LOG_START:MAKING_PREDICTIONS")
        print(f"Making predictions on {h2o_pred_data.nrow} rows")
        print("LOG_END:MAKING_PREDICTIONS")
        
        # Make predictions
        predictions = model.predict(h2o_pred_data)
        
        # Convert to pandas and save
        predictions_df = predictions.as_data_frame()
        
        # Combine with original data if available
        if not pred_data.empty:
            original_df = pred_data.reset_index(drop=True)
            result_df = pd.concat([original_df, predictions_df], axis=1)
        else:
            result_df = predictions_df
        
        # Save predictions
        result_df.to_csv(args.output_file, index=False)
        
        # Print success message
        output_path = os.path.join(args.pipeline_dir, args.output_file)
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        # Log final details
        print("LOG_START:PREDICTION_SUMMARY")
        print(f"Model: {os.path.basename(args.model_path)}")
        print(f"Rows predicted: {len(result_df)}")
        print(f"Output file: {args.output_file}")
        print(f"File size: {os.path.getsize(args.output_file)} bytes")
        print("LOG_END:PREDICTION_SUMMARY")
        
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
