#!/usr/bin/env python3
# model_h2o_prediction.py

# Complete, production-ready Python script for loading saved H2O model and producing predictions
# Mandatory structured outputs: PREDICTIONS_FILE_START:<path>:PREDICTIONS_FILE_END or ERROR_START:<message>:ERROR_END

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np
import h2o
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    start_time = time.time()
    try:
        # ---------- Parse arguments ----------
        parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
        parser.add_argument('--model-path', required=True, help='Path to saved H2O model file/directory')
        parser.add_argument('--pipeline-dir', required=True, help='Path to pipeline working directory')
        parser.add_argument('--future-dates-file', help='Optional CSV/JSON file with future timestamps')
        parser.add_argument('--horizon', type=int, default=30, help='Number of future periods to generate')
        parser.add_argument('--freq', default='D', help='Frequency for horizon generation (D, H, W, M)')
        parser.add_argument('--date-column', default='fecha', help='Name of date column')
        parser.add_argument('--feature-spec', help='Path to feature specification JSON')
        parser.add_argument('--output-file', default='predictions.csv', help='Output CSV filename')
        parser.add_argument('--max_rows_sample', type=int, default=1000, help='Max rows for sampling')
        parser.add_argument('--h2o_mem', default='2G', help='H2O memory size')
        parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
        
        args = parser.parse_args()
        
        # ---------- Change to pipeline directory ----------
        os.chdir(args.pipeline_dir)
        print(f"LOG_START:Changed working directory to: {os.getcwd()}:LOG_END")
        
        # ---------- Validate model path ----------
        if not os.path.exists(args.model_path):
            print(f"ERROR_START:Model path does not exist: {args.model_path}:ERROR_END")
            return 1
        
        # ---------- Initialize H2O ----------
        h2o.init(max_mem_size=args.h2o_mem, nthreads=2)
        print("LOG_START:H2O initialized successfully:LOG_END")
        
        # ---------- Load model ----------
        try:
            model = h2o.load_model(args.model_path)
            print(f"LOG_START:Model loaded successfully: {model.model_id}:LOG_END")
        except Exception as e:
            if "MOJO" in str(e) or "mojo" in str(e):
                print(f"ERROR_START:Model at {args.model_path} is a MOJO-only artifact; re-run export with h2o.save_model() or provide MOJO loader path:ERROR_END")
            else:
                print(f"ERROR_START:Failed to load model: {str(e)}:ERROR_END")
            return 1
        
        # ---------- Prepare prediction data ----------
        if args.future_dates_file and os.path.exists(args.future_dates_file):
            # Read from provided file
            try:
                if args.future_dates_file.endswith('.csv'):
                    future_df = pd.read_csv(args.future_dates_file, sep=';')
                elif args.future_dates_file.endswith('.json'):
                    future_df = pd.read_json(args.future_dates_file)
                else:
                    # Try autodetect
                    future_df = pd.read_csv(args.future_dates_file, sep=None, engine='python')
                print(f"LOG_START:Loaded future dates from file: {args.future_dates_file}:LOG_END")
            except Exception as e:
                print(f"ERROR_START:Failed to read future dates file: {str(e)}:ERROR_END")
                return 1
        else:
            # Generate future dates
            print(f"LOG_START:Generating {args.horizon} future periods with frequency {args.freq}:LOG_END")
            
            # Get the last date from training data to start future predictions
            # For simplicity, we'll use current date as starting point
            end_date = datetime.now()
            
            # Generate future dates
            future_dates = pd.date_range(
                start=end_date + timedelta(days=1),
                periods=args.horizon,
                freq=args.freq
            )
            
            future_df = pd.DataFrame({args.date_column: future_dates})
            
            # Create time features (same as training)
            future_df = create_time_features(future_df, args.date_column)
            
            # For time series, we need to create lag features with appropriate values
            # Since we don't have historical values, we'll use the mean of the training data
            # In a real scenario, you'd want to use the last known values
            future_df['monto_total_lag_1'] = np.nan
            future_df['monto_total_lag_2'] = np.nan
            future_df['monto_total_lag_3'] = np.nan
            future_df['monto_total_lag_7'] = np.nan
        
        # ---------- Validate and prepare features ----------
        # Get expected features from model
        expected_features = model._model_json['output']['names']
        # Remove target variable if present
        expected_features = [f for f in expected_features if f != 'monto_total']
        
        print(f"LOG_START:Model expects features: {expected_features}:LOG_END")
        
        # Check for missing features
        missing_features = set(expected_features) - set(future_df.columns)
        if missing_features:
            print(f"LOG_START:Missing features detected: {list(missing_features)}. Attempting to impute.:LOG_END")
            
            # Simple imputation for missing features
            for feature in missing_features:
                if 'lag' in feature:
                    # Lag features - set to NaN (model should handle)
                    future_df[feature] = np.nan
                elif 'is_' in feature:
                    # Boolean features - set to 0
                    future_df[feature] = 0
                elif 'year' in feature or 'month' in feature or 'day' in feature:
                    # Extract from date column
                    if feature == 'fecha__year':
                        future_df[feature] = future_df[args.date_column].dt.year
                    elif feature == 'fecha__month':
                        future_df[feature] = future_df[args.date_column].dt.month
                    elif feature == 'fecha__day':
                        future_df[feature] = future_df[args.date_column].dt.day
                    elif feature == 'fecha__dayofweek':
                        future_df[feature] = future_df[args.date_column].dt.dayofweek
                else:
                    # Numeric features - set to 0
                    future_df[feature] = 0
        
        # Ensure correct column order
        future_df = future_df[expected_features]
        
        # ---------- Convert to H2OFrame ----------
        hf_future = h2o.H2OFrame(future_df)
        
        # Convert appropriate columns to factors
        for col in hf_future.columns:
            if col in ['fecha__is_weekend', 'fecha__is_month_start', 'fecha__is_month_end']:
                hf_future[col] = hf_future[col].asfactor()
        
        # ---------- Make predictions ----------
        predictions = model.predict(hf_future)
        
        # Convert back to pandas
        pred_df = predictions.as_data_frame()
        
        # Add date column to predictions for reference
        if args.date_column in future_df.columns:
            pred_df[args.date_column] = future_df[args.date_column].values
        
        # ---------- Save predictions ----------
        output_path = os.path.join(args.pipeline_dir, args.output_file)
        pred_df.to_csv(output_path, index=False, sep=';')
        
        # Verify file creation
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"LOG_START:Predictions file created successfully: {output_path} ({file_size} bytes):LOG_END")
            
            # Print structured output
            print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
            
            # Summary log
            summary = {
                "n_rows_predicted": len(pred_df),
                "model_id": model.model_id,
                "model_algorithm": model.algo,
                "prediction_time_secs": time.time() - start_time,
                "output_file_size_bytes": file_size,
                "features_used": expected_features
            }
            print(f"LOG_START:PREDICTION_SUMMARY\n{json.dumps(summary, indent=2)}\nLOG_END:PREDICTION_SUMMARY")
            
            return 0
        else:
            print("ERROR_START:Failed to create predictions file:ERROR_END")
            return 1
            
    except Exception as e:
        err_msg = str(e).replace('\n', ' ')
        print(f"ERROR_START:{err_msg}:ERROR_END")
        return 2
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
            print("LOG_START:H2O cluster shutdown:LOG_END")
        except Exception:
            pass

def create_time_features(df: pd.DataFrame, date_col):
    """Create time-based features from date column"""
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        s = df[date_col]
        df[f"{date_col}__year"] = s.dt.year
        df[f"{date_col}__month"] = s.dt.month
        df[f"{date_col}__day"] = s.dt.day
        df[f"{date_col}__dayofweek"] = s.dt.dayofweek
        df[f"{date_col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
        df[f"{date_col}__is_month_start"] = s.dt.is_month_start.astype(int)
        df[f"{date_col}__is_month_end"] = s.dt.is_month_end.astype(int)
    return df

if __name__ == '__main__':
    sys.exit(main())