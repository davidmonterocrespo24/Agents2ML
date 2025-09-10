#!/usr/bin/env python3
# h2o_prediction_script.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Change to pipeline directory first
os.chdir('./')

try:
    import h2o
except ImportError:
    print("ERROR_START:H2O package not installed. Please install h2o package.:ERROR_END")
    sys.exit(1)

def parse_spanish_date(date_str):
    """Parse Spanish date format with month abbreviations"""
    month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    if pd.isna(date_str):
        return pd.NaT
    
    # Clean and normalize the date string
    date_str = str(date_str).strip().lower()
    date_str = re.sub(r'\.', '', date_str)  # Remove dots from month abbreviations
    
    # Replace Spanish month abbreviations with English ones
    for esp, eng in month_mapping.items():
        if esp in date_str:
            date_str = date_str.replace(esp, eng.lower())
            break
    
    try:
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except Exception:
        return pd.NaT

def create_time_features(df: pd.DataFrame, date_col):
    """Create time-based features from date column"""
    s = pd.to_datetime(df[date_col], errors='coerce')
    df[f"{date_col}__year"] = s.dt.year
    df[f"{date_col}__month"] = s.dt.month
    df[f"{date_col}__day"] = s.dt.day
    df[f"{date_col}__dayofweek"] = s.dt.dayofweek
    df[f"{date_col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{date_col}__is_month_start"] = s.dt.is_month_start.astype(int)
    df[f"{date_col}__is_month_end"] = s.dt.is_month_end.astype(int)
    df[f"{date_col}__quarter"] = s.dt.quarter
    df[f"{date_col}__dayofyear"] = s.dt.dayofyear
    df[f"{date_col}__weekofyear"] = s.dt.isocalendar().week
    return df

def main():
    parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline directory')
    parser.add_argument('--future-dates-file', help='CSV/JSON file with future dates (optional)')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--freq', default='D', help='Frequency for date generation')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--feature-spec', help='JSON file with feature specifications')
    parser.add_argument('--output-file', default='predictions.csv', help='Output file name')
    parser.add_argument('--max-rows-sample', type=int, default=10000, help='Max rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
            print(f"LOG_START:Model loaded successfully: {args.model_path}:LOG_END")
        except Exception as e:
            print(f"ERROR_START:Failed to load model {args.model_path}: {str(e)}:ERROR_END")
            return 1
        
        # Get model information
        model_type = type(model).__name__
        print(f"LOG_START:Model type: {model_type}:LOG_END")
        
        # Prepare future data for prediction
        if args.future_dates_file:
            # Load future dates from file
            if args.future_dates_file.endswith('.csv'):
                future_df = pd.read_csv(args.future_dates_file)
            elif args.future_dates_file.endswith('.json'):
                future_df = pd.read_json(args.future_dates_file)
            else:
                print(f"ERROR_START:Unsupported file format for future dates: {args.future_dates_file}:ERROR_END")
                return 1
        else:
            # Generate future dates
            # First, get the last date from training data to continue from there
            try:
                # Read the original data to find the last date
                original_data = pd.read_csv('d304ff5d-f2ed-4c30-9955-fa8acbb291b5_ventas.csv', 
                                          sep=';', header=None, names=['fecha', 'monto_total'])
                # Handle European decimal format
                original_data['monto_total'] = original_data['monto_total'].astype(str).str.replace(',', '.').astype(float)
                # Parse Spanish dates
                original_data['fecha'] = original_data['fecha'].apply(parse_spanish_date)
                
                last_date = original_data['fecha'].max()
                print(f"LOG_START:Last date in training data: {last_date}:LOG_END")
                
                # Generate future dates starting from day after last date
                future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                           periods=args.horizon, freq=args.freq)
                
                future_df = pd.DataFrame({args.date_column: future_dates})
                print(f"LOG_START:Generated {len(future_df)} future dates:LOG_END")
                
            except Exception as e:
                print(f"ERROR_START:Failed to generate future dates: {str(e)}:ERROR_END")
                return 1
        
        # Create time features (same as during training)
        future_df = create_time_features(future_df, args.date_column)
        
        # Convert to H2OFrame
        h2o_future = h2o.H2OFrame(future_df)
        
        # Make predictions
        predictions = model.predict(h2o_future)
        
        # Combine predictions with future dates
        result_df = future_df.copy()
        result_df['prediction'] = predictions.as_data_frame().values
        
        # Save results
        output_path = os.path.join(args.pipeline_dir, args.output_file)
        result_df.to_csv(output_path, index=False)
        
        # Print success message with file path
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        # Log some metadata
        file_size = os.path.getsize(output_path)
        print(f"LOG_START:Prediction completed. Rows: {len(result_df)}, File size: {file_size} bytes:LOG_END")
        
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