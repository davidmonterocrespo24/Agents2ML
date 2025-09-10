# PREDICTION SCRIPT
# Generated on: 2025-09-05T11:04:10.515922
# Pipeline: pipeline_eeaff94b-29d3-4e76-b80d-a2e7513699a8
# Filename: prediction_script.py
# Arguments: --model-path GBM_4_AutoML_1_20250905_135824 --pipeline-dir . --horizon 30 --freq D --output-file predictions.csv
# Script Type: prediction

#!/usr/bin/env python3
# prediction_script.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Check if H2O is available
try:
    import h2o
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    print("ERROR_START:H2O library is not available:ERROR_END")
    sys.exit(1)

def change_to_pipeline_dir(pipeline_dir):
    """Change to pipeline directory and validate"""
    try:
        os.chdir(pipeline_dir)
        print(f"LOG_START:Changed to pipeline directory: {os.getcwd()}:LOG_END")
        return True
    except Exception as e:
        print(f"ERROR_START:Failed to change to pipeline directory {pipeline_dir}: {str(e)}:ERROR_END")
        return False

def load_h2o_model(model_path):
    """Load H2O model with proper error handling"""
    try:
        if not os.path.exists(model_path):
            print(f"ERROR_START:Model file not found at {model_path}:ERROR_END")
            return None
        
        print(f"LOG_START:Loading model from {model_path}:LOG_END")
        model = h2o.load_model(model_path)
        print(f"LOG_START:Model loaded successfully: {model.model_id}:LOG_END")
        return model
    except Exception as e:
        print(f"ERROR_START:Failed to load model {model_path}: {str(e)}:ERROR_END")
        return None

def generate_future_dates(last_date, horizon, freq):
    """Generate future dates for prediction"""
    if freq == 'D':
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    elif freq == 'W':
        dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=horizon, freq='W-MON')
    elif freq == 'M':
        dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='M')
    else:
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    
    return dates

def create_time_series_features_from_dates(dates, historical_data=None):
    """Create time series features from date series"""
    df = pd.DataFrame({'fecha': dates})
    
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
    
    # If historical data is provided, calculate lag features
    if historical_data is not None and not historical_data.empty:
        last_values = historical_data['monto_total'].tail(30).values
        if len(last_values) >= 30:
            df['lag_1'] = last_values[-1]
            df['lag_7'] = historical_data['monto_total'].tail(7).mean()
            df['lag_30'] = historical_data['monto_total'].tail(30).mean()
        else:
            # Use available data
            df['lag_1'] = last_values[-1] if len(last_values) > 0 else 0
            df['lag_7'] = last_values.mean() if len(last_values) > 0 else 0
            df['lag_30'] = last_values.mean() if len(last_values) > 0 else 0
        
        # Rolling statistics
        if len(last_values) >= 7:
            df['rolling_mean_7'] = historical_data['monto_total'].tail(7).mean()
        else:
            df['rolling_mean_7'] = last_values.mean() if len(last_values) > 0 else 0
        
        if len(last_values) >= 30:
            df['rolling_mean_30'] = historical_data['monto_total'].tail(30).mean()
        else:
            df['rolling_mean_30'] = last_values.mean() if len(last_values) > 0 else 0
    else:
        # Default values if no historical data
        df['lag_1'] = 0
        df['lag_7'] = 0
        df['lag_30'] = 0
        df['rolling_mean_7'] = 0
        df['rolling_mean_30'] = 0
    
    return df

def validate_columns(prediction_data, model):
    """Validate that prediction data has all required columns"""
    try:
        # Get required columns from model
        required_columns = model._model_json['output']['names']
        
        # Check if all required columns are present
        missing_columns = set(required_columns) - set(prediction_data.columns)
        
        if missing_columns:
            print(f"LOG_START:Missing columns: {list(missing_columns)}:LOG_END")
            # Try to impute missing columns with default values
            for col in missing_columns:
                if col not in prediction_data.columns:
                    if 'lag' in col or 'rolling' in col:
                        prediction_data[col] = 0
                    elif 'is_' in col:
                        prediction_data[col] = 0
                    else:
                        prediction_data[col] = 0
            print("LOG_START:Imputed missing columns with default values:LOG_END")
        
        return True
        
    except Exception as e:
        print(f"ERROR_START:Column validation failed: {str(e)}:ERROR_END")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using trained H2O model')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--future-dates-file', help='CSV/JSON file with future dates (optional)')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon')
    parser.add_argument('--freq', default='D', choices=['D', 'W', 'M'], help='Frequency of predictions')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--output-file', default='predictions.csv', help='Output file name')
    parser.add_argument('--max-rows-sample', type=int, default=1000, help='Max rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Change to pipeline directory first
    if not change_to_pipeline_dir(args.pipeline_dir):
        return 1
    
    try:
        # Initialize H2O
        print("LOG_START:Initializing H2O:LOG_END")
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        
        # Load the model
        model = load_h2o_model(args.model_path)
        if model is None:
            return 1
        
        # Read historical data to get the last date
        historical_data = None
        try:
            historical_data = pd.read_csv('eeaff94b-29d3-4e76-b80d-a2e7513699a8_ventas.csv', 
                                        sep=';', 
                                        header=None,
                                        names=['fecha', 'monto_total'])
            
            # Convert Spanish dates and numeric format
            month_mapping = {
                'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
                'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
            }
            
            def convert_spanish_date(date_str):
                import re
                for esp, eng in month_mapping.items():
                    date_str = re.sub(rf'\b{esp}\b', eng, date_str, flags=re.IGNORECASE)
                return date_str
            
            historical_data['fecha'] = historical_data['fecha'].apply(convert_spanish_date)
            historical_data['fecha'] = pd.to_datetime(historical_data['fecha'], format='%d %b. %Y', errors='coerce')
            historical_data['monto_total'] = historical_data['monto_total'].astype(str).str.replace('.', '', regex=False)
            historical_data['monto_total'] = historical_data['monto_total'].str.replace(',', '.', regex=False).astype(float)
            
            last_date = historical_data['fecha'].max()
            print(f"LOG_START:Last historical date: {last_date}:LOG_END")
            
        except Exception as e:
            print(f"LOG_START:Warning: Could not read historical data: {str(e)}:LOG_END")
            # Use a default last date if historical data cannot be read
            last_date = pd.to_datetime('2025-08-07')
        
        # Generate or load future dates
        if args.future_dates_file and os.path.exists(args.future_dates_file):
            print(f"LOG_START:Loading future dates from {args.future_dates_file}:LOG_END")
            future_dates = pd.read_csv(args.future_dates_file)
            if args.date_column in future_dates.columns:
                future_dates[args.date_column] = pd.to_datetime(future_dates[args.date_column])
        else:
            print(f"LOG_START:Generating {args.horizon} future dates with frequency {args.freq}:LOG_END")
            future_dates = generate_future_dates(last_date, args.horizon, args.freq)
        
        # Create prediction data with features
        print("LOG_START:Creating time series features for prediction:LOG_END")
        prediction_data = create_time_series_features_from_dates(future_dates, historical_data)
        
        # Validate columns against model requirements
        if not validate_columns(prediction_data, model):
            return 1
        
        # Convert to H2OFrame
        print("LOG_START:Converting to H2OFrame:LOG_END")
        h2o_pred_data = h2o.H2OFrame(prediction_data)
        
        # Make predictions
        print("LOG_START:Generating predictions:LOG_END")
        predictions = model.predict(h2o_pred_data)
        
        # Combine predictions with dates
        result_df = pd.DataFrame({
            'fecha': future_dates,
            'predicted_monto_total': predictions.as_data_frame().values.flatten()
        })
        
        # Save predictions
        output_path = args.output_file
        result_df.to_csv(output_path, index=False)
        
        # Print success message with file info
        file_size = os.path.getsize(output_path)
        print(f"LOG_START:Predictions saved: {len(result_df)} rows, file size: {file_size} bytes:LOG_END")
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
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
