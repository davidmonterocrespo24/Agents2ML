#!/usr/bin/env python3
# h2o_prediction_script.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import h2o
from datetime import datetime, timedelta

def parse_spanish_date(date_str):
    """Parse Spanish date format like '16 nov. 2023'"""
    month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
        'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    try:
        date_str_clean = date_str.replace('.', '')
        parts = date_str_clean.split()
        if len(parts) >= 3:
            day = parts[0]
            month_abbr = parts[1][:3].lower()
            year = parts[2]
            
            if month_abbr in month_mapping:
                english_date_str = f"{day} {month_mapping[month_abbr]} {year}"
                return pd.to_datetime(english_date_str, dayfirst=True)
        
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except:
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')

def create_time_features(df: pd.DataFrame, col, target_series=None):
    """Create time-based features including lag features"""
    s = pd.to_datetime(df[col], errors='coerce')
    df[f"{col}__year"] = s.dt.year
    df[f"{col}__month"] = s.dt.month
    df[f"{col}__day"] = s.dt.day
    df[f"{col}__dayofweek"] = s.dt.dayofweek
    df[f"{col}__dayofyear"] = s.dt.dayofyear
    df[f"{col}__week"] = s.dt.isocalendar().week
    df[f"{col}__quarter"] = s.dt.quarter
    df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{col}__is_month_start"] = s.dt.is_month_start.astype(int)
    df[f"{col}__is_month_end"] = s.dt.is_month_end.astype(int)
    
    # For lag features, we need historical target data
    if target_series is not None and len(target_series) > 0:
        target_mean = target_series.mean()
        df[f"{col}__lag1"] = target_series.shift(1).iloc[-1] if len(target_series) > 0 else target_mean
        df[f"{col}__lag7"] = target_series.shift(7).iloc[-1] if len(target_series) > 6 else target_mean
        df[f"{col}__rolling_mean7"] = target_series.rolling(window=7, min_periods=1).mean().iloc[-1]
    else:
        # Default values if no historical data
        df[f"{col}__lag1"] = 0
        df[f"{col}__lag7"] = 0
        df[f"{col}__rolling_mean7"] = 0
    
    return df

def main():
    parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline directory')
    parser.add_argument('--future-dates-file', help='CSV/JSON file with future dates')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--freq', default='D', help='Frequency for date generation')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--feature-spec', help='JSON file with feature specifications')
    parser.add_argument('--output-file', default='predictions.csv', help='Output file name')
    parser.add_argument('--max-rows-sample', type=int, default=1000, help='Max rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    try:
        os.chdir(args.pipeline_dir)
        print(f"LOG_START:WORKING_DIRECTORY\nChanged to: {os.getcwd()}\nLOG_END:WORKING_DIRECTORY")
    except Exception as e:
        print(f"ERROR_START:Failed to change to pipeline directory: {str(e)}:ERROR_END")
        sys.exit(1)
    
    try:
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        print("LOG_START:H2O_INITIALIZED\nH2O cluster started successfully\nLOG_END:H2O_INITIALIZED")
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
            print(f"LOG_START:MODEL_LOADED\nModel: {model.model_id}\nType: {type(model).__name__}\nLOG_END:MODEL_LOADED")
        except Exception as e:
            print(f"ERROR_START:Failed to load H2O model: {str(e)}:ERROR_END")
            sys.exit(1)
        
        # Load historical data to get last values for lag features
        try:
            historical_df = pd.read_csv('fe7ccf65-e073-4c98-ba7d-6740e42a87cf_ventas.csv', 
                                      sep=';', header=None, decimal=',')
            historical_df.columns = ['fecha', 'monto total']
            historical_df['fecha'] = historical_df['fecha'].apply(parse_spanish_date)
            historical_df = historical_df.sort_values('fecha')
            last_target_values = historical_df['monto total']
            last_date = historical_df['fecha'].max()
            print(f"LOG_START:HISTORICAL_DATA_LOADED\nRows: {len(historical_df)}\nLast date: {last_date}\nLOG_END:HISTORICAL_DATA_LOADED")
        except Exception as e:
            print(f"LOG_START:HISTORICAL_DATA_WARNING\nCould not load historical data for lag features: {str(e)}\nLOG_END:HISTORICAL_DATA_WARNING")
            last_target_values = None
            last_date = datetime.now()
        
        # Generate future dates
        if args.future_dates_file:
            try:
                if args.future_dates_file.endswith('.csv'):
                    future_dates = pd.read_csv(args.future_dates_file)
                else:
                    with open(args.future_dates_file, 'r') as f:
                        future_dates = pd.DataFrame(json.load(f))
                print(f"LOG_START:FUTURE_DATES_LOADED\nFile: {args.future_dates_file}\nRows: {len(future_dates)}\nLOG_END:FUTURE_DATES_LOADED")
            except Exception as e:
                print(f"ERROR_START:Failed to load future dates file: {str(e)}:ERROR_END")
                sys.exit(1)
        else:
            # Generate future dates automatically
            start_date = last_date + timedelta(days=1) if last_date else datetime.now()
            future_dates = pd.DataFrame({
                args.date_column: pd.date_range(start=start_date, periods=args.horizon, freq=args.freq)
            })
            print(f"LOG_START:FUTURE_DATES_GENERATED\nHorizon: {args.horizon}\nStart: {start_date}\nFreq: {args.freq}\nLOG_END:FUTURE_DATES_GENERATED")
        
        # Create features for future dates
        prediction_df = future_dates.copy()
        prediction_df = create_time_features(prediction_df, args.date_column, last_target_values)
        
        # Convert to H2OFrame
        h2o_pred_frame = h2o.H2OFrame(prediction_df)
        
        # Make predictions
        predictions = model.predict(h2o_pred_frame)
        
        # Combine predictions with dates
        result_df = future_dates.copy()
        result_df['prediction'] = predictions.as_data_frame().values
        result_df['fecha'] = result_df['fecha'].dt.strftime('%d %b. %Y')
        
        # Save predictions
        result_df.to_csv(args.output_file, index=False)
        file_size = os.path.getsize(args.output_file)
        print(f"LOG_START:PREDICTIONS_SAVED\nFile: {args.output_file}\nRows: {len(result_df)}\nSize: {file_size} bytes\nLOG_END:PREDICTIONS_SAVED")
        
        # Success output
        print(f"PREDICTIONS_FILE_START:{os.path.abspath(args.output_file)}:PREDICTIONS_FILE_END")
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"ERROR_START:{error_msg}:ERROR_END")
        sys.exit(1)
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except:
            pass

if __name__ == '__main__':
    main()