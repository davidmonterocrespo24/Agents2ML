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

def main():
    parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--future-dates-file', required=False, help='CSV/JSON file with future dates')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon (days)')
    parser.add_argument('--freq', default='D', help='Frequency for date generation')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--feature-spec', required=False, help='JSON file with feature specifications')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--max-rows-sample', type=int, default=1000, help='Max rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Change to pipeline directory
        os.chdir(args.pipeline_dir)
        print(f"LOG_START:WORKING_DIR\nWorking in: {os.getcwd()}\nLOG_END:WORKING_DIR")
        
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        print("LOG_START:H2O_INIT\nH2O initialized successfully\nLOG_END:H2O_INIT")
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
            print(f"LOG_START:MODEL_LOADED\nModel: {args.model_path}\nModel type: {type(model).__name__}\nLOG_END:MODEL_LOADED")
        except Exception as e:
            print(f"ERROR_START:Failed to load model: {str(e)}:ERROR_END")
            return 1
        
        # Generate future dates
        if args.future_dates_file and os.path.exists(args.future_dates_file):
            try:
                if args.future_dates_file.endswith('.csv'):
                    future_df = pd.read_csv(args.future_dates_file)
                else:
                    future_df = pd.read_json(args.future_dates_file)
                print(f"LOG_START:FUTURE_DATES_LOADED\nRows: {len(future_df)}\nFrom file: {args.future_dates_file}\nLOG_END:FUTURE_DATES_LOADED")
            except Exception as e:
                print(f"ERROR_START:Failed to load future dates file: {str(e)}:ERROR_END")
                return 1
        else:
            # Generate future dates automatically
            end_date = datetime.now() + timedelta(days=args.horizon)
            future_dates = pd.date_range(start=datetime.now(), end=end_date, freq=args.freq)
            future_df = pd.DataFrame({args.date_column: future_dates})
            print(f"LOG_START:FUTURE_DATES_GENERATED\nHorizon: {args.horizon} days\nRows: {len(future_df)}\nLOG_END:FUTURE_DATES_GENERATED")
        
        # Create time features (same as training)
        def create_time_features(df, col):
            s = pd.to_datetime(df[col], errors='coerce')
            df[f"{col}__year"] = s.dt.year
            df[f"{col}__month"] = s.dt.month
            df[f"{col}__day"] = s.dt.day
            df[f"{col}__dayofweek"] = s.dt.dayofweek
            df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
            df[f"{col}__quarter"] = s.dt.quarter
            return df
        
        # Apply feature engineering
        future_df = create_time_features(future_df, args.date_column)
        
        # Remove original date column (same as training)
        if args.date_column in future_df.columns:
            future_df = future_df.drop(columns=[args.date_column])
        
        # Convert to H2OFrame
        h2o_future = h2o.H2OFrame(future_df)
        print(f"LOG_START:DATA_PREPARED\nFeatures: {h2o_future.columns}\nRows: {h2o_future.nrow}\nLOG_END:DATA_PREPARED")
        
        # Make predictions
        predictions = model.predict(h2o_future)
        
        # Convert to pandas and combine with future dates
        pred_df = predictions.as_data_frame()
        future_dates_only = pd.DataFrame({'date': pd.date_range(start=datetime.now(), 
                                                              periods=len(future_df), 
                                                              freq=args.freq)})
        result_df = pd.concat([future_dates_only, pred_df], axis=1)
        result_df.columns = ['date', 'prediction']
        
        # Save predictions
        output_path = os.path.join(args.pipeline_dir, args.output_file)
        result_df.to_csv(output_path, index=False)
        
        print(f"LOG_START:PREDICTIONS_SAVED\nFile: {output_path}\nRows: {len(result_df)}\nLOG_END:PREDICTIONS_SAVED")
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