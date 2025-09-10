# PREDICTION SCRIPT
# Generated on: 2025-09-04T15:47:31.282667
# Pipeline: pipeline_e3e0f8c3-d0c6-4620-9df5-fdf5ac58d4e8
# Filename: prediction_script.py
# Arguments: --model-path /workspace/pipeline_e3e0f8c3-d0c6-4620-9df5-fdf5ac58d4e8/DeepLearning_grid_1_AutoML_1_20250904_184011_model_7 --pipeline-dir /workspace/pipeline_e3e0f8c3-d0c6-4620-9df5-fdf5ac58d4e8 --horizon 30 --freq D --date-column fecha --output-file predictions.csv --h2o-mem 2G --seed 42
# Script Type: prediction

#!/usr/bin/env python3
# h2o_prediction.py

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import h2o
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='H2O Model Prediction Script')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--future-dates-file', help='CSV/JSON file with future dates (optional)')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--freq', default='D', help='Frequency for date generation')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--feature-spec', help='JSON file with feature specifications')
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
            print(f"LOG_START:MODEL_LOADED\nModel loaded: {args.model_path}\nModel type: {type(model).__name__}\nLOG_END:MODEL_LOADED")
        except Exception as e:
            error_msg = f"Failed to load H2O model: {str(e)}"
            print(f"ERROR_START:{error_msg}:ERROR_END")
            return 1
        
        # Get model information
        model_features = model._model_json['output']['names'] if hasattr(model, '_model_json') else []
        target_column = model._model_json['output']['response_column'] if hasattr(model, '_model_json') else 'predict'
        print(f"LOG_START:MODEL_INFO\nFeatures: {model_features}\nTarget: {target_column}\nLOG_END:MODEL_INFO")
        
        # Generate future dates
        if args.future_dates_file:
            try:
                if args.future_dates_file.endswith('.csv'):
                    future_dates_df = pd.read_csv(args.future_dates_file)
                elif args.future_dates_file.endswith('.json'):
                    future_dates_df = pd.read_json(args.future_dates_file)
                else:
                    raise ValueError("Unsupported future dates file format")
                print(f"LOG_START:FUTURE_DATES_LOADED\nLoaded {len(future_dates_df)} future dates from file\nLOG_END:FUTURE_DATES_LOADED")
            except Exception as e:
                error_msg = f"Failed to load future dates file: {str(e)}"
                print(f"ERROR_START:{error_msg}:ERROR_END")
                return 1
        else:
            # Generate future dates automatically
            end_date = datetime.now() + timedelta(days=args.horizon)
            future_dates = pd.date_range(start=datetime.now(), end=end_date, freq=args.freq)
            future_dates_df = pd.DataFrame({args.date_column: future_dates})
            print(f"LOG_START:FUTURE_DATES_GENERATED\nGenerated {len(future_dates_df)} future dates\nLOG_END:FUTURE_DATES_GENERATED")
        
        # Create time-based features
        def create_time_features(df, date_col):
            df = df.copy()
            df[f'{date_col}__year'] = df[date_col].dt.year
            df[f'{date_col}__month'] = df[date_col].dt.month
            df[f'{date_col}__day'] = df[date_col].dt.day
            df[f'{date_col}__dayofweek'] = df[date_col].dt.dayofweek
            df[f'{date_col}__is_weekend'] = df[date_col].dt.dayofweek.isin([5,6]).astype(int)
            df[f'{date_col}__is_month_start'] = df[date_col].dt.is_month_start.astype(int)
            df[f'{date_col}__is_month_end'] = df[date_col].dt.is_month_end.astype(int)
            return df
        
        # Apply feature engineering
        future_dates_df[args.date_column] = pd.to_datetime(future_dates_df[args.date_column])
        future_dates_df = create_time_features(future_dates_df, args.date_column)
        
        # Apply feature spec if provided
        if args.feature_spec:
            try:
                with open(args.feature_spec, 'r') as f:
                    feature_spec = json.load(f)
                # Apply feature transformations based on spec
                for feature, spec in feature_spec.items():
                    if feature in future_dates_df.columns:
                        if spec.get('type') == 'categorical':
                            future_dates_df[feature] = future_dates_df[feature].astype('category')
                print("LOG_START:FEATURE_SPEC_APPLIED\nFeature specification applied\nLOG_END:FEATURE_SPEC_APPLIED")
            except Exception as e:
                print(f"LOG_START:FEATURE_SPEC_WARNING\nFailed to apply feature spec: {str(e)}\nLOG_END:FEATURE_SPEC_WARNING")
        
        # Convert to H2OFrame
        h2o_future = h2o.H2OFrame(future_dates_df)
        
        # Make predictions
        predictions = model.predict(h2o_future)
        
        # Convert predictions to pandas
        predictions_df = predictions.as_data_frame()
        
        # Combine with future dates
        result_df = future_dates_df.copy()
        result_df['prediction'] = predictions_df['predict']
        
        # Save predictions
        output_path = args.output_file
        result_df.to_csv(output_path, index=False)
        
        # Print success message
        print(f"LOG_START:PREDICTION_SUMMARY\nRows predicted: {len(result_df)}\nOutput file: {output_path}\nFile size: {os.path.getsize(output_path)} bytes\nLOG_END:PREDICTION_SUMMARY")
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
