# PREDICTION SCRIPT
# Generated on: 2025-09-10T00:57:18.444799
# Pipeline: pipeline_e7b39ffd-ee11-418a-8abb-e08fe91c33ff
# Filename: prediction_script.py
# Arguments: --model-path=StackedEnsemble_AllModels_1_AutoML_1_20250910_35442 --pipeline-dir=. --horizon=30 --output-file=predictions.csv
# Script Type: prediction

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
    parser.add_argument('--model-path', required=True, help='Path to the saved H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--future-dates-file', required=False, help='CSV/JSON file with future dates')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--freq', default='D', help='Frequency for date generation')
    parser.add_argument('--date-column', default='date', help='Name of date column')
    parser.add_argument('--feature-spec', required=False, help='JSON file with feature specifications')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--max-rows-sample', type=int, default=10000, help='Maximum rows for sampling')
    parser.add_argument('--h2o-mem', default='2G', help='H2O memory allocation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    try:
        # Change to pipeline directory
        os.chdir(args.pipeline_dir)
        print(f"LOG_START:Working in pipeline directory: {os.getcwd()}:LOG_END")
        
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=-1)
        print("LOG_START:H2O initialized successfully:LOG_END")
        
        # Load the model
        try:
            model = h2o.load_model(args.model_path)
            print(f"LOG_START:Model loaded: {args.model_path}:LOG_END")
        except Exception as e:
            print(f"ERROR_START:Failed to load model {args.model_path}: {str(e)}:ERROR_END")
            return 1
        
        # Get model information
        model_type = type(model).__name__
        print(f"LOG_START:Model type: {model_type}:LOG_END")
        
        # Prepare future data
        if args.future_dates_file and os.path.exists(args.future_dates_file):
            # Load future dates from file
            if args.future_dates_file.endswith('.csv'):
                future_df = pd.read_csv(args.future_dates_file)
            elif args.future_dates_file.endswith('.json'):
                future_df = pd.read_json(args.future_dates_file)
            else:
                print(f"ERROR_START:Unsupported file format for future dates: {args.future_dates_file}:ERROR_END")
                return 1
            print(f"LOG_START:Loaded {len(future_df)} future dates from file:LOG_END")
        else:
            # Generate future dates
            last_date = datetime(2024, 2, 2)  # From training data
            future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                       periods=args.horizon, freq=args.freq)
            future_df = pd.DataFrame({'date': future_dates})
            print(f"LOG_START:Generated {len(future_df)} future dates:LOG_END")
        
        # Apply feature engineering (same as training)
        future_df['date'] = pd.to_datetime(future_df['date'])
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        
        print(f"LOG_START:Created features for {len(future_df)} future periods:LOG_END")
        
        # Convert to H2OFrame
        h2o_future = h2o.H2OFrame(future_df)
        
        # Make predictions
        predictions = model.predict(h2o_future)
        
        # Combine predictions with future dates
        result_df = future_df.copy()
        result_df['prediction'] = predictions.as_data_frame().values
        
        # Save predictions
        output_path = os.path.join(args.pipeline_dir, args.output_file)
        result_df.to_csv(output_path, index=False)
        
        # Verify file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path)
            print(f"LOG_START:Predictions saved successfully: {output_path} ({file_size} bytes):LOG_END")
            print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        else:
            print("ERROR_START:Failed to save predictions file:ERROR_END")
            return 1
            
        return 0
        
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"ERROR_START:{error_msg}:ERROR_END")
        return 1
    finally:
        try:
            h2o.cluster().shutdown()
        except:
            pass

if __name__ == '__main__':
    sys.exit(main())
