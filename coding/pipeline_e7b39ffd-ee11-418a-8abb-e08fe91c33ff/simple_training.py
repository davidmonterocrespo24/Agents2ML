import pandas as pd
import numpy as np
import json
import os
import h2o
from h2o.automl import H2OAutoML
import argparse
import sys

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', required=True)
        parser.add_argument('--target', required=True)
        parser.add_argument('--output-dir', default='.')
        args = parser.parse_args()
        
        print(f"Reading file: {args.file}")
        print(f"Target column: {args.target}")
        
        # Read and preprocess data
        df = pd.read_csv(args.file, sep=';', header=None, names=['date', 'total_amount'])
        
        print(f"Dataset shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        
        # Clean and transform data
        df['date'] = pd.to_datetime(df['date'].str.strip(), format='%d %b. %Y', errors='coerce')
        df['total_amount'] = df['total_amount'].astype(str).str.replace(',', '.').astype(float)
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Sort by date for time series
        df = df.sort_values('date')
        
        # Create time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        print(f"Data after preprocessing:\n{df.head()}")
        
        # Initialize H2O
        h2o.init()
        
        # Convert to H2O frame
        hf = h2o.H2OFrame(df.drop('date', axis=1))
        
        # Split data
        splits = hf.split_frame(ratios=[0.8, 0.1], seed=42)
        train, valid, test = splits[0], splits[1], splits[2]
        
        # Define features and target
        x = [col for col in hf.columns if col != 'total_amount']
        y = 'total_amount'
        
        print(f"Features: {x}")
        print(f"Target: {y}")
        
        # Train AutoML
        aml = H2OAutoML(max_models=5, max_runtime_secs=120, seed=42)
        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        
        # Get leader model
        leader = aml.leader
        
        # Save model
        model_path = h2o.save_model(model=leader, path=args.output_dir, force=True)
        
        # Get metrics
        perf = leader.model_performance(test)
        metrics = {
            'rmse': perf.rmse(),
            'mae': perf.mae(),
            'r2': perf.r2()
        }
        
        # Output structured results
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        
        result_summary = {
            'model_path': model_path,
            'metrics': metrics,
            'rows_trained': len(df),
            'training_successful': True
        }
        
        print("METRICS_START:" + json.dumps(result_summary) + ":METRICS_END")
        
        h2o.cluster().shutdown()
        return 0
        
    except Exception as e:
        print(f"ERROR_START:{str(e)}:ERROR_END")
        try:
            h2o.cluster().shutdown()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())