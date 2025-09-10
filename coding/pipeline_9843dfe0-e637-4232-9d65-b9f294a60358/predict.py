import pandas as pd
import h2o
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import json

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    spanish_months = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    try:
        day, month_abbr, year = date_str.split()
        month_abbr = month_abbr.lower().rstrip('.')
        month = spanish_months.get(month_abbr)
        if month:
            return datetime(int(year), month, int(day))
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate predictions using trained H2O model')
    parser.add_argument('--model-path', required=True, help='Path to saved H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Path to pipeline working directory')
    parser.add_argument('--output-file', default='predictions.csv', help='Output CSV filename')
    parser.add_argument('--h2o-mem', default='4G', help='H2O memory size')
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    os.chdir(args.pipeline_dir)
    print(f"Working directory changed to: {os.getcwd()}")
    
    # Initialize H2O
    try:
        h2o.init(max_mem_size=args.h2o_mem)
        print("H2O initialized successfully")
    except Exception as e:
        print(f"ERROR_START:Failed to initialize H2O: {str(e)}:ERROR_END")
        return 1
    
    try:
        # Load the model
        print(f"Loading model from: {args.model_path}")
        if not os.path.exists(args.model_path):
            print(f"ERROR_START:Model path does not exist: {args.model_path}:ERROR_END")
            return 1
            
        model = h2o.load_model(args.model_path)
        print(f"Model loaded successfully: {model.model_id}")
        
        # Load and preprocess the original data
        print("Loading and preprocessing original dataset...")
        df = pd.read_csv('9843dfe0-e637-4232-9d65-b9f294a60358_ventas.csv', 
                         sep=';', 
                         header=None,
                         decimal=',',
                         encoding='utf-8')
        
        df.columns = ['fecha', 'monto_total']
        print(f"Original dataset shape: {df.shape}")
        
        # Parse dates
        df['fecha_parsed'] = df['fecha'].apply(parse_spanish_date)
        
        # Check for failed date parsing
        failed_dates = df[df['fecha_parsed'].isnull()]
        if len(failed_dates) > 0:
            print(f"LOG_START:Warning: {len(failed_dates)} dates could not be parsed:LOG_END")
            print(failed_dates.head())
        
        # Drop rows with invalid dates
        df_clean = df.dropna(subset=['fecha_parsed']).copy()
        df_clean = df_clean.sort_values('fecha_parsed')
        
        # Create time-based features (same as training)
        df_clean['year'] = df_clean['fecha_parsed'].dt.year
        df_clean['month'] = df_clean['fecha_parsed'].dt.month
        df_clean['day'] = df_clean['fecha_parsed'].dt.day
        df_clean['day_of_week'] = df_clean['fecha_parsed'].dt.dayofweek
        df_clean['day_of_year'] = df_clean['fecha_parsed'].dt.dayofyear
        
        print(f"Clean dataset for prediction: {df_clean.shape}")
        
        # Convert to H2O frame
        predictors = ['year', 'month', 'day', 'day_of_week', 'day_of_year']
        h2o_df = h2o.H2OFrame(df_clean[predictors])
        
        # Make predictions
        print("Generating predictions...")
        predictions = model.predict(h2o_df)
        predictions_df = predictions.as_data_frame()
        predictions_df.columns = ['prediction']
        
        # Combine with original data
        result_df = df_clean[['fecha', 'monto_total', 'fecha_parsed']].copy()
        result_df['prediction'] = predictions_df['prediction'].values
        
        # Save predictions
        output_path = args.output_file
        result_df[['fecha', 'monto_total', 'prediction']].to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        # Print summary
        print("LOG_START:Prediction Summary:")
        print(f"  Rows predicted: {len(result_df)}")
        print(f"  Model: {model.model_id}")
        print(f"  Predictors used: {predictors}")
        print(f"  Date range: {result_df['fecha_parsed'].min()} to {result_df['fecha_parsed'].max()}")
        print(f"  Output file: {output_path}")
        print("LOG_END")
        
        # Print the required structured output
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        return 0
        
    except Exception as e:
        print(f"ERROR_START:Prediction failed: {str(e)}:ERROR_END")
        return 1
    finally:
        # Shutdown H2O
        h2o.cluster().shutdown()
        print("H2O shutdown completed")

if __name__ == "__main__":
    exit(main())