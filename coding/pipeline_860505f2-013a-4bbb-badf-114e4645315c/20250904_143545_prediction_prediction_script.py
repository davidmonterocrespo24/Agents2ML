# PREDICTION SCRIPT
# Generated on: 2025-09-04T14:35:45.938539
# Pipeline: pipeline_860505f2-013a-4bbb-badf-114e4645315c
# Filename: prediction_script.py
# Arguments: 
# Script Type: prediction

#!/usr/bin/env python3
# h2o_prediction_script.py

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Change to pipeline directory first
os.chdir('./')

try:
    import h2o
except ImportError:
    print("ERROR_START:H2O package not installed. Please install with: pip install h2o:ERROR_END")
    sys.exit(1)

def parse_spanish_date(date_str):
    """Parse Spanish date format with month abbreviations - same as training"""
    spanish_months = {
        'ene.': 'Jan', 'feb.': 'Feb', 'mar.': 'Mar', 'abr.': 'Apr',
        'may.': 'May', 'jun.': 'Jun', 'jul.': 'Jul', 'ago.': 'Aug',
        'sept.': 'Sep', 'oct.': 'Oct', 'nov.': 'Nov', 'dic.': 'Dec'
    }
    
    if pd.isna(date_str) or date_str is None or date_str == '':
        return pd.NaT
        
    date_str = str(date_str).strip()
    if date_str == '':
        return pd.NaT
    
    # Replace Spanish month abbreviations with English ones
    for esp, eng in spanish_months.items():
        if esp in date_str:
            date_str = date_str.replace(esp, eng)
            break
            
    try:
        return pd.to_datetime(date_str, format='%d %b %Y', errors='coerce')
    except:
        return pd.NaT

def create_time_features(df: pd.DataFrame, col):
    """Create time features exactly like training script"""
    s = df[col].apply(parse_spanish_date)
    df[f"{col}__year"] = s.dt.year
    df[f"{col}__month"] = s.dt.month
    df[f"{col}__day"] = s.dt.day
    df[f"{col}__dayofweek"] = s.dt.dayofweek
    df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{col}__is_month_start"] = s.dt.is_month_start.astype(int)
    df[f"{col}__is_month_end"] = s.dt.is_month_end.astype(int)
    df[f"{col}__quarter"] = s.dt.quarter
    df[f"{col}__dayofyear"] = s.dt.dayofyear
    df[f"{col}__weekofyear"] = s.dt.isocalendar().week
    return df

def generate_future_dates(last_date_str, horizon=30, freq='D'):
    """Generate future dates starting from the last date in training data"""
    last_date = parse_spanish_date(last_date_str)
    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                               periods=horizon, freq=freq)
    
    # Convert back to Spanish format for consistency
    spanish_months_reverse = {
        'Jan': 'ene.', 'Feb': 'feb.', 'Mar': 'mar.', 'Apr': 'abr.',
        'May': 'may.', 'Jun': 'jun.', 'Jul': 'jul.', 'Aug': 'ago.',
        'Sep': 'sept.', 'Oct': 'oct.', 'Nov': 'nov.', 'Dec': 'dic.'
    }
    
    spanish_dates = []
    for date in future_dates:
        month_eng = date.strftime('%b')
        month_esp = spanish_months_reverse.get(month_eng, month_eng)
        spanish_date = f"{date.day} {month_esp} {date.year}"
        spanish_dates.append(spanish_date)
    
    return spanish_dates

def main():
    # Hardcode parameters for pipeline context
    model_path = '/workspace/pipeline_860505f2-013a-4bbb-badf-114e4645315c/GBM_4_AutoML_1_20250904_172755'
    pipeline_dir = './'
    horizon = 30
    output_file = 'predictions.csv'
    
    try:
        # Change to pipeline directory
        os.chdir(pipeline_dir)
        print(f"LOG_START:Working in directory: {os.getcwd()}:LOG_END")
        
        # Initialize H2O
        h2o.init(max_mem_size='2G', nthreads=-1)
        print("LOG_START:H2O initialized successfully:LOG_END")
        
        # Load the model
        try:
            model = h2o.load_model(model_path)
            print(f"LOG_START:Model loaded: {model_path}:LOG_END")
        except Exception as e:
            print(f"ERROR_START:Failed to load model {model_path}: {str(e)}:ERROR_END")
            return 1
        
        # Get the last date from training data to generate future dates
        training_data = pd.read_csv('860505f2-013a-4bbb-badf-114e4645315c_ventas.csv', 
                                  sep=';', header=None, names=['fecha', 'monto_total'])
        last_date = training_data['fecha'].iloc[-1]
        print(f"LOG_START:Last training date: {last_date}:LOG_END")
        
        # Generate future dates
        future_dates = generate_future_dates(last_date, horizon, 'D')
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({'fecha': future_dates})
        
        # Create time features (same as training)
        pred_df = create_time_features(pred_df, 'fecha')
        
        # Drop original date column (model expects only features)
        pred_df = pred_df.drop(columns=['fecha'])
        
        # Convert to H2OFrame
        h2o_pred_frame = h2o.H2OFrame(pred_df)
        
        # Make predictions
        predictions = model.predict(h2o_pred_frame)
        
        # Combine results
        result_df = pd.DataFrame({
            'fecha': future_dates,
            'predicted_monto_total': predictions.as_data_frame().iloc[:, 0].values
        })
        
        # Save predictions
        output_path = os.path.join(pipeline_dir, output_file)
        result_df.to_csv(output_path, index=False)
        
        print(f"LOG_START:Predictions saved: {output_path}:LOG_END")
        print(f"LOG_START:Predictions shape: {result_df.shape}:LOG_END")
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        # Show sample predictions
        print("LOG_START:SAMPLE_PREDICTIONS")
        print(result_df.head(10).to_string())
        print("LOG_END:SAMPLE_PREDICTIONS")
        
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
