#!/usr/bin/env python3
# prediction_script.py

import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
import json
import sys
from datetime import datetime, timedelta

def generate_future_dates(last_date, horizon=30):
    """Generate future dates for prediction"""
    future_dates = []
    current_date = pd.to_datetime(last_date)
    
    for i in range(1, horizon + 1):
        future_date = current_date + timedelta(days=i)
        future_dates.append(future_date)
    
    return future_dates

def create_time_features_from_dates(dates):
    """Create time features from date list"""
    df = pd.DataFrame({'fecha': dates})
    
    df['year'] = df['fecha'].dt.year
    df['month'] = df['fecha'].dt.month
    df['day'] = df['fecha'].dt.day
    df['dayofweek'] = df['fecha'].dt.dayofweek
    df['is_weekend'] = df['fecha'].dt.dayofweek.isin([5,6]).astype(int)
    df['dayofyear'] = df['fecha'].dt.dayofyear
    df['quarter'] = df['fecha'].dt.quarter
    
    return df.drop('fecha', axis=1)

def main():
    try:
        # Load the trained model
        model_path = "./GBM_3_AutoML_1_20250904_181426"
        
        # Initialize H2O
        h2o.init()
        
        # Load the model
        model = h2o.load_model(model_path)
        
        # Get the last date from training data to start predictions
        # For simplicity, we'll use the current date as reference
        last_training_date = datetime(2024, 9, 4)  # Approximate last date
        
        # Generate future dates (30 days horizon)
        future_dates = generate_future_dates(last_training_date, horizon=30)
        
        # Create features for prediction
        prediction_features = create_time_features_from_dates(future_dates)
        
        # Convert to H2O frame
        h2o_pred_frame = h2o.H2OFrame(prediction_features)
        
        # Make predictions
        predictions = model.predict(h2o_pred_frame)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': predictions.as_data_frame().iloc[:, 0].values
        })
        
        # Save predictions to CSV
        output_path = "./predictions.csv"
        results_df.to_csv(output_path, index=False)
        
        # Print success message
        print(f"PREDICTIONS_SUCCESS_START:Predictions generated successfully:PREDICTIONS_SUCCESS_END")
        print(f"PREDICTIONS_FILE_START:{output_path}:PREDICTIONS_FILE_END")
        
        # Print sample predictions
        sample_predictions = results_df.head().to_dict('records')
        print("PREDICTIONS_SAMPLE_START:" + json.dumps(sample_predictions) + ":PREDICTIONS_SAMPLE_END")
        
        h2o.cluster().shutdown()
        return 0
        
    except Exception as e:
        print(f"PREDICTIONS_ERROR_START:{str(e)}:PREDICTIONS_ERROR_END")
        return 1

if __name__ == "__main__":
    sys.exit(main())