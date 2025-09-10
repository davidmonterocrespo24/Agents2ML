import h2o
import pandas as pd
from h2o.automl import H2OAutoML
import numpy as np
from datetime import datetime
import os
import sys

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    try:
        # Remove any extra spaces and split
        parts = date_str.strip().split()
        if len(parts) < 3:
            return None
            
        day = parts[0]
        month_abbr = parts[1].replace('.', '')[:3].lower()
        year = parts[2]
        
        if month_abbr in month_mapping:
            english_date_str = f"{day} {month_mapping[month_abbr]} {year}"
            return datetime.strptime(english_date_str, '%d %b %Y')
        else:
            return None
    except:
        return None

def main():
    # Initialize H2O
    h2o.init()
    
    try:
        # Read the CSV file with proper settings
        print("Reading CSV file...")
        df = pd.read_csv(
            '580b3ebe-3c5f-4b2c-b27c-06394f9dc50c_ventas.csv', 
            sep=';', 
            decimal=',',
            header=None,  # No header row
            names=['fecha', 'monto_total']  # Set proper column names
        )
        
        print(f"Dataset shape: {df.shape}")
        print(f"First few rows:\n{df.head()}")
        
        # Parse Spanish dates
        print("Parsing Spanish dates...")
        df['fecha_parsed'] = df['fecha'].apply(parse_spanish_date)
        
        # Check for any failed date parses
        failed_dates = df[df['fecha_parsed'].isnull()]
        if len(failed_dates) > 0:
            print(f"Warning: {len(failed_dates)} dates could not be parsed")
            print(failed_dates['fecha'].head())
            # Remove rows with invalid dates
            df = df.dropna(subset=['fecha_parsed'])
        
        # Extract time features
        df['year'] = df['fecha_parsed'].dt.year
        df['month'] = df['fecha_parsed'].dt.month
        df['day'] = df['fecha_parsed'].dt.day
        df['day_of_week'] = df['fecha_parsed'].dt.dayofweek
        df['day_of_year'] = df['fecha_parsed'].dt.dayofyear
        df['week_of_year'] = df['fecha_parsed'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Sort by date
        df = df.sort_values('fecha_parsed').reset_index(drop=True)
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Date range: {df['fecha_parsed'].min()} to {df['fecha_parsed'].max()}")
        
        # Convert to H2O frame
        h2o_df = h2o.H2OFrame(df.drop(columns=['fecha', 'fecha_parsed']))
        
        # Define features and target
        target = "monto_total"
        features = [col for col in h2o_df.columns if col != target]
        
        print(f"Target: {target}")
        print(f"Features: {features}")
        
        # Split data (time-series aware split)
        train_size = int(0.8 * len(h2o_df))
        train = h2o_df[:train_size]
        test = h2o_df[train_size:]
        
        print(f"Train size: {len(train)}")
        print(f"Test size: {len(test)}")
        
        # Run AutoML
        print("Starting H2O AutoML...")
        aml = H2OAutoML(
            max_models=20,
            seed=42,
            max_runtime_secs=300,
            sort_metric="RMSE",
            nfolds=5
        )
        
        aml.train(x=features, y=target, training_frame=train)
        
        # Get leaderboard
        lb = aml.leaderboard
        print("AutoML Leaderboard:")
        print(lb.head())
        
        # Get best model
        best_model = aml.leader
        print(f"Best model: {best_model.model_id}")
        
        # Make predictions
        predictions = best_model.predict(test)
        print("Predictions sample:")
        print(predictions.head())
        
        # Evaluate model
        performance = best_model.model_performance(test)
        print("Model performance on test set:")
        print(f"RMSE: {performance.rmse()}")
        print(f"MSE: {performance.mse()}")
        print(f"MAE: {performance.mae()}")
        print(f"R²: {performance.r2()}")
        
        # Save the model
        model_path = h2o.save_model(best_model, path="./", force=True)
        print(f"Model saved to: {model_path}")
        
        # Save predictions for analysis
        test_with_preds = test.cbind(predictions)
        test_with_preds_df = test_with_preds.as_data_frame()
        test_with_preds_df.to_csv('test_predictions.csv', index=False)
        print("Test predictions saved to test_predictions.csv")
        
        return {
            "status": "success",
            "model_path": model_path,
            "rmse": performance.rmse(),
            "r2": performance.r2(),
            "best_model": best_model.model_id
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    
    finally:
        # Shutdown H2O
        h2o.cluster().shutdown()

if __name__ == "__main__":
    result = main()
    print(f"\nExecution result: {result}")