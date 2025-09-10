import pandas as pd
import numpy as np
from datetime import datetime
import json

# Read and analyze the dataset
try:
    # Read the CSV with proper settings for Spanish format
    df = pd.read_csv('860505f2-013a-4bbb-badf-114e4645315c_ventas.csv', 
                    sep=';', 
                    header=None, 
                    names=['fecha', 'monto_total'],
                    decimal=',')
    
    # Convert dates from Spanish format
    def parse_spanish_date(date_str):
        month_map = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
            'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        for esp, eng in month_map.items():
            date_str = date_str.replace(esp, eng)
        return datetime.strptime(date_str, '%d %b. %Y')
    
    df['fecha'] = df['fecha'].apply(parse_spanish_date)
    
    # Basic analysis
    analysis = {
        'dataset_info': {
            'file_name': '860505f2-013a-4bbb-badf-114e4645315c_ventas.csv',
            'target_column': 'monto_total',
            'total_rows': len(df),
            'date_range': {
                'start': df['fecha'].min().strftime('%Y-%m-%d'),
                'end': df['fecha'].max().strftime('%Y-%m-%d'),
                'days': (df['fecha'].max() - df['fecha'].min()).days
            },
            'data_types': str(df.dtypes.to_dict())
        },
        'target_statistics': {
            'mean': float(df['monto_total'].mean()),
            'std': float(df['monto_total'].std()),
            'min': float(df['monto_total'].min()),
            'max': float(df['monto_total'].max()),
            'missing_values': int(df['monto_total'].isnull().sum())
        },
        'time_series_analysis': {
            'is_time_series': True,
            'frequency': 'daily',
            'has_missing_dates': len(pd.date_range(df['fecha'].min(), df['fecha'].max())) != len(df),
            'recommended_test_size': 0.2
        },
        'preprocessing_recommendations': [
            'Dates successfully parsed from Spanish format',
            'Numeric values converted from comma decimal format',
            'Create time-based features (day_of_week, month, year, etc.)',
            'Consider adding lag features for time series forecasting',
            'Split data chronologically for time series validation'
        ],
        'h2o_configuration': {
            'problem_type': 'regression',
            'max_runtime_secs': 300,
            'max_models': 20,
            'stopping_metric': 'RMSE',
            'seed': 42
        }
    }
    
    print(json.dumps(analysis, indent=2))
    
    # Save processed data for modeling
    df.to_csv('processed_sales_data.csv', index=False)
    print(f"Processed data saved to processed_sales_data.csv with {len(df)} rows")
    
except Exception as e:
    print(f"Error analyzing dataset: {str(e)}")
    raise