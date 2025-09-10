#!/usr/bin/env python3
# data_analysis.py

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def detect_separator(file_path):
    """Detect the CSV separator"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    separators = [',', ';', '\t', '|']
    best_sep = ','
    best_count = 0
    for sep in separators:
        count = first_line.count(sep)
        if count > best_count:
            best_count = count
            best_sep = sep
    return best_sep

def analyze_dataset(file_path, target_column):
    """Analyze the dataset and return metadata"""
    
    # Detect separator
    sep = detect_separator(file_path)
    
    # Read the data
    df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
    
    # Basic information
    n_rows, n_cols = df.shape
    column_names = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = {col: (missing_values[col] / n_rows * 100) for col in df.columns}
    
    # Target column analysis
    target_info = {}
    if target_column in df.columns:
        target_series = df[target_column]
        target_info = {
            'type': str(target_series.dtype),
            'missing_count': int(missing_values[target_column]),
            'missing_percentage': float(missing_percentage[target_column]),
            'unique_count': int(target_series.nunique()),
            'min': float(target_series.min()) if pd.api.types.is_numeric_dtype(target_series) else None,
            'max': float(target_series.max()) if pd.api.types.is_numeric_dtype(target_series) else None,
            'mean': float(target_series.mean()) if pd.api.types.is_numeric_dtype(target_series) else None,
            'std': float(target_series.std()) if pd.api.types.is_numeric_dtype(target_series) else None
        }
    
    # Date column detection
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to parse as date
            try:
                sample = pd.to_datetime(df[col].head(10), errors='coerce', dayfirst=True)
                if sample.notna().sum() >= 5:  # At least 5 valid dates
                    date_columns.append(col)
            except:
                pass
    
    # Data sample
    sample_data = df.head(5).to_dict('records')
    
    # Recommendations
    recommendations = {
        'problem_type': 'regression' if pd.api.types.is_numeric_dtype(df[target_column]) else 'classification',
        'preprocessing_needed': True,
        'date_features': len(date_columns) > 0,
        'separator': sep,
        'suggested_validation_split': 0.2,
        'suggested_test_split': 0.1
    }
    
    # Compile final report
    report = {
        'dataset_info': {
            'file_name': os.path.basename(file_path),
            'total_rows': int(n_rows),
            'total_columns': int(n_cols),
            'column_names': column_names,
            'data_types': dtypes,
            'separator': sep
        },
        'missing_values': {
            'counts': missing_values,
            'percentages': missing_percentage
        },
        'target_column': {
            'name': target_column,
            'analysis': target_info
        },
        'date_columns': date_columns,
        'data_sample': sample_data,
        'recommendations': recommendations
    }
    
    return report

if __name__ == "__main__":
    dataset_file = "eeaff94b-29d3-4e76-b80d-a2e7513699a8_ventas.csv"
    target_column = "monto total"
    
    try:
        report = analyze_dataset(dataset_file, target_column)
        print("DATA_ANALYSIS_START:" + json.dumps(report, indent=2, ensure_ascii=False) + ":DATA_ANALYSIS_END")
    except Exception as e:
        print(f"ERROR_START:Data analysis failed: {str(e)}:ERROR_END")