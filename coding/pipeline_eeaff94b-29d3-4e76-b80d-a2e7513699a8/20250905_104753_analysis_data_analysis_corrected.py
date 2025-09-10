# ANALYSIS SCRIPT
# Generated on: 2025-09-05T10:47:53.341397
# Pipeline: pipeline_eeaff94b-29d3-4e76-b80d-a2e7513699a8
# Filename: data_analysis_corrected.py
# Arguments: 
# Script Type: analysis

#!/usr/bin/env python3
# data_analysis_corrected.py

import pandas as pd
import numpy as np
import json
import sys
import re
from datetime import datetime

def main():
    try:
        # Read the CSV file with proper handling for Spanish format
        df = pd.read_csv('eeaff94b-29d3-4e76-b80d-a2e7513699a8_ventas.csv', 
                        sep=';', 
                        header=None,
                        names=['fecha', 'monto_total'],
                        encoding='utf-8')
        
        # Convert Spanish month names to English for proper parsing
        month_mapping = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
            'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        
        def convert_spanish_date(date_str):
            for esp, eng in month_mapping.items():
                date_str = re.sub(rf'\b{esp}\b', eng, date_str, flags=re.IGNORECASE)
            return date_str
        
        # Apply date conversion
        df['fecha'] = df['fecha'].apply(convert_spanish_date)
        
        # Parse dates with dayfirst=True for European format
        df['fecha'] = pd.to_datetime(df['fecha'], format='%d %b. %Y', errors='coerce')
        
        # Convert numeric column - replace comma with dot for decimal and handle thousands
        df['monto_total'] = df['monto_total'].astype(str).str.replace('.', '', regex=False)
        df['monto_total'] = df['monto_total'].str.replace(',', '.', regex=False).astype(float)
        
        # Basic analysis
        analysis = {
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "date_range": {
                "start": df['fecha'].min().strftime('%Y-%m-%d') if not df['fecha'].isna().all() else None,
                "end": df['fecha'].max().strftime('%Y-%m-%d') if not df['fecha'].isna().all() else None
            },
            "target_stats": {
                "mean": float(df['monto_total'].mean()),
                "std": float(df['monto_total'].std()),
                "min": float(df['monto_total'].min()),
                "max": float(df['monto_total'].max()),
                "null_count": int(df['monto_total'].isna().sum())
            },
            "sample_data": df.head(10).to_dict(orient='records')
        }
        
        print(json.dumps(analysis, indent=2, default=str))
        return 0
        
    except Exception as e:
        print(f"ERROR_START:Data analysis failed: {str(e)}:ERROR_END")
        return 1

if __name__ == "__main__":
    sys.exit(main())
