# VISUALIZATION SCRIPT
# Generated on: 2025-09-04T20:48:39.898808
# Pipeline: pipeline_d304ff5d-f2ed-4c30-9955-fa8acbb291b5
# Filename: visualization_script.py
# Arguments: 
# Script Type: visualization

#!/usr/bin/env python3
# visualization_script.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import re

def parse_spanish_date(date_str):
    """Parse Spanish date format with month abbreviations"""
    month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    if pd.isna(date_str):
        return pd.NaT
    
    # Clean and normalize the date string
    date_str = str(date_str).strip().lower()
    date_str = re.sub(r'\.', '', date_str)  # Remove dots from month abbreviations
    
    # Replace Spanish month abbreviations with English ones
    for esp, eng in month_mapping.items():
        if esp in date_str:
            date_str = date_str.replace(esp, eng.lower())
            break
    
    try:
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except Exception:
        return pd.NaT

def main():
    # Read historical data
    try:
        historical_df = pd.read_csv('d304ff5d-f2ed-4c30-9955-fa8acbb291b5_ventas.csv', 
                                  sep=';', header=None, names=['fecha', 'monto_total'])
        # Handle European decimal format
        historical_df['monto_total'] = historical_df['monto_total'].astype(str).str.replace(',', '.').astype(float)
        # Parse Spanish dates
        historical_df['fecha'] = historical_df['fecha'].apply(parse_spanish_date)
    except Exception as e:
        print(f"ERROR_START:Failed to read historical data: {str(e)}:ERROR_END")
        return 1
    
    # Read predictions
    try:
        predictions_df = pd.read_csv('predictions.csv')
        predictions_df['fecha'] = pd.to_datetime(predictions_df['fecha'])
    except Exception as e:
        print(f"ERROR_START:Failed to read predictions: {str(e)}:ERROR_END")
        return 1
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot historical data
    plt.plot(historical_df['fecha'], historical_df['monto_total'], 
             'b-', label='Historical Data', linewidth=2, alpha=0.8)
    
    # Plot predictions
    plt.plot(predictions_df['fecha'], predictions_df['prediction'], 
             'r--', label='Forecast', linewidth=3, alpha=0.9)
    
    # Formatting
    plt.title('Sales Forecast: Historical Data vs Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Amount (monto_total)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    # Add vertical line at the start of predictions
    prediction_start_date = predictions_df['fecha'].min()
    plt.axvline(x=prediction_start_date, color='green', linestyle=':', 
                alpha=0.7, label='Forecast Start')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
        print("LOG_START:Visualization saved as forecast_plot.png:LOG_END")
        
        # Check file size
        import os
        file_size = os.path.getsize('forecast_plot.png')
        print(f"LOG_START:Plot file size: {file_size} bytes:LOG_END")
        
        # Show some stats
        print(f"LOG_START:Historical data points: {len(historical_df)}:LOG_END")
        print(f"LOG_START:Prediction points: {len(predictions_df)}:LOG_END")
        print(f"LOG_START:Historical date range: {historical_df['fecha'].min()} to {historical_df['fecha'].max()}:LOG_END")
        print(f"LOG_START:Prediction date range: {predictions_df['fecha'].min()} to {predictions_df['fecha'].max()}:LOG_END")
        
        print("VISUALIZATION_SUCCESS_START:forecast_plot.png:VISUALIZATION_SUCCESS_END")
        
    except Exception as e:
        print(f"ERROR_START:Failed to save visualization: {str(e)}:ERROR_END")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
