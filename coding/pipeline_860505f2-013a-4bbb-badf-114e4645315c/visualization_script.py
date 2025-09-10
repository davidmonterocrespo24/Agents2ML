#!/usr/bin/env python3
# visualization_script.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

def parse_spanish_date(date_str):
    """Parse Spanish date format with month abbreviations"""
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

def convert_comma_decimal_to_float(value):
    """Convert string with comma decimal to float with robust empty handling"""
    if pd.isna(value) or value is None or value == '':
        return np.nan
    try:
        if isinstance(value, str):
            cleaned = value.strip().replace(',', '.').replace(' ', '')
            if cleaned == '':
                return np.nan
            return float(cleaned)
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def main():
    try:
        # Load historical data
        historical_df = pd.read_csv('860505f2-013a-4bbb-badf-114e4645315c_ventas.csv', 
                                  sep=';', header=None, names=['fecha', 'monto_total'])
        
        # Convert historical data
        historical_df['monto_total'] = historical_df['monto_total'].apply(convert_comma_decimal_to_float)
        historical_df['fecha_dt'] = historical_df['fecha'].apply(parse_spanish_date)
        
        # Load predictions
        predictions_df = pd.read_csv('predictions.csv')
        predictions_df['fecha_dt'] = predictions_df['fecha'].apply(parse_spanish_date)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        plt.plot(historical_df['fecha_dt'], historical_df['monto_total'], 
                label='Datos Históricos', color='blue', linewidth=2, marker='o', markersize=3)
        
        # Plot predictions
        plt.plot(predictions_df['fecha_dt'], predictions_df['predicted_monto_total'], 
                label='Predicciones', color='red', linewidth=2, linestyle='--', marker='s', markersize=3)
        
        # Add vertical line to separate historical from predictions
        last_historical_date = historical_df['fecha_dt'].max()
        plt.axvline(x=last_historical_date, color='green', linestyle=':', alpha=0.7, 
                   label='Inicio Predicciones')
        
        # Formatting
        plt.title('Predicción de Ventas - Datos Históricos vs Predicciones', fontsize=16, fontweight='bold')
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Monto Total', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        # Add some padding
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("LOG_START:Plot created successfully: forecast_plot.png:LOG_END")
        print("VISUALIZATION_FILE_START:forecast_plot.png:VISUALIZATION_FILE_END")
        
        return 0
        
    except Exception as e:
        print(f"ERROR_START:Error creating visualization: {str(e)}:ERROR_END")
        return 1

if __name__ == '__main__':
    main()