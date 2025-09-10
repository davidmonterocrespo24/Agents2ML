#!/usr/bin/env python3
# forecast_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Create forecast visualization plot')
    parser.add_argument('--historical-file', required=True, help='Path to historical data CSV')
    parser.add_argument('--predictions-file', required=True, help='Path to predictions CSV')
    parser.add_argument('--output-file', default='forecast_plot.png', help='Output plot file')
    
    args = parser.parse_args()
    
    try:
        # Load historical data
        historical_df = pd.read_csv(args.historical_file, sep=';', header=None, names=['fecha', 'monto_total'])
        
        # Convert date column to datetime (Spanish format)
        historical_df['fecha'] = pd.to_datetime(historical_df['fecha'], format='%d %b. %Y', errors='coerce')
        
        # Convert amount column to numeric (handle comma decimal separator)
        historical_df['monto_total'] = historical_df['monto_total'].str.replace(',', '.').astype(float)
        
        # Load predictions
        predictions_df = pd.read_csv(args.predictions_file)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(historical_df['fecha'], historical_df['monto_total'], 
                label='Historical Sales', color='blue', linewidth=2, marker='o', markersize=3)
        
        # Plot predictions
        plt.plot(predictions_df['date'], predictions_df['prediction'], 
                label='Forecast', color='red', linewidth=2, linestyle='--', marker='s', markersize=3)
        
        # Formatting
        plt.title('Sales Forecast: Historical vs Predicted', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Amount', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"VISUALIZATION_SUCCESS: Plot saved to {args.output_file}")
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())