#!/usr/bin/env python3
# visualization_script.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import os

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 10

def load_and_prepare_data(historical_file, predictions_file):
    """Load and prepare historical and prediction data"""
    try:
        # Load historical data
        historical_df = pd.read_csv(historical_file, sep=';', header=None, 
                                  names=['fecha', 'monto_total'], encoding='utf-8')
        
        # Parse dates and convert numeric values
        historical_df['fecha'] = pd.to_datetime(historical_df['fecha'], 
                                              format='%d %b. %Y', errors='coerce')
        historical_df['monto_total'] = historical_df['monto_total'].str.replace(',', '.').astype(float)
        
        # Load predictions
        predictions_df = pd.read_csv(predictions_file)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        
        return historical_df, predictions_df
        
    except Exception as e:
        print(f"ERROR_START:Failed to load data: {str(e)}:ERROR_END")
        return None, None

def create_forecast_plot(historical_df, predictions_df):
    """Create the forecast plot with historical data and predictions"""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        ax.plot(historical_df['fecha'], historical_df['monto_total'], 
                'b-', linewidth=2, label='Historical Sales', alpha=0.8)
        
        # Plot predictions
        ax.plot(predictions_df['date'], predictions_df['predicted_sales'], 
                'r--', linewidth=2, label='Forecast', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Amount', fontsize=12, fontweight='bold')
        ax.set_title('Sales Forecast: Historical Data vs Predictions', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
        
        # Add statistics to plot
        last_historical = historical_df['monto_total'].iloc[-1]
        first_prediction = predictions_df['predicted_sales'].iloc[0]
        change_pct = ((first_prediction - last_historical) / last_historical) * 100
        
        stats_text = f'Last historical: ${last_historical:,.0f}\nFirst forecast: ${first_prediction:,.0f}\nChange: {change_pct:+.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"ERROR_START:Failed to create plot: {str(e)}:ERROR_END")
        return None

def main():
    """Main function to generate the forecast plot"""
    try:
        # File paths
        historical_file = './ed12db74-4167-4756-a811-8fddfa33753b_ventas.csv'
        predictions_file = './predictions.csv'
        output_file = './forecast_plot.png'
        
        # Check if files exist
        if not os.path.exists(historical_file):
            print(f"ERROR_START:Historical file not found: {historical_file}:ERROR_END")
            return 1
            
        if not os.path.exists(predictions_file):
            print(f"ERROR_START:Predictions file not found: {predictions_file}:ERROR_END")
            return 1
        
        # Load data
        historical_df, predictions_df = load_and_prepare_data(historical_file, predictions_file)
        if historical_df is None or predictions_df is None:
            return 1
        
        # Create plot
        fig = create_forecast_plot(historical_df, predictions_df)
        if fig is None:
            return 1
        
        # Save plot
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # Verify file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"LOG_START:Plot saved successfully: {output_file} ({file_size} bytes):LOG_END")
            print(f"VISUALIZATION_SUCCESS_START:{output_file}:VISUALIZATION_SUCCESS_END")
            return 0
        else:
            print("ERROR_START:Failed to save plot file:ERROR_END")
            return 1
            
    except Exception as e:
        print(f"ERROR_START:Unexpected error: {str(e)}:ERROR_END")
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)