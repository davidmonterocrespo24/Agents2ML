#!/usr/bin/env python3
# forecast_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from datetime import datetime

def parse_spanish_date(date_str):
    """Parse Spanish date format with robust error handling"""
    spanish_month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
        'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'sept': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    # Handle both abbreviated and full month names with/without periods
    import re
    date_str = re.sub(r'(\w+)\.', r'\1', date_str)  # Remove periods from month abbreviations
    date_str = date_str.lower()  # Convert to lowercase for consistent matching
    
    # Replace Spanish month abbreviations with English ones
    for esp, eng in spanish_month_mapping.items():
        if esp in date_str:
            date_str = date_str.replace(esp, eng)
            break
    
    return pd.to_datetime(date_str, format='%d %b %Y', errors='coerce')

def convert_european_number(num_str):
    """Convert European decimal format (comma as decimal separator)"""
    if isinstance(num_str, str):
        # Remove any thousand separators (dots or spaces) and replace comma with dot
        num_str = num_str.replace('.', '').replace(' ', '').replace(',', '.')
    try:
        return float(num_str) if pd.notna(num_str) else None
    except (ValueError, TypeError):
        return None

def main():
    parser = argparse.ArgumentParser(description='Forecast Visualization Script')
    parser.add_argument('--historical-file', required=True, help='Path to historical data CSV')
    parser.add_argument('--predictions-file', required=True, help='Path to predictions CSV')
    parser.add_argument('--output-file', default='forecast_plot.png', help='Output plot file')
    parser.add_argument('--date-column', default='fecha', help='Name of date column')
    parser.add_argument('--value-column', default='monto total', help='Name of value column')
    parser.add_argument('--title', default='Sales Forecast', help='Plot title')
    parser.add_argument('--width', type=int, default=12, help='Plot width')
    parser.add_argument('--height', type=int, default=6, help='Plot height')
    
    args = parser.parse_args()
    
    try:
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Read historical data
        print(f"Reading historical data from: {args.historical_file}")
        historical_df = pd.read_csv(args.historical_file, sep=';')
        
        # Process historical data
        if len(historical_df.columns) == 2:
            historical_df.columns = [args.date_column, args.value_column]
        
        # Parse dates and convert values
        historical_df[args.date_column] = historical_df[args.date_column].apply(parse_spanish_date)
        historical_df[args.value_column] = historical_df[args.value_column].apply(convert_european_number)
        
        # Drop any rows with invalid dates or values
        historical_df = historical_df.dropna(subset=[args.date_column, args.value_column])
        
        # Read predictions
        print(f"Reading predictions from: {args.predictions_file}")
        predictions_df = pd.read_csv(args.predictions_file)
        
        # Ensure date column is datetime
        predictions_df[args.date_column] = pd.to_datetime(predictions_df[args.date_column])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        
        # Plot historical data
        ax.plot(historical_df[args.date_column], 
                historical_df[args.value_column], 
                'o-', linewidth=2, markersize=4, 
                label='Historical Sales', color='#2E86AB', alpha=0.8)
        
        # Plot predictions
        ax.plot(predictions_df[args.date_column], 
                predictions_df['prediction'], 
                's--', linewidth=2, markersize=5,
                label='Forecast', color='#A23B72', alpha=0.9)
        
        # Add vertical line to separate historical and forecast
        last_historical_date = historical_df[args.date_column].max()
        ax.axvline(x=last_historical_date, color='red', linestyle=':', alpha=0.7, 
                  label='Forecast Start')
        
        # Customize the plot
        ax.set_title(args.title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Amount', fontsize=12, fontweight='bold')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(args.output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {args.output_file}")
        
        # Show some statistics
        print(f"Historical data points: {len(historical_df)}")
        print(f"Forecast points: {len(predictions_df)}")
        print(f"Historical date range: {historical_df[args.date_column].min()} to {historical_df[args.date_column].max()}")
        print(f"Forecast date range: {predictions_df[args.date_column].min()} to {predictions_df[args.date_column].max()}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())