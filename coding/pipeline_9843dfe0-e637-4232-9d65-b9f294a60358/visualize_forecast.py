
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import argparse

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    spanish_months = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    try:
        parts = date_str.strip().split()
        if len(parts) != 3:
            return None
        day, month_abbr, year = parts
        month_abbr = month_abbr.lower().rstrip('.')
        month = spanish_months.get(month_abbr)
        if month:
            return datetime(int(year), month, int(day))
        return None
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate forecast visualization')
    parser.add_argument('--pipeline-dir', default='.', help='Pipeline working directory')
    parser.add_argument('--historical-file', required=True, help='Historical data CSV file')
    parser.add_argument('--predictions-file', required=True, help='Predictions CSV file')
    parser.add_argument('--output-file', default='forecast_plot.png', help='Output plot file')
    
    args = parser.parse_args()
    
    # Change to pipeline directory
    os.chdir(args.pipeline_dir)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Read historical data (no header)
        print(f"Reading historical data from: {args.historical_file}")
        historical_df = pd.read_csv(args.historical_file, header=None, names=['fecha', 'monto_total'])
        print(f"Historical data shape: {historical_df.shape}")
        
        # Read predictions data (with header)
        print(f"Reading predictions from: {args.predictions_file}")
        predictions_df = pd.read_csv(args.predictions_file)
        print(f"Predictions data shape: {predictions_df.shape}")
        
        # Parse dates
        print("Parsing dates...")
        historical_df['date_parsed'] = historical_df['fecha'].apply(parse_spanish_date)
        predictions_df['date_parsed'] = predictions_df['fecha'].apply(parse_spanish_date)
        
        # Remove rows with unparseable dates
        historical_df = historical_df.dropna(subset=['date_parsed'])
        predictions_df = predictions_df.dropna(subset=['date_parsed'])
        
        print(f"After date parsing - Historical: {len(historical_df)}, Predictions: {len(predictions_df)}")
        
        # Sort by date
        historical_df = historical_df.sort_values('date_parsed')
        predictions_df = predictions_df.sort_values('date_parsed')
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot historical data
        plt.plot(historical_df['date_parsed'], historical_df['monto_total'], 
                'o-', label='Historical', linewidth=2, markersize=4, alpha=0.7, color='blue')
        
        # Plot predictions
        plt.plot(predictions_df['date_parsed'], predictions_df['prediction'], 
                's--', label='Predictions', linewidth=2, markersize=4, alpha=0.8, color='red')
        
        # Formatting
        plt.title('Sales Forecast: Historical vs Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales Amount', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(args.output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {args.output_file}")
        
        # Success marker
        print(f"PLOT_FILE_START:{args.output_file}:PLOT_FILE_END")
        
    except Exception as e:
        print(f"ERROR_START:Failed to generate plot: {str(e)}:ERROR_END")
        raise

if __name__ == "__main__":
    main()
