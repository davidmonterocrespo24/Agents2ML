# VISUALIZATION SCRIPT
# Generated on: 2025-09-05T13:14:37.263142
# Pipeline: pipeline_fe7ccf65-e073-4c98-ba7d-6740e42a87cf
# Filename: visualization_final.py
# Arguments: 
# Script Type: visualization

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def parse_spanish_date(date_str):
    """Parse Spanish date format like '16 nov. 2023'"""
    month_mapping = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
        'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    
    try:
        # Handle both string and datetime objects
        if isinstance(date_str, str):
            date_str_clean = date_str.replace('.', '')
            parts = date_str_clean.split()
            if len(parts) >= 3:
                day = parts[0]
                month_abbr = parts[1][:3].lower()
                year = parts[2]
                
                if month_abbr in month_mapping:
                    english_date_str = f"{day} {month_mapping[month_abbr]} {year}"
                    return pd.to_datetime(english_date_str, dayfirst=True)
            
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        else:
            return pd.to_datetime(date_str)
    except:
        return pd.to_datetime(date_str, errors='coerce')

def main():
    print("Starting visualization process...")
    
    try:
        # Load historical data
        print("Loading historical data...")
        historical_df = pd.read_csv('fe7ccf65-e073-4c98-ba7d-6740e42a87cf_ventas.csv', 
                                  sep=';', header=None, decimal=',')
        historical_df.columns = ['fecha', 'monto total']
        historical_df['fecha'] = historical_df['fecha'].apply(parse_spanish_date)
        historical_df = historical_df.sort_values('fecha')
        historical_df = historical_df.dropna(subset=['fecha', 'monto total'])
        print(f"Loaded historical data: {len(historical_df)} rows")
        
        # Load predictions
        print("Loading predictions...")
        predictions_df = pd.read_csv('predictions.csv')
        predictions_df['fecha'] = predictions_df['fecha'].apply(parse_spanish_date)
        predictions_df = predictions_df.sort_values('fecha')
        predictions_df = predictions_df.dropna(subset=['fecha', 'prediction'])
        print(f"Loaded predictions: {len(predictions_df)} rows")
        
        # Create the plot with professional styling
        plt.figure(figsize=(16, 9))
        
        # Plot historical data
        plt.plot(historical_df['fecha'], historical_df['monto total'], 
                 'b-', linewidth=2.5, label='Datos Históricos', alpha=0.8, marker='o', markersize=3)
        
        # Plot predictions with different style
        plt.plot(predictions_df['fecha'], predictions_df['prediction'], 
                 'r--', linewidth=3, label='Predicciones', alpha=0.9, marker='s', markersize=4)
        
        # Add a vertical line at the transition point
        last_historical_date = historical_df['fecha'].max()
        plt.axvline(x=last_historical_date, color='green', linestyle=':', linewidth=2, alpha=0.8, 
                    label='Inicio de Predicciones')
        
        # Add confidence interval (simple approximation)
        historical_std = historical_df['monto total'].std()
        plt.fill_between(predictions_df['fecha'], 
                        predictions_df['prediction'] - historical_std,
                        predictions_df['prediction'] + historical_std,
                        color='red', alpha=0.2, label='Intervalo de Confianza')
        
        # Formatting and styling
        plt.title('Predicción de Ventas - Series Temporales\nModelo H2O AutoML', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Fecha', fontsize=14, fontweight='bold')
        plt.ylabel('Monto Total (€)', fontsize=14, fontweight='bold')
        
        # Format y-axis with Euro formatting
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
        # Add grid and legend
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='upper left')
        
        # Add some statistics to the plot
        stats_text = f"""Estadísticas:
• Datos históricos: {len(historical_df)} puntos
• Período de predicción: {len(predictions_df)} días
• Último dato histórico: {last_historical_date.strftime('%d %b %Y')}
• Rango de predicción: {predictions_df['fecha'].min().strftime('%d %b %Y')} - {predictions_df['fecha'].max().strftime('%d %b %Y')}"""
        
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        # Adjust layout to prevent cutting off
        plt.tight_layout()
        
        # Save the plot with high quality
        plt.savefig('forecast_plot.png', bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print("Gráfico guardado exitosamente como forecast_plot.png")
        print("Visualization completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    main()
