import pandas as pd
import numpy as np
from datetime import datetime
import locale

# Set Spanish locale for date parsing
locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')

# Read the dataset
df = pd.read_csv('e3e0f8c3-d0c6-4620-9df5-fdf5ac58d4e8_ventas.csv', sep=';', header=None, names=['fecha', 'monto_total'])

print("Dataset shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Convert date column to datetime
def parse_spanish_date(date_str):
    try:
        return datetime.strptime(date_str, '%d %b. %Y')
    except ValueError:
        return pd.NaT

df['fecha_dt'] = df['fecha'].apply(parse_spanish_date)

# Convert monto_total to float (handling European decimal format)
df['monto_total_float'] = df['monto_total'].str.replace(',', '.').astype(float)

print("\nAfter conversion:")
print(df.head())
print("\nDate range:")
print("Min date:", df['fecha_dt'].min())
print("Max date:", df['fecha_dt'].max())

# Check for duplicate dates
print("\nDuplicate dates:", df['fecha_dt'].duplicated().sum())

# Extract time-based features
df['year'] = df['fecha_dt'].dt.year
df['month'] = df['fecha_dt'].dt.month
df['day'] = df['fecha_dt'].dt.day
df['dayofweek'] = df['fecha_dt'].dt.dayofweek

print("\nTime-based features:")
print(df[['fecha_dt', 'year', 'month', 'day', 'dayofweek', 'monto_total_float']].head())