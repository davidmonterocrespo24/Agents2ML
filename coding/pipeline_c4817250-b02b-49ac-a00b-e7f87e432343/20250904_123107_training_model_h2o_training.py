# TRAINING SCRIPT
# Generated on: 2025-09-04T12:31:07.078555
# Pipeline: pipeline_c4817250-b02b-49ac-a00b-e7f87e432343
# Filename: model_h2o_training.py
# Arguments: --file c4817250-b02b-49ac-a00b-e7f87e432343_ventas.csv --sep ; --max_runtime_secs 180 --max_models 5 --nthreads 2 --max_mem_size 1G
# Script Type: training

#!/usr/bin/env python3
# model_h2o_training.py

# Generic, robust script for training with H2O AutoML with time series adaptation

# Mandatory structured outputs on success:
# MODEL_PATH_START:<path>:MODEL_PATH_END
# METRICS_START:<json>:METRICS_END

# On critical error:
# ERROR_START:<message>:ERROR_END

import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime
import locale
import re

import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

# Set locale for Spanish date parsing and European number format
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')
    except locale.Error:
        print("LOG_START:WARNING: Could not set Spanish locale, using default:LOG_END")

def detect_sep(sample_bytes: bytes):
    # Try common separators on a byte sample
    text = sample_bytes.decode('utf-8', errors='ignore').splitlines()[0:10]
    text = '\n'.join(text)
    candidates = [',',';','\t','|']
    best = None
    best_count = -1
    for c in candidates:
        cols = [len(row.split(c)) for row in text.splitlines() if row.strip()]
        if cols:
            avg = sum(cols)/len(cols)
            if avg > best_count:
                best_count = avg
                best = c
    return best or ';'  # Default to semicolon based on sample

def try_read_csv(path, sep=None, encoding_hints=['utf-8','latin1','cp1252']):
    # Try reading with multiple encodings and separators
    if sep is None:
        with open(path, 'rb') as f:
            sample = f.read(8192)
        sep = detect_sep(sample)
    
    last_exc = None
    for enc in encoding_hints:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, header=None)
            return df, sep, enc
        except Exception as e:
            last_exc = e
    
    # Fallback: let pandas try autodetect with python engine
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False, header=None)
        return df, ';', 'utf-8'
    except Exception as e:
        raise last_exc or e

def parse_spanish_date(date_str):
    """Parse Spanish date format like '15 nov. 2023'"""
    try:
        # Replace Spanish month abbreviations
        month_map = {
            'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
            'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
        }
        
        for esp, eng in month_map.items():
            date_str = re.sub(rf'\b{esp}\b\.?', eng, date_str, flags=re.IGNORECASE)
        
        return pd.to_datetime(date_str, format='%d %b %Y', errors='coerce')
    except Exception:
        return pd.NaT

def parse_european_number(num_str):
    """Parse European number format with comma as decimal separator"""
    try:
        if isinstance(num_str, str):
            # Remove thousand separators (dots or spaces) and replace comma with dot
            cleaned = num_str.replace('.', '').replace(',', '.').strip()
            return float(cleaned)
        return float(num_str)
    except (ValueError, TypeError):
        return np.nan

def detect_date_columns(df: pd.DataFrame, thresh=0.75):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            # Try Spanish date parsing
            parsed = df[col].apply(parse_spanish_date)
            non_null = parsed.notna().sum()
            if len(df) > 0 and (non_null / max(1, len(df))) >= thresh:
                date_cols.append(col)
    return date_cols

def create_time_features(df: pd.DataFrame, col):
    s = df[col]  # Already parsed as datetime
    df[f"{col}__year"] = s.dt.year
    df[f"{col}__month"] = s.dt.month
    df[f"{col}__day"] = s.dt.day
    df[f"{col}__dayofweek"] = s.dt.dayofweek
    df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{col}__is_month_start"] = s.dt.is_month_start.astype(int)
    df[f"{col}__is_month_end"] = s.dt.is_month_end.astype(int)
    return df

def basic_impute_and_cast(df: pd.DataFrame):
    # Simple imputations
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            df[col] = df[col].fillna('UNKNOWN')
    return df

def summarize_df(df: pd.DataFrame, n=5):
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_nulls": {col: int(df[col].isna().sum()) for col in df.columns},
        "sample": df.head(n).to_dict(orient='records')
    }
    return summary

def to_h2o_frame(df: pd.DataFrame):
    # Convert pandas to H2OFrame
    h2o_frame = h2o.H2OFrame(df)
    return h2o_frame

def extract_metrics(leader_model, valid_frame, problem_type):
    perf = leader_model.model_performance(valid_frame)
    metrics = {}
    try:
        # Regression metrics
        metrics['rmse'] = perf.rmse() if hasattr(perf, 'rmse') else None
        metrics['mae'] = perf.mae() if hasattr(perf, 'mae') else None
        metrics['r2'] = perf.r2() if hasattr(perf, 'r2') else None
    except Exception:
        pass
    return metrics

def main(args):
    start_time = time.time()
    try:
        # ---------- Read and preprocess ----------
        df, detected_sep, used_encoding = try_read_csv(args.file, sep=args.sep)
        
        # Rename columns based on sample structure (date, amount)
        if len(df.columns) >= 2:
            df.columns = ['fecha', 'monto_total']
        else:
            print("ERROR_START:Dataset must have at least 2 columns: fecha and monto_total:ERROR_END")
            return 1
        
        # Parse European number format
        df['monto_total'] = df['monto_total'].apply(parse_european_number)
        
        # Parse Spanish dates
        df['fecha'] = df['fecha'].apply(parse_spanish_date)
        
        # ---------- Initial summary ----------
        summary = summarize_df(df, n=3)
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summary, default=str))
        print("LOG_END:DATA_SUMMARY")
        
        # ---------- Validations ----------
        if 'monto_total' not in df.columns:
            print("ERROR_START:Target column 'monto_total' not found:ERROR_END")
            return 1
        
        # Check if target is constant
        if df['monto_total'].nunique() <= 1:
            print("ERROR_START:Target column 'monto_total' is constant:ERROR_END")
            return 1
        
        # ---------- Create time features ----------
        if 'fecha' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha']):
            df = create_time_features(df, 'fecha')
            # Sort by date for time series
            df = df.sort_values('fecha').reset_index(drop=True)
            
            # Create lag features for time series
            for lag in [1, 2, 3, 7]:  # 1,2,3 days and 1 week lag
                if len(df) > lag:
                    df[f'monto_total_lag_{lag}'] = df['monto_total'].shift(lag)
        
        # ---------- Basic imputation ----------
        df = basic_impute_and_cast(df)
        
        # Remove rows with NaN in target or essential features
        df = df.dropna(subset=['monto_total'])
        
        # For small datasets, use all data for training
        if len(df) < 100:
            df_sample = df.copy()
            train_ratio = 1.0
            valid_ratio = 0.0
        else:
            # Sample if dataset is large
            if args.max_rows_for_validation and len(df) > args.max_rows_for_validation:
                df_sample = df.sample(n=args.max_rows_for_validation, random_state=args.seed)
            else:
                df_sample = df
            
            train_ratio = args.train_ratio
            valid_ratio = args.valid_ratio
        
        # ---------- Initialize H2O ----------
        h2o.init(max_mem_size=args.max_mem_size, nthreads=args.nthreads)
        
        # Convert to H2OFrame
        hf = to_h2o_frame(df_sample)
        
        # Mark appropriate columns as factors
        for col in hf.columns:
            if col.startswith('fecha__') or col in ['fecha__is_weekend', 'fecha__is_month_start', 'fecha__is_month_end']:
                hf[col] = hf[col].asfactor()
        
        # ---------- Define x, y ----------
        y = 'monto_total'
        x = [c for c in hf.columns if c != y and c != 'fecha']  # Exclude original date column
        
        # ---------- Split train/valid ----------
        if valid_ratio > 0:
            splits = hf.split_frame(ratios=[train_ratio], seed=args.seed)
            train = splits[0]
            valid = splits[1] if len(splits) > 1 else None
        else:
            train = hf
            valid = None
        
        # ---------- Train AutoML ----------
        aml = H2OAutoML(max_models=min(args.max_models, 10),  # Fewer models for small datasets
                       max_runtime_secs=args.max_runtime_secs,
                       seed=args.seed, 
                       stopping_metric=args.stopping_metric,
                       nfolds=min(5, max(2, len(df_sample) // 10)))  # Adaptive cross-validation
        
        if valid is not None:
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        else:
            aml.train(x=x, y=y, training_frame=train)
        
        leader = aml.leader
        # Save best model
        model_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        
        # ---------- Metrics ----------
        metrics = {}
        try:
            if valid is not None:
                metrics = extract_metrics(leader, valid, 'regression')
            else:
                metrics = extract_metrics(leader, train, 'regression')
        except Exception as e:
            print(f"LOG_START:METRIC_EXTRACT_ISSUE\n{str(e)}\nLOG_END:METRIC_EXTRACT_ISSUE")
        
        # Final summary
        total_time = time.time() - start_time
        result_summary = {
            "model_path": model_path,
            "metrics": metrics,
            "rows_trained": int(hf.nrow) if hasattr(hf, 'nrow') else len(df_sample),
            "n_features": len(x),
            "training_time_secs": total_time,
            "seed": args.seed,
            "h2o_version": h2o.__version__,
            "dataset_size": len(df)
        }
        
        # Structured outputs
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        print("METRICS_START:" + json.dumps(result_summary, default=str) + ":METRICS_END")
        return 0
    
    except Exception as e:
        err_msg = str(e).replace('\n', ' ')
        print(f"ERROR_START:{err_msg}:ERROR_END")
        return 2
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H2O AutoML training for time series sales data')
    parser.add_argument('--file', required=True, help='Path to input CSV')
    parser.add_argument('--target', required=False, default='monto_total', help='Name of the target column')
    parser.add_argument('--sep', required=False, default=';', help='CSV separator (default: ;)')
    parser.add_argument('--output-dir', required=False, default='.', help='Directory to save the model')
    parser.add_argument('--max_models', type=int, default=10, help='Maximum number of models for AutoML')
    parser.add_argument('--max_runtime_secs', type=int, default=300, help='Max runtime seconds for AutoML')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--valid-ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--max_rows_for_validation', type=int, default=1000, help='Max rows for fast validations')
    parser.add_argument('--max_mem_size', type=str, default='2G', help='Max memory for h2o.init')
    parser.add_argument('--nthreads', type=int, default=2, help='H2O threads')
    parser.add_argument('--stopping_metric', type=str, default='RMSE', help='Stopping metric for AutoML')
    args = parser.parse_args()
    sys.exit(main(args))
