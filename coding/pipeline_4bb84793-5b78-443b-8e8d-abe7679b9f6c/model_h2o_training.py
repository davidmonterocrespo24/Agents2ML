#!/usr/bin/env python3
# model_h2o_training.py

import argparse
import json
import os
import sys
import time
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML

def detect_sep(sample_bytes):
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
    return best or ','

def try_read_csv(path, sep=None):
    if sep is None:
        with open(path, 'rb') as f:
            sample = f.read(8192)
        sep = detect_sep(sample)
    
    try:
        df = pd.read_csv(path, sep=sep, encoding='utf-8', low_memory=False, header=None)
        first_row = df.iloc[0].astype(str).str.strip()
        if any('nov.' in val or 'dic.' in val for val in first_row) or any(',' in val for val in first_row):
            df.columns = ['fecha', 'monto_total']
            return df, sep, 'utf-8'
        else:
            df_with_headers = pd.read_csv(path, sep=sep, encoding='utf-8', low_memory=False)
            return df_with_headers, sep, 'utf-8'
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False, header=None)
            first_row = df.iloc[0].astype(str).str.strip()
            if any('nov.' in val or 'dic.' in val for val in first_row) or any(',' in val for val in first_row):
                df.columns = ['fecha', 'monto_total']
                return df, ',', 'utf-8'
            else:
                df_with_headers = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False)
                return df_with_headers, ',', 'utf-8'
        except Exception as e:
            raise e

def detect_date_columns(df, thresh=0.75):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            # Check if column contains Spanish date patterns
            sample_values = df[col].head(10).astype(str)
            spanish_date_pattern = r'\d{1,2}\s+(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\.?\s+\d{4}'
            
            date_like_count = sum(sample_values.str.contains(spanish_date_pattern, case=False, na=False))
            if date_like_count > 0:
                date_cols.append(col)
                continue
                
            # Also try parsing with pandas
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                non_null = parsed.notna().sum()
                if len(df) > 0 and (non_null / max(1, len(df))) >= thresh:
                    date_cols.append(col)
            except Exception:
                pass
    return date_cols

def create_time_features(df, col):
    df_col_copy = df[col].copy().astype(str)
    month_replacements = {
        'ene.': '01', 'feb.': '02', 'mar.': '03', 'abr.': '04', 'may.': '05', 'jun.': '06',
        'jul.': '07', 'ago.': '08', 'sep.': '09', 'oct.': '10', 'nov.': '11', 'dic.': '12',
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }
    
    for abbr, num in month_replacements.items():
        df_col_copy = df_col_copy.str.replace(abbr, num, regex=False)
    
    s = pd.to_datetime(df_col_copy, errors='coerce', dayfirst=True)
    
    df[f"{col}__year"] = s.dt.year
    df[f"{col}__month"] = s.dt.month
    df[f"{col}__day"] = s.dt.day
    df[f"{col}__dayofweek"] = s.dt.dayofweek
    df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{col}__quarter"] = s.dt.quarter
    
    return df

def basic_impute_and_cast(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
        elif df[col].dtype == 'object':
            sample_val = df[col].iloc[0] if len(df) > 0 else ''
            if isinstance(sample_val, str) and ',' in sample_val:
                try:
                    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
                except (ValueError, TypeError):
                    df[col] = df[col].fillna('UNKNOWN')
            else:
                df[col] = df[col].fillna('UNKNOWN')
        else:
            df[col] = df[col].fillna('UNKNOWN')
    return df

def summarize_df(df, n=5):
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "n_nulls": {col: int(df[col].isna().sum()) for col in df.columns},
        "sample": df.head(n).to_dict(orient='records')
    }
    return summary

def to_h2o_frame(df):
    return h2o.H2OFrame(df)

def extract_metrics(leader_model, valid_frame):
    perf = leader_model.model_performance(valid_frame)
    metrics = {}
    try:
        metrics['rmse'] = perf.rmse() if hasattr(perf, 'rmse') else None
        metrics['mae'] = perf.mae() if hasattr(perf, 'mae') else None
        metrics['r2'] = perf.r2() if hasattr(perf, 'r2') else None
    except Exception:
        pass
    return metrics

def main(args):
    start_time = time.time()
    try:
        df, detected_sep, used_encoding = try_read_csv(args.file, sep=args.sep)
        
        summary = summarize_df(df, n=3)
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summary, default=str))
        print("LOG_END:DATA_SUMMARY")
        
        if args.target not in df.columns:
            print(f"ERROR_START:Target column '{args.target}' not found:ERROR_END")
            return 1
        
        date_cols = detect_date_columns(df, thresh=0.75)
        print(f"LOG_START:DATE_COLS_DETECTED\n{date_cols}\nLOG_END:DATE_COLS_DETECTED")
        
        # Manual fallback for date detection
        if not date_cols:
            for col in df.columns:
                if df[col].dtype == object:
                    sample_val = str(df[col].iloc[0]) if len(df) > 0 else ''
                    if any(x in sample_val.lower() for x in ['nov', 'dic', 'ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct']):
                        date_cols.append(col)
                        print(f"LOG_START:MANUAL_DATE_DETECTED\nColumn:{col} detected as date via fallback\nLOG_END:MANUAL_DATE_DETECTED")
        
        for c in date_cols:
            try:
                df = create_time_features(df, c)
                features_created = [col for col in df.columns if col.startswith(f'{c}__')]
                print(f"LOG_START:TIME_FEATURES_CREATED\nColumn:{c}\nFeatures:{features_created}\nLOG_END:TIME_FEATURES_CREATED")
            except Exception as e:
                print(f"LOG_START:DATE_FEATURE_ISSUE\nColumn:{c}\n{str(e)}\nLOG_END:DATE_FEATURE_ISSUE")
        
        df = basic_impute_and_cast(df)
        
        for c in date_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
        
        if args.max_rows_for_validation and len(df) > args.max_rows_for_validation:
            df_sample = df.sample(n=args.max_rows_for_validation, random_state=args.seed)
        else:
            df_sample = df
        
        h2o.init(max_mem_size=args.max_mem_size, nthreads=args.nthreads)
        hf = to_h2o_frame(df_sample)
        
        for col in hf.columns:
            try:
                pandas_col = df_sample[col] if col in df_sample.columns else None
                if pandas_col is not None:
                    uniq_ratio = pandas_col.nunique() / max(1, len(pandas_col))
                    if uniq_ratio <= 0.05:
                        hf[col] = hf[col].asfactor()
            except Exception:
                pass
        
        y = args.target
        x = [c for c in hf.columns if c != y]
        
        print(f"LOG_START:FEATURES_TARGET\nFeatures:{x}\nTarget:{y}\nLOG_END:FEATURES_TARGET")
        
        splits = hf.split_frame(ratios=[args.train_ratio, args.valid_ratio], seed=args.seed)
        train = splits[0]
        valid = splits[1] if len(splits) > 1 else None
        
        aml = H2OAutoML(max_models=args.max_models, max_runtime_secs=args.max_runtime_secs,
                       seed=args.seed, stopping_metric=args.stopping_metric)
        if valid is not None:
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        else:
            aml.train(x=x, y=y, training_frame=train)
        
        leader = aml.leader
        model_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        
        metrics = {}
        try:
            metrics = extract_metrics(leader, valid if valid is not None else train)
        except Exception as e:
            print(f"LOG_START:METRIC_EXTRACT_ISSUE\n{str(e)}\nLOG_END:METRIC_EXTRACT_ISSUE")
        
        total_time = time.time() - start_time
        result_summary = {
            "model_path": model_path,
            "metrics": metrics,
            "rows_trained": int(hf.nrow) if hasattr(hf, 'nrow') else len(df_sample),
            "n_cols": len(hf.columns),
            "training_time_secs": total_time,
            "seed": args.seed,
            "h2o_version": h2o.__version__
        }
        
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
    parser = argparse.ArgumentParser(description='H2O AutoML training')
    parser.add_argument('--file', required=True, help='Path to input CSV')
    parser.add_argument('--target', required=True, help='Name of the target column')
    parser.add_argument('--sep', default=None, help='CSV separator')
    parser.add_argument('--output-dir', default='.', help='Directory to save the model')
    parser.add_argument('--max_models', type=int, default=20, help='Maximum number of models')
    parser.add_argument('--max_runtime_secs', type=int, default=600, help='Max runtime seconds')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--valid-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--max_rows_for_validation', type=int, default=200000, help='Max rows for validation')
    parser.add_argument('--max_mem_size', default='4G', help='Max memory for h2o.init')
    parser.add_argument('--nthreads', type=int, default=-1, help='H2O threads')
    parser.add_argument('--stopping_metric', default='AUTO', help='Stopping metric')
    args = parser.parse_args()
    sys.exit(main(args))