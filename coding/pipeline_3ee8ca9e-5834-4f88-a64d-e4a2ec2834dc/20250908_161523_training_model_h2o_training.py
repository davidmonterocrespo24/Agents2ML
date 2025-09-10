# TRAINING SCRIPT
# Generated on: 2025-09-08T16:15:23.384051
# Pipeline: pipeline_3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc
# Filename: model_h2o_training.py
# Arguments: --file 3ee8ca9e-5834-4f88-a64d-e4a2ec2834dc.csv --output-dir ./ --max_rows_for_validation 50000 --max_runtime_secs 300
# Script Type: training

#!/usr/bin/env python3
# model_h2o_training.py

import argparse
import json
import os
import sys
import time
import re
import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2OIsolationForestEstimator

def detect_sep(sample_bytes: bytes):
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

def try_read_csv(path, sep=None, encoding_hints=['utf-8','latin1','cp1252']):
    if sep is None:
        with open(path, 'rb') as f:
            sample = f.read(8192)
        sep = detect_sep(sample)
    last_exc = None
    for enc in encoding_hints:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
            return df, sep, enc
        except Exception as e:
            last_exc = e
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False)
        return df, ',', 'utf-8'
    except Exception as e:
        raise last_exc or e

def extract_invoice_features(df: pd.DataFrame, invoice_col='factura_nombre'):
    df = df.copy()
    
    def extract_prefix(text):
        if pd.isna(text) or text == '/':
            return 'UNKNOWN'
        parts = str(text).split('/')
        if len(parts) > 0:
            return parts[0].strip()
        return 'UNKNOWN'
    
    df['invoice_prefix'] = df[invoice_col].apply(extract_prefix)
    
    def has_date_pattern(text):
        if pd.isna(text) or text == '/':
            return 0
        text_str = str(text)
        if re.search(r'/\d{4}/\d{2}/', text_str) or re.search(r'/\d{4}/', text_str):
            return 1
        return 0
    
    df['has_date_format'] = df[invoice_col].apply(has_date_pattern)
    df['invoice_name_length'] = df[invoice_col].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    return df

def basic_impute_and_cast(df: pd.DataFrame):
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
    h2o_frame = h2o.H2OFrame(df)
    return h2o_frame

def extract_metrics(leader_model, valid_frame):
    metrics = {}
    try:
        if valid_frame is not None:
            predictions = leader_model.predict(valid_frame)
            if 'predict' in predictions.columns:
                anomaly_scores = predictions['predict'].as_data_frame()['predict']
                metrics['mean_anomaly_score'] = float(anomaly_scores.mean())
                metrics['std_anomaly_score'] = float(anomaly_scores.std())
                metrics['min_anomaly_score'] = float(anomaly_scores.min())
                metrics['max_anomaly_score'] = float(anomaly_scores.max())
                threshold = anomaly_scores.quantile(0.05)
                anomaly_percentage = (anomaly_scores <= threshold).mean() * 100
                metrics['potential_anomalies_percentage'] = float(anomaly_percentage)
    except Exception as e:
        metrics['extraction_error'] = str(e)
    return metrics

def main(args):
    start_time = time.time()
    try:
        df, detected_sep, used_encoding = try_read_csv(args.file, sep=args.sep)
        
        summary = summarize_df(df, n=3)
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summary, default=str))
        print("LOG_END:DATA_SUMMARY")
        
        df = extract_invoice_features(df, 'factura_nombre')
        
        df['payment_ratio'] = np.where(
            df['factura_monto_total'] > 0,
            df['factura_total_pagos'] / df['factura_monto_total'],
            0
        )
        
        df['is_double_payment'] = ((df['factura_total_pagos'] == 2 * df['factura_monto_total']) & 
                                  (df['factura_monto_total'] > 0)).astype(int)
        
        df['is_zero_payment'] = (df['factura_total_pagos'] == 0).astype(int)
        
        df = basic_impute_and_cast(df)
        
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
                    if uniq_ratio <= args.categorical_unique_ratio_threshold:
                        hf[col] = hf[col].asfactor()
            except Exception:
                pass
        
        x = [c for c in hf.columns if c != 'factura_nombre']
        
        splits = hf.split_frame(ratios=[args.train_ratio, args.valid_ratio], seed=args.seed)
        train = splits[0]
        valid = splits[1] if len(splits) > 1 else None
        
        print("LOG_START:UNSUPERVISED_LEARNING")
        print("Using Isolation Forest for anomaly detection")
        print("LOG_END:UNSUPERVISED_LEARNING")
        
        leader = H2OIsolationForestEstimator(
            ntrees=100,
            sample_rate=0.8,
            max_depth=8,
            seed=args.seed
        )
        leader.train(x=x, training_frame=train, validation_frame=valid)
        
        model_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        
        mojo_path = None
        try:
            mojo_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        except Exception:
            pass
        
        metrics = {}
        try:
            metrics = extract_metrics(leader, valid if valid is not None else train)
        except Exception as e:
            print(f"LOG_START:METRIC_EXTRACT_ISSUE\n{str(e)}\nLOG_END:METRIC_EXTRACT_ISSUE")
        
        total_time = time.time() - start_time
        result_summary = {
            "model_path": model_path,
            "mojo_path": mojo_path,
            "metrics": metrics,
            "rows_trained": int(hf.nrow) if hasattr(hf, 'nrow') else len(df_sample),
            "n_cols": len(hf.columns),
            "training_time_secs": total_time,
            "seed": args.seed,
            "h2o_version": h2o.__version__,
            "learning_type": "unsupervised",
            "target_column": None,
            "features_used": x
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
    parser = argparse.ArgumentParser(description='H2O Isolation Forest for anomaly detection')
    parser.add_argument('--file', required=True, help='Path to input CSV')
    parser.add_argument('--target', required=False, default=None, help='Name of the target column')
    parser.add_argument('--sep', required=False, default=None, help='CSV separator')
    parser.add_argument('--output-dir', required=False, default='.', help='Directory to save the model')
    parser.add_argument('--max_models', type=int, default=20, help='Maximum number of models')
    parser.add_argument('--max_runtime_secs', type=int, default=600, help='Max runtime seconds')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--valid-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--max_rows_for_validation', type=int, default=50000, help='Max rows for validation')
    parser.add_argument('--max_mem_size', type=str, default='4G', help='Max memory for h2o.init')
    parser.add_argument('--nthreads', type=int, default=-1, help='H2O threads')
    parser.add_argument('--stopping_metric', type=str, default='AUTO', help='Stopping metric')
    parser.add_argument('--categorical_unique_ratio_threshold', type=float, default=0.05, help='Categorical threshold')
    args = parser.parse_args()
    sys.exit(main(args))
