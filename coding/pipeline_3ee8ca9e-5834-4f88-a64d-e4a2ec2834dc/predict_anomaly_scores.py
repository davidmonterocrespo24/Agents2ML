#!/usr/bin/env python3
# predict_anomaly_scores.py

import argparse
import json
import os
import sys
import re
import pandas as pd
import numpy as np
import h2o

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

def to_h2o_frame(df: pd.DataFrame):
    h2o_frame = h2o.H2OFrame(df)
    return h2o_frame

def main(args):
    try:
        # Change to pipeline directory
        os.chdir(args.pipeline_dir)
        
        # Load the dataset
        df, detected_sep, used_encoding = try_read_csv(args.data_file, sep=args.sep)
        
        print("LOG_START:DATA_LOADED")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print("LOG_END:DATA_LOADED")
        
        # Apply the same feature engineering as during training
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
        
        # Initialize H2O
        h2o.init(max_mem_size=args.h2o_mem, nthreads=args.nthreads)
        
        # Convert to H2OFrame
        hf = to_h2o_frame(df)
        
        # Mark categorical columns as factors (same logic as training)
        for col in hf.columns:
            try:
                pandas_col = df[col] if col in df.columns else None
                if pandas_col is not None:
                    uniq_ratio = pandas_col.nunique() / max(1, len(pandas_col))
                    if uniq_ratio <= 0.05:  # Same threshold as training
                        hf[col] = hf[col].asfactor()
            except Exception:
                pass
        
        # Load the trained model
        try:
            model = h2o.load_model(args.model_path)
            print("LOG_START:MODEL_LOADED")
            print(f"Model: {model.model_id}")
            print(f"Algorithm: {type(model).__name__}")
            print("LOG_END:MODEL_LOADED")
        except Exception as e:
            print(f"ERROR_START:Failed to load model: {str(e)}:ERROR_END")
            return 1
        
        # Generate predictions (anomaly scores)
        predictions = model.predict(hf)
        
        # Convert predictions to pandas
        pred_df = predictions.as_data_frame()
        
        # Combine with original data
        result_df = df.copy()
        result_df['anomaly_score'] = pred_df['predict']
        result_df['is_anomaly'] = (result_df['anomaly_score'] <= result_df['anomaly_score'].quantile(0.05)).astype(int)
        
        # Save predictions
        output_path = args.output_file or 'predictions.csv'
        result_df.to_csv(output_path, index=False)
        
        # Print structured output
        print(f"PREDICTIONS_FILE_START:{os.path.abspath(output_path)}:PREDICTIONS_FILE_END")
        
        # Log summary
        print("LOG_START:PREDICTION_SUMMARY")
        print(json.dumps({
            "total_predictions": len(result_df),
            "anomalies_detected": int(result_df['is_anomaly'].sum()),
            "anomaly_percentage": float(result_df['is_anomaly'].mean() * 100),
            "mean_anomaly_score": float(result_df['anomaly_score'].mean()),
            "min_anomaly_score": float(result_df['anomaly_score'].min()),
            "max_anomaly_score": float(result_df['anomaly_score'].max())
        }, default=str))
        print("LOG_END:PREDICTION_SUMMARY")
        
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
    parser = argparse.ArgumentParser(description='Generate anomaly predictions using trained Isolation Forest model')
    parser.add_argument('--model-path', required=True, help='Path to the trained H2O model')
    parser.add_argument('--pipeline-dir', required=True, help='Pipeline working directory')
    parser.add_argument('--data-file', required=True, help='Path to input CSV file')
    parser.add_argument('--output-file', default='predictions.csv', help='Output predictions file')
    parser.add_argument('--sep', default=None, help='CSV separator')
    parser.add_argument('--h2o-mem', default='4G', help='H2O memory size')
    parser.add_argument('--nthreads', type=int, default=-1, help='H2O threads')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    sys.exit(main(args))