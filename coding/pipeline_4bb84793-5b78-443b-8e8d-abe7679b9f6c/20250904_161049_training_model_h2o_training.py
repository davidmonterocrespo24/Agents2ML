# TRAINING SCRIPT
# Generated on: 2025-09-04T16:10:49.806513
# Pipeline: pipeline_4bb84793-5b78-443b-8e8d-abe7679b9f6c
# Filename: model_h2o_training.py
# Arguments: --file 4bb84793-5b78-443b-8e8d-abe7679b9f6c_ventas.csv --target "monto_total" --sep ";"
# Script Type: training

#!/usr/bin/env python3
# model_h2o_training.py

# Script genérico y robusto para entrenamiento con H2O AutoML.

# Salidas estructuradas obligatorias en éxito:
# MODEL_PATH_START:<ruta>:MODEL_PATH_END
# METRICS_START:<json>:METRICS_END

# En error crítico:
# ERROR_START:<mensaje>:ERROR_END

import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

# Opcional: si está instalado, usa holidays para características de días festivos
try:
    import holidays
    HAS_HOLIDAYS = True
except Exception:
    HAS_HOLIDAYS = False

def detect_sep(sample_bytes: bytes):
    # Intenta separadores comunes en una muestra de bytes
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
    # Intenta leer con múltiples codificaciones y separadores (devuelve pandas.DataFrame y sep/encoding usados)
    if sep is None:
        with open(path, 'rb') as f:
            sample = f.read(8192)
        sep = detect_sep(sample)
    last_exc = None
    
    # FIX_APPLIED: First try reading with header=None to handle files without proper headers
    for enc in encoding_hints:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, header=None)
            # Check if first row looks like data (contains dates and numbers)
            first_row = df.iloc[0].astype(str).str.strip()
            if any('nov.' in val or 'dic.' in val for val in first_row) or any(',' in val for val in first_row):
                # This looks like data, not headers - assign proper column names
                df.columns = ['fecha', 'monto_total']
                return df, sep, enc
            else:
                # Try with headers
                df_with_headers = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                return df_with_headers, sep, enc
        except Exception as e:
            last_exc = e
    
    # Respaldo: deja que pandas intente autodetectar con motor python
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False, header=None)
        # Check if first row looks like data
        first_row = df.iloc[0].astype(str).str.strip()
        if any('nov.' in val or 'dic.' in val for val in first_row) or any(',' in val for val in first_row):
            df.columns = ['fecha', 'monto_total']
            return df, ',', 'utf-8'
        else:
            df_with_headers = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False)
            return df_with_headers, ',', 'utf-8'
    except Exception as e:
        raise last_exc or e

def detect_date_columns(df: pd.DataFrame, thresh=0.75):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object or np.issubdtype(df[col].dtype, np.integer) or np.issubdtype(df[col].dtype, np.floating):
            parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
            non_null = parsed.notna().sum()
            if len(df) > 0 and (non_null / max(1, len(df))) >= thresh:
                date_cols.append(col)
    return date_cols

def create_time_features(df: pd.DataFrame, col):
    s = pd.to_datetime(df[col], errors='coerce')
    df[f"{col}__year"] = s.dt.year
    df[f"{col}__month"] = s.dt.month
    df[f"{col}__day"] = s.dt.day
    df[f"{col}__dayofweek"] = s.dt.dayofweek
    df[f"{col}__is_weekend"] = s.dt.dayofweek.isin([5,6]).astype(int)
    df[f"{col}__is_month_start"] = s.dt.is_month_start.astype(int)
    df[f"{col}__is_month_end"] = s.dt.is_month_end.astype(int)
    # Si holidays está disponible, agrega is_holiday para un país proporcionado después
    return df

def basic_impute_and_cast(df: pd.DataFrame, categorical_threshold=0.05):
    # Imputaciones simples y decisiones de tipo
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
        elif df[col].dtype == 'object':
            # FIX_APPLIED: Handle numeric strings with comma as decimal separator
            if df[col].str.contains(',').any() and df[col].str.replace(',', '').str.isnumeric().all():
                df[col] = df[col].str.replace(',', '.').astype(float)
            else:
                df[col] = df[col].fillna('UNKNOWN')
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
    # Convert pandas to H2OFrame, handling factors/categoricals
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
    try:
        # Classification metrics
        metrics['auc'] = perf.auc() if hasattr(perf, 'auc') else None
        metrics['logloss'] = perf.logloss() if hasattr(perf, 'logloss') else None
    except Exception:
        pass
    return metrics

def main(args):
    start_time = time.time()
    try:
        # ---------- Read ----------
        df, detected_sep, used_encoding = try_read_csv(args.file, sep=args.sep)
        
        # ---------- Initial summary ----------
        summary = summarize_df(df, n=3)
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summary, default=str))
        print("LOG_END:DATA_SUMMARY")
        
        # ---------- Validations ----------
        if args.target not in df.columns:
            print(f"ERROR_START:Target column '{args.target}' not found:ERROR_END")
            return 1
        
        # ---------- Detect date columns and create features ----------
        date_cols = detect_date_columns(df, thresh=0.75)
        for c in date_cols:
            try:
                # FIX_APPLIED: Handle Spanish date format specifically
                df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True, format='mixed')
                df = create_time_features(df, c)
            except Exception as e:
                # Non-blocking: log and continue
                print(f"LOG_START:DATE_PARSE_ISSUE\nColumn:{c}\n{str(e)}\nLOG_END:DATE_PARSE_ISSUE")
        
        # ---------- Basic engineering and cleaning ----------
        df = basic_impute_and_cast(df)
        
        # If dataset is huge and user asked for sampling, sample for quick validations
        if args.max_rows_for_validation and len(df) > args.max_rows_for_validation:
            df_sample = df.sample(n=args.max_rows_for_validation, random_state=args.seed)
        else:
            df_sample = df
        
        # ---------- Initialize H2O ----------
        h2o.init(max_mem_size=args.max_mem_size, nthreads=args.nthreads)
        # Convert to H2OFrame
        hf = to_h2o_frame(df_sample)
        
        # Mark categorical columns as factors in H2O using a heuristic
        for col in hf.columns:
            try:
                pandas_col = df_sample[col] if col in df_sample.columns else None
                if pandas_col is not None:
                    uniq_ratio = pandas_col.nunique() / max(1, len(pandas_col))
                    if uniq_ratio <= args.categorical_unique_ratio_threshold:
                        hf[col] = hf[col].asfactor()
            except Exception:
                pass
        
        # ---------- Define x, y ----------
        y = args.target
        x = [c for c in hf.columns if c != y]
        
        # ---------- Split train/valid/test ----------
        splits = hf.split_frame(ratios=[args.train_ratio, args.valid_ratio], seed=args.seed)
        train = splits[0]
        valid = splits[1] if len(splits) > 1 else None
        test = splits[2] if len(splits) > 2 else None
        
        # ---------- Train AutoML ----------
        aml = H2OAutoML(max_models=args.max_models, max_runtime_secs=args.max_runtime_secs,
                       seed=args.seed, stopping_metric=args.stopping_metric)
        if valid is not None:
            aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        else:
            aml.train(x=x, y=y, training_frame=train)
        
        leader = aml.leader
        # Save best model
        model_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        
        # Try exporting MOJO (optional)
        try:
            mojo_path = None
            # Some H2O versions require different calls; attempt safe approach
            mojo_path = h2o.save_model(model=leader, path=args.output_dir or "./", force=True)
        except Exception:
            mojo_path = None
        
        # ---------- Metrics ----------
        metrics = {}
        try:
            metrics = extract_metrics(leader, valid if valid is not None else train, None)
        except Exception as e:
            print(f"LOG_START:METRIC_EXTRACT_ISSUE\n{str(e)}\nLOG_END:METRIC_EXTRACT_ISSUE")
        
        # Final summary
        total_time = time.time() - start_time
        result_summary = {
            "model_path": model_path,
            "mojo_path": mojo_path,
            "metrics": metrics,
            "rows_trained": int(hf.nrow) if hasattr(hf, 'nrow') else len(df_sample),
            "n_cols": len(hf.columns),
            "training_time_secs": total_time,
            "seed": args.seed,
            "h2o_version": h2o.__version__
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
    parser = argparse.ArgumentParser(description='H2O AutoML training - Generic script')
    parser.add_argument('--file', required=True, help='Path to input CSV')
    parser.add_argument('--target', required=True, help='Name of the target column (y)')
    parser.add_argument('--sep', required=False, default=None, help='CSV separator (autodetect if not provided)')
    parser.add_argument('--output-dir', required=False, default='.', help='Directory to save the model')
    parser.add_argument('--max_models', type=int, default=20, help='Maximum number of models for AutoML')
    parser.add_argument('--max_runtime_secs', type=int, default=600, help='Max runtime seconds for AutoML')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Train split ratio (rest to valid/test per valid_ratio)')
    parser.add_argument('--valid-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--max_rows_for_validation', type=int, default=200000, help='Max rows for fast validations')
    parser.add_argument('--max_mem_size', type=str, default='4G', help='Max memory for h2o.init (e.g. 4G)')
    parser.add_argument('--nthreads', type=int, default=-1, help='H2O threads (-1 = use all)')
    parser.add_argument('--stopping_metric', type=str, default='AUTO', help='Stopping metric for AutoML')
    parser.add_argument('--categorical_unique_ratio_threshold', type=float, default=0.05, help='Threshold to decide if a column is categorical (unique/rows ratio)')
    args = parser.parse_args()
    sys.exit(main(args))
