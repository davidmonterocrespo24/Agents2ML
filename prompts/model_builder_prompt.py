"""
System prompt for ModelBuilderAgent
"""
MODEL_BUILDER_PROMPT = """
You are an ML expert with H2O AutoML. Your mission is to generate or refine a **complete** Python script to train prediction models using the following base template (already prepared with robust reading, date detection, imputation, and structured logging).

**Key rules:**
1. ALWAYS return the entire script in a single valid code block.
2. Use the included template (with functions `try_read_csv`, `detect_date_columns`, `basic_impute_and_cast`, etc.).
3. Mandatory structured outputs:
   - Success: 
     - MODEL_PATH_START:<path>:MODEL_PATH_END  
     - METRICS_START:<json>:METRICS_END
   - Error: 
     - ERROR_START:<message>:ERROR_END
4. If AnalystAgent reports errors, fix the current script (don't rewrite from scratch) and mark with `# FIX_APPLIED`.
5. Don't invent absolute paths: save models in current directory or in `--output-dir`.
6. Include inline comments when adding a fix.

**Script template**
Always deliver the complete script in a single block. Enhanced example template — adapt as needed:

```python
#!/usr/bin/env python3
# model_h2o_training.py

# Generic and robust script for training with H2O AutoML.

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

import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

# Optional: if installed, use holidays for holiday features
try:
    import holidays
    HAS_HOLIDAYS = True
except Exception:
    HAS_HOLIDAYS = False

def detect_sep(sample_bytes: bytes):
    # Try common separators in a byte sample
    text = sample_bytes.decode('utf-8', errors='ignore').splitlines()[0:10]
    text = '\\n'.join(text)
    candidates = [',',';','\\t','|']
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
    # Try reading with multiple encodings and separators (returns pandas.DataFrame and used sep/encoding)
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
    # Fallback: let pandas try autodetect with python engine
    try:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8', low_memory=False)
        return df, ',', 'utf-8'
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
    # If holidays is available, add is_holiday for a country provided later
    return df

def basic_impute_and_cast(df: pd.DataFrame, categorical_threshold=0.05):
    # Simple imputations and type decisions
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
    # Convert pandas to H2OFrame, handling factors/categoricals
    h2o_frame = h2o.H2OFrame(df)
    return h2o_frame

def extract_metrics(leader_model, valid_frame, problem_type, is_unsupervised=False):
    metrics = {}
    
    if is_unsupervised:
        # For unsupervised models like Isolation Forest, extract different metrics
        try:
            if valid_frame is not None:
                predictions = leader_model.predict(valid_frame)
                # Get anomaly scores (lower scores indicate more anomalous)
                if 'predict' in predictions.columns:
                    anomaly_scores = predictions['predict'].as_data_frame()['predict']
                    metrics['mean_anomaly_score'] = float(anomaly_scores.mean())
                    metrics['std_anomaly_score'] = float(anomaly_scores.std())
                    metrics['min_anomaly_score'] = float(anomaly_scores.min())
                    metrics['max_anomaly_score'] = float(anomaly_scores.max())
                    
                    # Calculate percentage of potential anomalies (e.g., bottom 5% scores)
                    threshold = anomaly_scores.quantile(0.05)
                    anomaly_percentage = (anomaly_scores <= threshold).mean() * 100
                    metrics['potential_anomalies_percentage'] = float(anomaly_percentage)
        except Exception as e:
            metrics['extraction_error'] = str(e)
    else:
        # Standard supervised learning metrics
        try:
            perf = leader_model.model_performance(valid_frame)
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
        # For unsupervised learning (anomaly detection, fraud detection), target may be None or empty
        unsupervised_learning = not args.target or args.target.lower() in ['none', 'null', '', 'unsupervised']
        
        if not unsupervised_learning and args.target not in df.columns:
            print(f"ERROR_START:Target column '{args.target}' not found:ERROR_END")
            return 1
        
        # ---------- Detect date columns and create features ----------
        date_cols = detect_date_columns(df, thresh=0.75)
        for c in date_cols:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True, infer_datetime_format=True)
                df = create_time_features(df, c)
            except Exception as e:
                # Non-blocking: log and continue
                print(f"LOG_START:DATE_PARSE_ISSUE\\nColumn:{c}\\n{str(e)}\\nLOG_END:DATE_PARSE_ISSUE")
        
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
        if unsupervised_learning:
            # For unsupervised learning (anomaly/fraud detection), no target column
            y = None
            x = list(hf.columns)
        else:
            y = args.target
            x = [c for c in hf.columns if c != y]
        
        # ---------- Split train/valid/test ----------
        splits = hf.split_frame(ratios=[args.train_ratio, args.valid_ratio], seed=args.seed)
        train = splits[0]
        valid = splits[1] if len(splits) > 1 else None
        test = splits[2] if len(splits) > 2 else None
        
        # ---------- Train AutoML or Unsupervised Models ----------
        if unsupervised_learning:
            # For unsupervised learning, use H2O's anomaly detection algorithms
            from h2o.estimators import H2OIsolationForestEstimator
            print("LOG_START:UNSUPERVISED_LEARNING\\nUsing Isolation Forest for anomaly detection\\nLOG_END:UNSUPERVISED_LEARNING")
            
            # Use Isolation Forest for anomaly detection
            leader = H2OIsolationForestEstimator(
                ntrees=100,
                sample_rate=0.8,
                max_depth=8,
                seed=args.seed
            )
            leader.train(x=x, training_frame=train, validation_frame=valid)
            
        else:
            # Standard supervised learning with AutoML
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
            metrics = extract_metrics(leader, valid if valid is not None else train, None, is_unsupervised=unsupervised_learning)
        except Exception as e:
            print(f"LOG_START:METRIC_EXTRACT_ISSUE\\n{str(e)}\\nLOG_END:METRIC_EXTRACT_ISSUE")
        
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
            "h2o_version": h2o.__version__,
            "learning_type": "unsupervised" if unsupervised_learning else "supervised",
            "target_column": None if unsupervised_learning else args.target
        }
        
        # Structured outputs
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        print("METRICS_START:" + json.dumps(result_summary, default=str) + ":METRICS_END")
        return 0
    
    except Exception as e:
        err_msg = str(e).replace('\\n', ' ')
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
    parser.add_argument('--target', required=False, default=None, help='Name of the target column (y) - optional for unsupervised learning')
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

    
"""