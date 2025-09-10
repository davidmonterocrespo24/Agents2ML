# TRAINING SCRIPT
# Generated on: 2025-09-10T00:28:35.512343
# Pipeline: pipeline_e7b39ffd-ee11-418a-8abb-e08fe91c33ff
# Filename: model_h2o_training.py
# Arguments: --file e7b39ffd-ee11-418a-8abb-e08fe91c33ff_ventas por categoria tecnologia.csv --output-dir . --target total_amount
# Script Type: training

#!/usr/bin/env python3
import argparse, json, os, sys, time
import pandas as pd, numpy as np
import h2o
from h2o.automl import H2OAutoML

def detect_sep(sample_bytes: bytes):
    text = sample_bytes.decode('utf-8', errors='ignore').splitlines()[:10]
    text = '\n'.join(text)
    candidates = [',', ';', '\t', '|']
    best = None
    best_count = -1
    for c in candidates:
        cols = [len(row.split(c)) for row in text.splitlines() if row.strip()]
        if cols:
            avg = sum(cols) / len(cols)
            if avg > best_count:
                best_count = avg
                best = c
    return best or ','

def try_read_csv(path, sep=None, decimal='.', encoding_hints=('utf-8','latin1','cp1252')):
    if sep is None:
        with open(path, 'rb') as f:
            sample = f.read(8192)
        sep = detect_sep(sample)
    last_exc = None
    for enc in encoding_hints:
        try:
            df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=enc, header=None, low_memory=False, dtype=str)
            return df, sep, enc
        except Exception as e:
            last_exc = e
    try:
        df = pd.read_csv(path, sep=sep, decimal=decimal, encoding='utf-8', header=None, engine='python', low_memory=False, dtype=str)
        return df, sep, 'utf-8'
    except Exception as e:
        raise last_exc or e

def summarize_df(df, n=5):
    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "sample": df.head(n).to_dict(orient='records')}

def to_h2o_frame(df):
    return h2o.H2OFrame(df)

def extract_metrics(leader_model, valid_frame, is_unsupervised=False):
    metrics = {}
    if is_unsupervised:
        try:
            preds = leader_model.predict(valid_frame)
            if 'predict' in preds.columns:
                scores = preds['predict'].as_data_frame()['predict']
                metrics.update({"mean_anomaly_score": float(scores.mean()), "std_anomaly_score": float(scores.std()), "min_anomaly_score": float(scores.min()), "max_anomaly_score": float(scores.max())})
        except Exception as e:
            metrics["unsup_metric_error"] = str(e)
    else:
        try:
            perf = leader_model.model_performance(valid_frame)
            if hasattr(perf, "rmse"): metrics["rmse"] = perf.rmse()
            if hasattr(perf, "mae"): metrics["mae"] = perf.mae()
            if hasattr(perf, "r2"): metrics["r2"] = perf.r2()
            if hasattr(perf, "auc"): metrics["auc"] = perf.auc()
            if hasattr(perf, "logloss"): metrics["logloss"] = perf.logloss()
        except Exception as e:
            metrics["metric_error"] = str(e)
    return metrics

def main(args):
    start = time.time()
    try:
        df_raw, used_sep, used_enc = try_read_csv(args.file, sep=args.sep, decimal=args.decimal)
        print("LOG_START:RAW_LOAD")
        print(json.dumps({"detected_separator": used_sep, "used_encoding": used_enc, "shape": df_raw.shape}))
        print("LOG_END:RAW_LOAD")
        df_raw.columns = ["date_str", "total_amount_raw"]
        df_raw["date_str"] = df_raw["date_str"].str.strip()
        df_raw["total_amount_raw"] = df_raw["total_amount_raw"].str.strip()
        df_raw["total_amount"] = pd.to_numeric(df_raw["total_amount_raw"], errors="coerce")
        df_raw["date"] = pd.to_datetime(df_raw["date_str"], dayfirst=True, errors="coerce")
        df = df_raw[["date", "total_amount"]].copy()
        if df.isna().any().any():
            print("LOG_START:DATA_QUALITY_ISSUES")
            issues = {"null_dates": int(df["date"].isna().sum()), "null_targets": int(df["total_amount"].isna().sum())}
            print(json.dumps(issues))
            print("LOG_END:DATA_QUALITY_ISSUES")
        print("LOG_START:DATA_SUMMARY")
        print(json.dumps(summarize_df(df, n=3), default=str))
        print("LOG_END:DATA_SUMMARY")
        h2o.init(max_mem_size=args.max_mem_size, nthreads=args.nthreads, silent=True, enable_assertions=False)
        hf = to_h2o_frame(df)
        y = args.target
        if y not in hf.columns:
            print(f"ERROR_START:Target column '{y}' not found in processed data.:ERROR_END")
            return 1
        x = [c for c in hf.columns if c != y]
        splits = hf.split_frame(ratios=[args.train_ratio, args.valid_ratio], seed=args.seed)
        train = splits[0]
        valid = splits[1] if len(splits) > 1 else None
        aml = H2OAutoML(max_models=args.max_models, max_runtime_secs=args.max_runtime_secs, seed=args.seed, stopping_metric=args.stopping_metric)
        aml.train(x=x, y=y, training_frame=train, validation_frame=valid)
        leader = aml.leader
        model_path = h2o.save_model(model=leader, path=args.output_dir, force=True)
        metrics = extract_metrics(leader, valid if valid is not None else train, is_unsupervised=False)
        result = {"model_path": model_path, "metrics": metrics, "rows_trained": int(train.nrow), "n_features": len(x), "training_time_secs": round(time.time() - start, 2), "seed": args.seed, "h2o_version": h2o.__version__, "target_column": y, "output_dir": args.output_dir}
        print(f"MODEL_PATH_START:{model_path}:MODEL_PATH_END")
        print("METRICS_START:" + json.dumps(result, default=str) + ":METRICS_END")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an H2O AutoML regression model on the sales CSV")
    parser.add_argument("--file", required=True)
    parser.add_argument("--target", default="total_amount")
    parser.add_argument("--sep", default=";")
    parser.add_argument("--decimal", default=",")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--max_models", type=int, default=20)
    parser.add_argument("--max_runtime_secs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--max_mem_size", default="4G")
    parser.add_argument("--nthreads", type=int, default=-1)
    args = parser.parse_args()
    sys.exit(main(args))

