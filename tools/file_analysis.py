"""
File analysis tools for the ML Pipeline.
Handles CSV reading, analysis, and file sampling operations.
"""

import csv
import json
import numpy as np
import pandas as pd
import re
import traceback
from io import StringIO
from pathlib import Path
from typing import Dict, Any

from config import Config


def try_read_csv_with_options(path: Path, sep_candidates: list, encodings=None, decimal: str = '.'):
    """
    Enhanced version that supports an explicit decimal separator.
    Intenta leer con el motor 'c' (rápido) y si falla, usa 'python' (flexible).
    """
    if encodings is None:
        encodings = ['utf-8', 'latin1', 'cp1252']
    last_exc = None

    for sep in sep_candidates:
        for enc in encodings:
            try:
                # --- Intento 1: Motor 'c' (rápido y soporta decimal) ---
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    decimal=decimal,  # <--- Parámetro clave que faltaba
                    low_memory=False
                )

                if df.shape[1] > 1:
                    print(f"Éxito al leer con motor 'c', sep='{sep}', decimal='{decimal}', encoding='{enc}'.")
                    # Devuelve el decimal usado para consistencia
                    return df, enc, sep, decimal

            except Exception:
                try:
                    # --- Intento 2: Motor 'python' (flexible y soporta decimal) ---
                    df = pd.read_csv(
                        path,
                        sep=sep,
                        encoding=enc,
                        decimal=decimal,  # <--- Parámetro clave que faltaba
                        engine='python'
                    )

                    if df.shape[1] > 1:
                        print(f"Éxito al leer con motor 'python', sep='{sep}', decimal='{decimal}', encoding='{enc}'.")
                        # Devuelve el decimal usado para consistencia
                        return df, enc, sep, decimal

                except Exception as e:
                    last_exc = e
                    continue

    raise last_exc or ValueError("Could not read the CSV with any of the options tried.")


def detect_decimal_separator(sample_text: str) -> str:
    """
    Simple heuristic to detect if the decimal separator is ',' or '.'.
    Counts which of the two appears more frequently after a digit.
    """
    # Look for patterns like "digit,digit" or "digit.digit"
    comma_as_decimal = len(re.findall(r'\d,\d', sample_text))
    period_as_decimal = len(re.findall(r'\d\.\d', sample_text))

    # If there are many more commas than periods in numeric contexts, it's likely the decimal.
    if comma_as_decimal > period_as_decimal * 2:
        return ','
    return '.'


def read_and_analyze_csv(file_path: str, separator: str, decimal: str = '.', pipeline_name: str = None) -> Dict[
    str, Any]:
    """
    Reads a CSV file using the provided column separator and decimal separator,
    and then returns a complete analysis of the resulting DataFrame.
    """
    base_work_dir = Path(Config.CODING_DIR)
    work_dir = base_work_dir / pipeline_name if pipeline_name else base_work_dir
    full_path = work_dir / file_path

    if not full_path.exists():
        return {"read_status": "error", "error_message": f"File not found: {full_path}"}

    try:
        # Read the complete DataFrame using agent parameters
        # We use a robust reading function like the one we defined before
        df, used_enc, used_sep, used_decimal = try_read_csv_with_options(
            full_path,
            sep_candidates=[separator],  # Only test the one provided by the agent
            decimal=decimal
        )

        # Once read, we perform the analysis (extracted from the old function)
        analysis = analyze_dataframe(df)  # Use an auxiliary function to avoid code repetition

        # Combine reading information with analysis
        analysis["detected"] = {
            "read_status": "ok",
            "encoding": used_enc,
            "used_separator": used_sep,
            "used_decimal": used_decimal
        }
        return analysis

    except Exception as exc:
        return {"read_status": "error", "error_message": str(exc), "traceback": traceback.format_exc()}


def analyze_dataframe(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    """
    Takes an already loaded pandas DataFrame and extracts detailed metadata.

    This function performs the analysis on a sample of the DataFrame to be efficient
    and returns a structured dictionary with the information.
    """
    # If the DataFrame is very large, take a sample for analysis
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()  # Use .copy() to avoid pandas warnings

    # Normalize column names (remove whitespace)
    df_sample.columns = [str(c).strip() for c in df_sample.columns]
    columns = df_sample.columns.tolist()

    # --- START OF ANALYSIS ---

    # 1. Data types
    dtypes = {col: str(df_sample[col].dtype) for col in columns}

    # 2. Detect columns that could be dates
    date_candidates = {}
    for col in columns:
        # Only try to convert if the column is not already a date/time type
        if not pd.api.types.is_datetime64_any_dtype(df_sample[col]):
            # Try to convert to date, 'coerce' converts errors to NaT (null)
            parsed = pd.to_datetime(df_sample[col], errors='coerce', dayfirst=True)
            non_null = int(parsed.notna().sum())
            ratio = non_null / max(1, len(parsed))

            # If more than 60% of values could be converted, it's a good candidate
            if ratio >= 0.6:
                samples = df_sample[col].dropna().astype(str).head(10).tolist()
                date_candidates[col] = {"parse_ratio": round(ratio, 3), "sample_values": samples}

    # 3. Null values and cardinality (number of unique values)
    missing_values = {col: int(df_sample[col].isna().sum()) for col in columns}
    unique_counts = {col: int(df_sample[col].nunique(dropna=True)) for col in columns}
    unique_ratio = {col: round(unique_counts[col] / max(1, len(df_sample)), 4) for col in columns}

    # 4. Build the final dictionary with all information
    info = {
        "columns": columns,
        "dtypes": dtypes,
        "n_rows": int(df.shape[0]),  # Count rows from original DataFrame
        "n_columns": int(df.shape[1]),  # Count columns from original DataFrame
        "date_candidates": date_candidates,
        "missing_values": missing_values,
        "unique_counts": unique_counts,
        "unique_ratio": unique_ratio,
        "sample_rows": df_sample.head(5).to_dict(orient='records'),
    }

    return info


def get_file_sample(file_path: str, num_lines: int = 20, pipeline_name: str = None) -> Dict[str, Any]:
    """
    Reads the first `num_lines` of a file and returns them as a text string.
    Tries to decode with various common encodings.
    """
    base_work_dir = Path(Config.CODING_DIR)
    work_dir = base_work_dir / pipeline_name if pipeline_name else base_work_dir
    full_path = work_dir / file_path

    if not full_path.exists():
        return {"read_status": "error", "error_message": f"File not found: {full_path}"}

    try:
        # Leer una muestra de bytes
        raw_sample = full_path.open('rb').read(8192)  # 8KB es suficiente para ~20-50 líneas

        # Intentar decodificar
        sample_text = None
        encoding_used = None
        for enc in ['utf-8', 'latin1', 'cp1252']:
            try:
                sample_text = raw_sample.decode(enc)
                encoding_used = enc
                break
            except UnicodeDecodeError:
                continue

        if not sample_text:
            raise ValueError("Could not decode file sample with common encodings.")

        # Devolver las primeras N líneas
        lines = sample_text.splitlines()
        sample_to_return = "\n".join(lines[:num_lines])

        return {
            "read_status": "ok",
            "sample_text": sample_to_return,
            "encoding_detected": encoding_used,
            "line_count": len(lines[:num_lines])
        }
    except Exception as exc:
        return {"read_status": "error", "error_message": str(exc)}


def check_generated_files(pipeline_name: str = None) -> str:
    """Verify that the necessary pipeline files were generated"""
    base_work_dir = Path(Config.CODING_DIR)
    if pipeline_name:
        work_dir = base_work_dir / pipeline_name
    else:
        work_dir = base_work_dir

    files_to_check = {
        "predictions.csv": work_dir / "predictions.csv",
        "forecast_plot.png": work_dir / "forecast_plot.png",
    }

    results = {}
    for file_name, file_path in files_to_check.items():
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        results[file_name] = {
            "exists": exists,
            "path": str(file_path),
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2) if size > 0 else 0
        }

    return json.dumps(results, indent=2)


def create_file_analysis_wrappers(pipeline_name: str):
    """Create file analysis wrapper functions with pipeline context"""

    def get_file_sample_with_context(file_path: str, num_lines: int = 20) -> str:
        """
        Wrapper for the 'get_file_sample' tool.
        Passes the 'pipeline_name' from context to the real tool.
        Returns a JSON string for the agent.
        """
        result = get_file_sample(file_path, num_lines, pipeline_name)
        return json.dumps(result, indent=2)

    def read_and_analyze_csv_with_context(file_path: str, separator: str, decimal: str = '.') -> str:
        """
        Wrapper for the 'read_and_analyze_csv' tool.
        Passes the 'pipeline_name' from context to the real tool.
        Returns a JSON string for the agent.
        """
        result = read_and_analyze_csv(file_path, separator, decimal, pipeline_name)
        return json.dumps(result, indent=2)

    def check_pipeline_files() -> str:
        """Return JSON describing generated files in the pipeline folder."""
        return check_generated_files(pipeline_name)

    return get_file_sample_with_context, read_and_analyze_csv_with_context, check_pipeline_files
