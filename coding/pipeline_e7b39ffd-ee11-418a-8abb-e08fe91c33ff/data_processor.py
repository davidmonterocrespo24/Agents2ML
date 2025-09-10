import pandas as pd, json, sys, os

def detect_sep(sample_bytes):
    text = sample_bytes.decode('utf-8', errors='ignore').splitlines()[:10]
    candidates = [',', ';', '\t', '|']
    best, best_count = None, -1
    for c in candidates:
        cols = [len(row.split(c)) for row in text if row.strip()]
        if cols:
            avg = sum(cols) / len(cols)
            if avg > best_count:
                best_count = avg
                best = c
    return best or ','

def try_read_csv(path, sep=None, decimal='.', encodings=['utf-8','latin1','cp1252']):
    if sep is None:
        with open(path,'rb') as f:
            sample=f.read(8192)
        sep=detect_sep(sample)
    for enc in encodings:
        try:
            df=pd.read_csv(path,sep=sep,decimal=decimal,encoding=enc,header=None,dtype=str,low_memory=False)
            return df,sep,enc
        except Exception:
            continue
    # final attempt
    df=pd.read_csv(path,sep=sep,decimal=decimal,encoding='utf-8',header=None,engine='python',dtype=str,low_memory=False)
    return df,sep,'utf-8'

if len(sys.argv)!=2:
    print('Usage: data_processor.py <csv_path>')
    sys.exit(1)
path=sys.argv[1]
df,sep,enc=try_read_csv(path,sep=';',decimal=',')
# infer column count
n_cols=df.shape[1]
report={
    'file':os.path.basename(path),
    'n_rows':int(df.shape[0]),
    'n_cols':int(n_cols),
    'separator':sep,
    'encoding':enc,
    'sample_rows':df.head(5).astype(str).values.tolist()
}
print(json.dumps(report))
