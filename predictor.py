"""Load the saved pipeline, validate user input and make predictions."""
from pathlib import Path
import json
import joblib
import pandas as pd
import neural_models

OUT = Path(__file__).parent / 'outputs'
MODEL = joblib.load(OUT / 'final_pipeline.joblib')
FEATURES = json.loads((OUT / 'feature_columns.json').read_text())
CLASS_NAMES = ['Low (0-49 transactions)', 'Medium (50-199 transactions)', 'High (200+ transactions)']
MONTHS = [f'month_{i:02d}' for i in range(1, 13)]
NON_NEGATIVE = [c for c in FEATURES if c.startswith(('month_', 'count_', 'category_'))] + [
    'lifetime_count', 'account_count', 'active_days', 'recency_days', 'active_months_12m', 'fin_products']
MAX_ROWS = 1000


def check_input(df):
    """Return a clean numeric copy of df, or raise ValueError with a helpful message."""
    if len(df) == 0 or len(df) > MAX_ROWS:
        raise ValueError(f'Please provide between 1 and {MAX_ROWS} rows.')
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing[:10]) + (' ...' if len(missing) > 10 else ''))
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    not_numeric = [c for c in FEATURES if (X[c].isna() & df[c].notna()).any()]
    if not_numeric:
        raise ValueError('These columns must contain numbers only: ' + ', '.join(not_numeric))
    if X[MONTHS].isna().any().any():
        raise ValueError('All 12 monthly counts (month_01 ... month_12) are required.')
    negative = [c for c in NON_NEGATIVE if c in X and (X[c] < 0).any()]
    if negative:
        raise ValueError('Counts cannot be negative. Check: ' + ', '.join(negative))
    if not X['debit_share'].dropna().between(0, 1).all():
        raise ValueError('debit_share must be between 0 and 1.')
    return X


def predict(df):
    """Predict a demand class, its probability and a plain-language sentence for each row."""
    X = check_input(df)
    proba = MODEL.predict_proba(X)
    results = []
    for i, p in enumerate(proba):
        k = int(p.argmax())
        results.append({
            'row': i + 1,
            'predicted_class': CLASS_NAMES[k],
            'probability': round(float(p[k]), 3),
            'P(Low)': round(float(p[0]), 3), 'P(Medium)': round(float(p[1]), 3), 'P(High)': round(float(p[2]), 3),
            'explanation': (f'Row {i + 1}: the model expects {CLASS_NAMES[k].lower()} between Nov 2015 and Jan 2016 '
                            f'({p[k]:.0%} model probability). This customer made {int(X.iloc[i]["count_3m"])} '
                            f'transactions in the last three months.'),
        })
    return pd.DataFrame(results)
