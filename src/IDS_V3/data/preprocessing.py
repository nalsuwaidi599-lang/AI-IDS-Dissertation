import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess(df, scaler_type='standard'):
    print("\nPreprocessing...")

    label = df['binary_label'].copy()

    # grab only numeric columns, drop id/label
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['binary_label', 'id', 'Unnamed: 0']:
        if col in num_cols:
            num_cols.remove(col)

    df_num = df[num_cols].copy()
    print(f"  numeric features: {len(num_cols)}")

    # clean up inf and nan
    df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing = df_num.isnull().sum().sum()
    print(f"  missing values: {missing}")
    df_num.fillna(df_num.median(), inplace=True)

    # remove duplicate rows
    before = len(df_num)
    keep = ~df_num.duplicated()
    df_num = df_num[keep].reset_index(drop=True)
    label = label[keep].reset_index(drop=True)
    print(f"  removed {before - len(df_num)} duplicates, {len(df_num)} rows left")

    # scale
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X = scaler.fit_transform(df_num.values)
    y = label.values
    print(f"  scaled with {scaler_type}")

    return X, y, num_cols, scaler
