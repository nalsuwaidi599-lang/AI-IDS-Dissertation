import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import QUICK_MODE, MAX_RECORDS


def stratified_sample(
    data,
    target,
    n,
    seed=42
):
    if n is None or len(data) <= n:
        return data.reset_index(drop=True)

    sampled, _ = train_test_split(
        data,
        train_size=n,
        random_state=seed,
        stratify=data[target]
    )

    return sampled.reset_index(drop=True)


def preprocess_dataset(df):

    # -------------------------------------------------
    # 1. Remove exact duplicate rows
    # -------------------------------------------------

    before_duplicates = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f'Exact duplicate rows removed: '
        f'{duplicates_removed:,}'
    )

    # -------------------------------------------------
    # 2. Stratified sampling
    # -------------------------------------------------

    working_limit = (
        20_000
        if QUICK_MODE
        else MAX_RECORDS
    )

    df = stratified_sample(
        df,
        'binary_label',
        working_limit,
        seed=42
    )

    print(
        f'Records used in this run: '
        f'{len(df):,}'
    )

    # -------------------------------------------------
    # 3. Remove labels / identifiers
    # -------------------------------------------------

    drop_names = {
        'label',
        'binary_label',
        'attack_cat',
        'attack category',
        'id',
        'unnamed: 0',
        'flow id',
        'source ip',
        'destination ip',
        'srcip',
        'dstip',
        'timestamp'
    }

    feature_columns = [
        c
        for c in df.columns
        if c.strip().lower()
        not in drop_names
    ]

    X_raw = df[
        feature_columns
    ].copy()

    y = (
        df['binary_label']
        .astype(int)
        .copy()
    )

    # -------------------------------------------------
    # 4. Keep numeric features
    # -------------------------------------------------

    non_numeric = (
        X_raw
        .select_dtypes(
            exclude=[np.number]
        )
        .columns
        .tolist()
    )

    X_raw = (
        X_raw
        .select_dtypes(
            include=[np.number]
        )
        .copy()
    )

    # Replace infinity with missing values
    X_raw = X_raw.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X_raw.shape[1] == 0:
        raise ValueError(
            'No numeric input features remain '
            'after removing labels/identifiers.'
        )

    # -------------------------------------------------
    # 5. Remove duplicate feature vectors
    # -------------------------------------------------

    feature_duplicate_mask = (
        X_raw.duplicated(
            keep='first'
        )
    )

    feature_duplicates_removed = int(
        feature_duplicate_mask.sum()
    )

    X_raw = (
        X_raw
        .loc[
            ~feature_duplicate_mask
        ]
        .reset_index(drop=True)
    )

    y = (
        y
        .loc[
            ~feature_duplicate_mask
        ]
        .reset_index(drop=True)
    )

    print(
        f'Numeric features retained: '
        f'{X_raw.shape[1]}'
    )

    print(
        'Excluded non-numeric columns:',
        non_numeric
    )

    print(
        'Duplicate numeric feature vectors removed:',
        f'{feature_duplicates_removed:,}'
    )

    print(
        'Missing numeric values before imputation:',
        int(
            X_raw
            .isna()
            .sum()
            .sum()
        )
    )

    # -------------------------------------------------
    # 6. 70 / 15 / 15 split
    # -------------------------------------------------

    X_train_raw, X_temp_raw, y_train, y_temp = (
        train_test_split(
            X_raw,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )
    )

    X_val_raw, X_test_raw, y_val, y_test = (
        train_test_split(
            X_temp_raw,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )
    )

    # -------------------------------------------------
    # 7. Leakage check
    # -------------------------------------------------

    train_hashes = set(
        pd.util.hash_pandas_object(
            X_train_raw,
            index=False
        ).astype(str)
    )

    test_hashes = set(
        pd.util.hash_pandas_object(
            X_test_raw,
            index=False
        ).astype(str)
    )

    overlap_count = len(
        train_hashes.intersection(
            test_hashes
        )
    )

    print(
        f'Train: '
        f'{len(X_train_raw):,} '
        f'({len(X_train_raw)/len(X_raw):.1%})'
    )

    print(
        f'Validation: '
        f'{len(X_val_raw):,} '
        f'({len(X_val_raw)/len(X_raw):.1%})'
    )

    print(
        f'Test: '
        f'{len(X_test_raw):,} '
        f'({len(X_test_raw)/len(X_raw):.1%})'
    )

    print(
        'Identical feature records shared '
        'by train and test:',
        overlap_count
    )

    # -------------------------------------------------
    # 8. Median imputation
    # -------------------------------------------------

    imputer = SimpleImputer(
        strategy='median'
    )

    X_train_imp = (
        imputer
        .fit_transform(
            X_train_raw
        )
    )

    X_val_imp = (
        imputer
        .transform(
            X_val_raw
        )
    )

    X_test_imp = (
        imputer
        .transform(
            X_test_raw
        )
    )

    # -------------------------------------------------
    # 9. Standardisation
    # -------------------------------------------------

    scaler = StandardScaler()

    X_train_all = (
        scaler
        .fit_transform(
            X_train_imp
        )
        .astype('float32')
    )

    X_val_all = (
        scaler
        .transform(
            X_val_imp
        )
        .astype('float32')
    )

    X_test_all = (
        scaler
        .transform(
            X_test_imp
        )
        .astype('float32')
    )

    all_feature_names = (
        X_train_raw
        .columns
        .tolist()
    )

    assert overlap_count == 0, (
        'Leakage warning: identical '
        'feature records exist '
        'in train and test.'
    )

    print(
        'Leakage-safe preprocessing completed.'
    )

    return {
        'df': df,

        'X_train_raw': X_train_raw,
        'X_val_raw': X_val_raw,
        'X_test_raw': X_test_raw,

        'X_train_all': X_train_all,
        'X_val_all': X_val_all,
        'X_test_all': X_test_all,

        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,

        'all_feature_names':
            all_feature_names,

        'duplicates_removed':
            duplicates_removed,

        'feature_duplicates_removed':
            feature_duplicates_removed,

        'overlap_count':
            overlap_count,

        'before_duplicates':
            before_duplicates
    }
