import os, sys, glob
import pandas as pd


def load_file(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_cicids(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")) +
                   glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        print(f"No files found in {folder}")
        print("Get the dataset from kaggle: cicids2017")
        sys.exit(1)

    dfs = []
    for f in files:
        print(f"  loading {os.path.basename(f)}")
        chunk = load_file(f)
        chunk.columns = chunk.columns.str.strip()
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    # convert labels to binary: 0 = benign, 1 = attack
    df['binary_label'] = (df['Label'] != 'BENIGN').astype(int)
    print(f"  loaded {len(df)} rows total")
    return df


def load_unsw(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")) +
                   glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        print(f"No files found in {folder}")
        sys.exit(1)

    dfs = []
    for f in files:
        print(f"  loading {os.path.basename(f)}")
        dfs.append(load_file(f))

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    # unsw already has 0/1 label column
    df.rename(columns={'label': 'binary_label'}, inplace=True)
    print(f"  loaded {len(df)} rows total")
    return df


def load_dataset(name, cicids_path, unsw_path):
    print(f"Loading {name} dataset...")
    if name == 'cicids':
        return load_cicids(cicids_path)
    elif name == 'unsw':
        return load_unsw(unsw_path)
    else:
        print(f"unknown dataset: {name}")
        sys.exit(1)


def find_datasets(cicids_path, unsw_path):
    """check which dataset folders actually have files in them"""
    found = []
    for name, folder in [('cicids', cicids_path), ('unsw', unsw_path)]:
        files = glob.glob(os.path.join(folder, "*.csv")) + \
                glob.glob(os.path.join(folder, "*.parquet"))
        if files:
            found.append(name)
    return found
