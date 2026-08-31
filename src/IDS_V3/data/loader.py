import os
import glob
from pathlib import Path

import pandas as pd

from config import CICIDS_FOLDER, UNSW_FOLDER


def load_file(path):
    extension = Path(path).suffix.lower()

    if extension == '.parquet':
        return pd.read_parquet(path)

    return pd.read_csv(path, low_memory=False)


def find_data_files(folder):
    files = sorted(
        glob.glob(os.path.join(folder, '*.csv'))
        + glob.glob(os.path.join(folder, '*.parquet'))
    )

    if not files:
        raise FileNotFoundError(
            f'No CSV or Parquet files were found in {folder}. '
            'Check the folder path in config.py.'
        )

    return files


def load_dataset(dataset):
    if dataset == 'cicids':
        folder = CICIDS_FOLDER

    elif dataset == 'unsw':
        folder = UNSW_FOLDER

    else:
        raise ValueError("DATASET must be 'cicids' or 'unsw'.")

    files = find_data_files(folder)

    frames = []

    for file in files:
        print('Loading:', os.path.basename(file))

        part = load_file(file)

        part.columns = (
            part.columns
            .astype(str)
            .str.strip()
        )

        frames.append(part)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    # -------------------------------------------------
    # Create binary labels
    # -------------------------------------------------

    if dataset == 'cicids':

        label_column = next(
            (
                c for c in data.columns
                if c.lower() == 'label'
            ),
            None
        )

        if label_column is None:
            raise ValueError(
                'CICIDS label column was not found.'
            )

        text_label = (
            data[label_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # BENIGN = 0
        # Attack = 1
        data['binary_label'] = (
            text_label != 'BENIGN'
        ).astype(int)

    elif dataset == 'unsw':

        label_column = next(
            (
                c for c in data.columns
                if c.lower() == 'label'
            ),
            None
        )

        if label_column is None:
            raise ValueError(
                'UNSW label column was not found.'
            )

        data['binary_label'] = pd.to_numeric(
            data[label_column],
            errors='raise'
        ).astype(int)

    return data, files
