# ids_prototype.py
# IDS comparative evaluation using real benchmark datasets
# Datasets: CICIDS2017 (Sharafaldin et al., 2018), UNSW-NB15 (Moustafa & Slay, 2015)
# Models: Snort baseline, FFNN, CNN (proxy), LSTM (proxy)
# Explainability: SHAP, LIME

import os
import sys
import glob
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, fbeta_score,
                             confusion_matrix, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────
# Set these paths to where you downloaded the datasets.
# Both CSV and Parquet formats are supported (Kaggle now defaults to Parquet).
#
# CICIDS2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
#   Download and extract all files into a cicids2017/ folder
#
# UNSW-NB15: https://www.kaggle.com/datasets/dhoogla/unswnb15
#   Or official: https://research.unsw.edu.au/projects/unsw-nb15-dataset
#   Download and extract all files into an unsw-nb15/ folder

CICIDS_FOLDER = "./cicids2017/"       # folder containing the dataset files
UNSW_FOLDER   = "./unsw-nb15/"        # folder containing the UNSW-NB15 files

# Choose which dataset to use: 'cicids' or 'unsw'
DATASET = 'cicids'


def load_file(path):
    """Load a single file, auto-detecting CSV or Parquet format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, low_memory=False)


def load_cicids(folder):
    """Load and merge all CICIDS2017 files (CSV or Parquet)."""
    files = sorted(
        glob.glob(os.path.join(folder, "*.csv")) +
        glob.glob(os.path.join(folder, "*.parquet"))
    )
    if not files:
        print(f"ERROR: No CSV or Parquet files found in {folder}")
        print("Download from: https://www.kaggle.com/datasets/cicdataset/cicids2017")
        print("Extract the files into the cicids2017/ folder.")
        sys.exit(1)

    frames = []
    for f in files:
        print(f"  Loading {os.path.basename(f)}...")
        chunk = load_file(f)
        chunk.columns = chunk.columns.str.strip()
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df)} total rows from {len(files)} files")

    # binary label: BENIGN=0, everything else=1
    df['binary_label'] = (df['Label'] != 'BENIGN').astype(int)
    return df


def load_unsw(folder):
    """Load UNSW-NB15 files (CSV or Parquet)."""
    files = sorted(
        glob.glob(os.path.join(folder, "*.csv")) +
        glob.glob(os.path.join(folder, "*.parquet"))
    )
    if not files:
        print(f"ERROR: No CSV or Parquet files found in {folder}")
        print("Download from: https://www.kaggle.com/datasets/dhoogla/unswnb15")
        print("Or: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        print("Extract the files into the unsw-nb15/ folder.")
        sys.exit(1)

    frames = []
    for f in files:
        print(f"  Loading {os.path.basename(f)}...")
        frames.append(load_file(f))

    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"  Loaded {len(df)} total rows from {len(files)} files")

    # UNSW-NB15 has a 'label' column: 0=normal, 1=attack
    df.rename(columns={'label': 'binary_label'}, inplace=True)
    return df


# ── 1. Load dataset ───────────────────────────────────────────────────
print(f"Loading {DATASET.upper()} dataset...\n")

if DATASET == 'cicids':
    df = load_cicids(CICIDS_FOLDER)
elif DATASET == 'unsw':
    df = load_unsw(UNSW_FOLDER)
else:
    print(f"Unknown dataset '{DATASET}'. Use 'cicids' or 'unsw'.")
    sys.exit(1)

print(f"\nRaw data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class distribution:\n  Normal: {(df['binary_label']==0).sum()}")
print(f"  Attack: {(df['binary_label']==1).sum()}")


# ── 2. Pre-processing ─────────────────────────────────────────────────
# drop non-numeric columns (IPs, timestamps, categorical labels)
label = df['binary_label'].copy()

# keep only numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'binary_label' in numeric_cols:
    numeric_cols.remove('binary_label')

# for UNSW, drop the 'id' column if present
for drop_col in ['id', 'Unnamed: 0']:
    if drop_col in numeric_cols:
        numeric_cols.remove(drop_col)

df_numeric = df[numeric_cols].copy()
print(f"\nNumeric features: {len(numeric_cols)}")

# replace inf values with NaN
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# report missing
missing = df_numeric.isnull().sum().sum()
print(f"Missing values: {missing}")

# fill missing with median
df_numeric.fillna(df_numeric.median(), inplace=True)

# drop duplicates
before = len(df_numeric)
mask = ~df_numeric.duplicated()
df_numeric = df_numeric[mask].reset_index(drop=True)
label = label[mask].reset_index(drop=True)
print(f"Duplicates removed: {before - len(df_numeric)}")
print(f"Clean data: {len(df_numeric)} rows")

# normalise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric.values)
y = label.values

feature_names = numeric_cols
print(f"Features normalised with StandardScaler")


# ── 3. Feature extraction & selection ──────────────────────────────────
# mutual information ranking
print("\nComputing mutual information (this may take a minute)...")
mi = mutual_info_classif(X_scaled, y, random_state=42, n_neighbors=5)
mi_rank = pd.Series(mi, index=feature_names).sort_values(ascending=False)
print("Top 10 features:")
for f, s in mi_rank.head(10).items():
    print(f"  {f}: {s:.4f}")

# RFE to select top features
n_select = min(15, len(feature_names))
print(f"\nRunning RFE (selecting {n_select} features)...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(rf, n_features_to_select=n_select, step=3)
rfe.fit(X_scaled, y)

sel_features = [f for f, keep in zip(feature_names, rfe.support_) if keep]
X_sel = X_scaled[:, rfe.support_]
print(f"Selected features: {sel_features}")


# ── 4. Model training ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(y_train)}, Test: {len(y_test)}")

# Snort baseline: threshold on top MI feature
top_feat = mi_rank.index[0]
if top_feat in sel_features:
    snort_idx = sel_features.index(top_feat)
else:
    snort_idx = 0
threshold = np.median(X_train[y_train == 1, snort_idx])
snort_preds = (X_test[:, snort_idx] > threshold).astype(int)
print(f"Snort baseline: threshold on '{sel_features[snort_idx]}'")

# FFNN
print("Training FFNN...")
ffnn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                     max_iter=300, random_state=42, early_stopping=True,
                     validation_fraction=0.1)
ffnn.fit(X_train, y_train)
ffnn_preds = ffnn.predict(X_test)

# CNN proxy (gradient boosting)
print("Training CNN proxy...")
cnn = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                 learning_rate=0.1, random_state=42)
cnn.fit(X_train, y_train)
cnn_preds = cnn.predict(X_test)

# LSTM proxy (extra trees)
print("Training LSTM proxy...")
lstm = ExtraTreesClassifier(n_estimators=150, max_depth=10,
                            random_state=42, n_jobs=-1)
lstm.fit(X_train, y_train)
lstm_preds = lstm.predict(X_test)

print("All models trained.")


# ── 5. Evaluation ─────────────────────────────────────────────────────
all_models = {
    'Snort':  (snort_preds, X_test[:, snort_idx]),
    'FFNN':   (ffnn_preds,  ffnn.predict_proba(X_test)[:, 1]),
    'CNN':    (cnn_preds,   cnn.predict_proba(X_test)[:, 1]),
    'LSTM':   (lstm_preds,  lstm.predict_proba(X_test)[:, 1]),
}

rows = []
for name, (preds, probs) in all_models.items():
    p  = precision_score(y_test, preds, zero_division=0)
    r  = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    f2 = fbeta_score(y_test, preds, beta=2, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0
    auc = roc_auc_score(y_test, probs)
    rows.append({'Model': name, 'Precision': round(p, 4), 'Recall': round(r, 4),
                 'F1': round(f1, 4), 'F2': round(f2, 4),
                 'FPR': round(fpr, 4), 'AUC': round(auc, 4)})

results = pd.DataFrame(rows)
print(f"\n{'='*60}")
print(results.to_string(index=False))
print(f"{'='*60}")


# ── 6. ROC curve ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, (_, probs) in all_models.items():
    fpr_v, tpr_v, _ = roc_curve(y_test, probs)
    ax.plot(fpr_v, tpr_v, label=f"{name} (AUC={roc_auc_score(y_test, probs):.3f})")
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve Comparison ({DATASET.upper()})')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=150)
print("\nSaved roc_curve_comparison.png")


# ── 7. Explainability ─────────────────────────────────────────────────
try:
    import shap
    print("\nRunning SHAP...")
    explainer = shap.TreeExplainer(cnn)
    sv = explainer.shap_values(X_test[:200])
    shap.summary_plot(sv, X_test[:200], feature_names=sel_features, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150)
    plt.close()
    print("Saved shap_summary.png")
except Exception as e:
    print(f"SHAP skipped: {e}")

try:
    from lime.lime_tabular import LimeTabularExplainer
    print("Running LIME...")
    lx = LimeTabularExplainer(X_train, feature_names=sel_features,
                              class_names=['Normal', 'Attack'], mode='classification')
    idx = np.where(y_test == 1)[0][0]
    exp = lx.explain_instance(X_test[idx], cnn.predict_proba, num_features=10)
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png', dpi=150)
    plt.close()
    print("Saved lime_explanation.png")
    print("\nLIME contributions (sample attack):")
    for feat, w in exp.as_list():
        print(f"  {feat}: {w:+.4f}")
except Exception as e:
    print(f"LIME skipped: {e}")

print("\nDone.")
