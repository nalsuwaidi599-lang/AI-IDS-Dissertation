import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             fbeta_score, confusion_matrix, roc_auc_score, roc_curve)

from models.train import snort_predict


def score_model(name, y_true, preds, probs):
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        'Model': name,
        'Precision': round(precision_score(y_true, preds, zero_division=0), 4),
        'Recall': round(recall_score(y_true, preds, zero_division=0), 4),
        'F1': round(f1_score(y_true, preds, zero_division=0), 4),
        'F2': round(fbeta_score(y_true, preds, beta=2, zero_division=0), 4),
        'FPR': round(fp / (fp + tn) if (fp + tn) else 0, 4),
        'AUC': round(roc_auc_score(y_true, probs), 4),
    }


def evaluate_all(models_dict, X_test, y_test, X_train, y_train):
    rows = []
    probs_dict = {}

    for name, (model, feat_idx) in models_dict.items():
        if model == 'snort':
            preds, probs = snort_predict(X_test, X_train, y_train, feat_idx)
        else:
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        rows.append(score_model(name, y_test, preds, probs))
        probs_dict[name] = probs

    return pd.DataFrame(rows), probs_dict


def print_results(df, tag=""):
    header = f"Results ({tag})" if tag else "Results"
    print(f"\n{'='*60}")
    print(header)
    print('='*60)
    cols = ['Model', 'Precision', 'Recall', 'F1', 'F2', 'FPR', 'AUC']
    print(df[cols].to_string(index=False))
    print('='*60)


def plot_roc(probs_dict, y_test, path, title="ROC Curves"):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in probs_dict.items():
        fpr_v, tpr_v, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr_v, tpr_v, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def comparison_plot(res_std, res_hyb, path):
    """bar chart comparing standard vs hybrid across key metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric, title in zip(axes,
            ['F1', 'FPR', 'AUC'],
            ['F1-Score', 'False Positive Rate', 'AUC-ROC']):
        models = res_std['Model'].values
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w/2, res_std[metric].values, w, label='Standard')
        ax.bar(x + w/2, res_hyb[metric].values, w, label='Hybrid (ours)')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")
