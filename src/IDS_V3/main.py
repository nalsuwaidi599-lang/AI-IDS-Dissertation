import os
import numpy as np
from sklearn.model_selection import train_test_split

import config
from data.loader import load_dataset, find_datasets
from data.preprocessing import preprocess
from features.selection import standard_selection, hybrid_selection
from models.train import train_all
from evaluation.metrics import evaluate_all, print_results, plot_roc, comparison_plot
from explainability.xai import run_shap, run_lime, compare_shap
from utils.logging import Timer

np.random.seed(config.RANDOM_STATE)


def run_pipeline(X, y, feat_names, method, label, out_prefix):
    """runs one full pipeline with a given feature selection method"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if method == 'hybrid':
        X_sel, sel_names, mi, mask = hybrid_selection(
            X, y, feat_names,
            corr_thresh=config.CORRELATION_THRESHOLD,
            n_select=config.N_FEATURES,
            random_state=config.RANDOM_STATE)
    else:
        X_sel, sel_names, mi, mask = standard_selection(
            X, y, feat_names,
            n_select=config.N_FEATURES,
            random_state=config.RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)
    print(f"\n  train: {len(y_train)}, test: {len(y_test)}")

    with Timer("training"):
        models = train_all(X_train, y_train, sel_names, mi, config)

    results, probs = evaluate_all(models, X_test, y_test, X_train, y_train)
    print_results(results, label)
    plot_roc(probs, y_test, f"{config.OUTPUT_DIR}{out_prefix}_roc_{method}.png",
             f"ROC - {label}")

    return results, models, X_train, X_test, y_train, y_test, sel_names


def run_dataset(dataset_name):
    """run the full evaluation on one dataset"""
    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    prefix = dataset_name

    with Timer("loading data"):
        df = load_dataset(dataset_name, config.CICIDS_FOLDER, config.UNSW_FOLDER)
        n = (df['binary_label']==0).sum()
        a = (df['binary_label']==1).sum()
        print(f"  {len(df)} rows, normal={n}, attack={a}")

    with Timer("preprocessing"):
        X, y, feat_names, scaler = preprocess(df, config.SCALER)

    # standard pipeline (baseline)
    res_std, mod_std, Xtr_s, Xte_s, ytr, yte, fn_s = \
        run_pipeline(X, y, feat_names, 'standard',
                     f'{dataset_name.upper()} - Standard', prefix)

    # hybrid pipeline (contribution)
    res_hyb, mod_hyb, Xtr_h, Xte_h, _, _, fn_h = \
        run_pipeline(X, y, feat_names, 'hybrid',
                     f'{dataset_name.upper()} - Hybrid (Contribution)', prefix)

    # comparison
    comparison_plot(res_std, res_hyb,
                    f"{config.OUTPUT_DIR}{prefix}_contribution_comparison.png")

    # explainability
    cnn_h = mod_hyb['CNN'][0]
    cnn_s = mod_std['CNN'][0]

    run_shap(cnn_h, Xte_h, fn_h, f"{config.OUTPUT_DIR}{prefix}_shap_hybrid.png")
    run_lime(cnn_h, Xtr_h, Xte_h, yte, fn_h,
             f"{config.OUTPUT_DIR}{prefix}_lime_hybrid.png")
    compare_shap(cnn_s, cnn_h, Xte_s, Xte_h, fn_s, fn_h,
                 f"{config.OUTPUT_DIR}{prefix}_shap_comparison.png")

    return res_std, res_hyb


def main():
    datasets = find_datasets(config.CICIDS_FOLDER, config.UNSW_FOLDER)

    if not datasets:
        print("No datasets found.")
       
       
        return

    print(f"Found datasets: {datasets}")

    all_results = {}
    for ds in datasets:
        std, hyb = run_dataset(ds)
        all_results[ds] = (std, hyb)

    # summary
    print(f"\n{'#'*60}")
    print("  SUMMARY")
    print(f"{'#'*60}")
    for ds, (std, hyb) in all_results.items():
        print(f"\n--- {ds.upper()} ---")
        print_results(std, f"{ds} standard")
        print_results(hyb, f"{ds} hybrid")

    print(f"\nDone. all outputs in {config.OUTPUT_DIR}")


if __name__ == '__main__':
    main()
