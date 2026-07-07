import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


def mi_ranking(X, y, names, random_state=42):
    """rank features by mutual information"""
    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    ranking = pd.Series(mi, index=names).sort_values(ascending=False)
    return ranking


def drop_correlated(X, names, threshold=0.95):
    """if two features correlate above threshold, drop the one with lower variance"""
    corr = np.corrcoef(X, rowvar=False)
    n = len(names)
    to_drop = set()

    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i+1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                # keep whichever has higher variance
                if np.var(X[:, i]) >= np.var(X[:, j]):
                    to_drop.add(j)
                else:
                    to_drop.add(i)

    keep = [i for i in range(n) if i not in to_drop]
    kept_names = [names[i] for i in keep]
    print(f"  correlation filter: dropped {len(to_drop)}, kept {len(kept_names)}")
    return X[:, keep], kept_names, keep


def run_rfe(X, y, names, n_select=15, random_state=42):
    n_select = min(n_select, len(names))
    print(f"  RFE: selecting {n_select} from {len(names)} features")
    rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=n_select, step=3)
    selector.fit(X, y)

    selected = [f for f, s in zip(names, selector.support_) if s]
    X_out = X[:, selector.support_]
    return X_out, selected, selector


def standard_selection(X, y, names, n_select=15, random_state=42):
    """baseline approach: MI ranking then RFE"""
    print("\nStandard feature selection...")

    ranking = mi_ranking(X, y, names, random_state)
    print("  top 10 by MI:")
    for f, s in ranking.head(10).items():
        print(f"    {f}: {s:.4f}")

    X_sel, sel_names, rfe = run_rfe(X, y, names, n_select, random_state)
    print(f"  selected: {sel_names}")
    return X_sel, sel_names, ranking, rfe.support_


def hybrid_selection(X, y, names, corr_thresh=0.95, n_select=15, random_state=42):
    """
    proposed hybrid approach (our contribution):
    1) remove redundant features via correlation filtering
    2) rank whats left by mutual information
    3) run RFE on the filtered set

    most IDS papers just use one method. combining all three should
    give a more robust feature set and reduce overfitting since we
    eliminate correlated features before the model even sees them.
    """
    print("\nHybrid feature selection (contribution)...")

    # step 1 - drop correlated
    X_filt, filt_names, kept_idx = drop_correlated(X, names, corr_thresh)

    # step 2 - MI ranking on whats left
    ranking = mi_ranking(X_filt, y, filt_names, random_state)
    print("  top 10 after filtering:")
    for f, s in ranking.head(10).items():
        print(f"    {f}: {s:.4f}")

    # step 3 - RFE on filtered set
    n = min(n_select, len(filt_names))
    X_sel, sel_names, rfe = run_rfe(X_filt, y, filt_names, n, random_state)
    print(f"  final features: {sel_names}")

    # map back to original feature indices
    sel_in_filt = [i for i, s in enumerate(rfe.support_) if s]
    original_idx = [kept_idx[i] for i in sel_in_filt]
    full_mask = np.zeros(len(names), dtype=bool)
    for idx in original_idx:
        full_mask[idx] = True

    return X_sel, sel_names, ranking, full_mask
