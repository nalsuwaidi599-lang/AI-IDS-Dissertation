import os

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import ExtraTreesClassifier

from config import (
    TOP_FEATURES,
    RESULTS_FOLDER,
    DATASET
)


def select_shap_features(
    X_train_all,
    X_val_all,
    X_test_all,
    y_train,
    all_feature_names
):

    print(
        "Starting SHAP feature selection..."
    )

    # Faster Extra Trees selector
    selector_model = (
        ExtraTreesClassifier(
            n_estimators=60,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )

    print(
        "Training Extra Trees..."
    )

    selector_model.fit(
        X_train_all,
        np.asarray(
            y_train
        ).ravel()
    )

    print(
        "Extra Trees training completed."
    )

    # -------------------------------------------------
    # Use 300 training records for SHAP
    # -------------------------------------------------

    sample_size = min(
        300,
        len(X_train_all)
    )

    rng = np.random.default_rng(
        42
    )

    sample_indices = rng.choice(
        len(X_train_all),
        size=sample_size,
        replace=False
    )

    background_sample = (
        X_train_all[
            sample_indices
        ]
    )

    print(
        f"Calculating SHAP values "
        f"using {sample_size} records..."
    )

    # -------------------------------------------------
    # TreeSHAP
    # -------------------------------------------------

    selector_explainer = (
        shap.TreeExplainer(
            selector_model
        )
    )

    shap_values = (
        selector_explainer
        .shap_values(
            background_sample,
            check_additivity=False
        )
    )

    # -------------------------------------------------
    # Handle SHAP output formats
    # -------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        attack_shap = np.asarray(
            shap_values[-1]
        )

    else:

        attack_shap = np.asarray(
            shap_values
        )

        # New SHAP versions may return:
        #
        # records × features × classes
        if attack_shap.ndim == 3:

            attack_shap = (
                attack_shap[
                    :,
                    :,
                    -1
                ]
            )

    # -------------------------------------------------
    # Global feature importance
    # -------------------------------------------------

    shap_importance = (
        np.mean(
            np.abs(
                attack_shap
            ),
            axis=0
        )
        .reshape(-1)
    )

    # Safety check
    if (
        len(shap_importance)
        !=
        len(all_feature_names)
    ):

        raise ValueError(
            f"SHAP produced "
            f"{len(shap_importance)} "
            f"importance values, "
            f"but the dataset has "
            f"{len(all_feature_names)} "
            f"features."
        )

    # -------------------------------------------------
    # Rank features
    # -------------------------------------------------

    ranking = pd.DataFrame({
        "Feature":
            all_feature_names,

        "MeanAbsoluteSHAP":
            shap_importance
    })

    ranking = (
        ranking
        .sort_values(
            "MeanAbsoluteSHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    n_refined = min(
        TOP_FEATURES,
        len(all_feature_names)
    )

    selected_features = (
        ranking
        .head(n_refined)[
            "Feature"
        ]
        .tolist()
    )

    selected_indices = [
        all_feature_names.index(
            feature
        )
        for feature
        in selected_features
    ]

    # -------------------------------------------------
    # Create refined datasets
    # -------------------------------------------------

    X_train_refined = (
        X_train_all[
            :,
            selected_indices
        ]
    )

    X_val_refined = (
        X_val_all[
            :,
            selected_indices
        ]
    )

    X_test_refined = (
        X_test_all[
            :,
            selected_indices
        ]
    )

    reduction_percentage = (
        100
        *
        (
            1
            -
            n_refined
            /
            len(all_feature_names)
        )
    )

    print(
        "\nSelected SHAP features:"
    )

    print(
        ranking
        .head(n_refined)
        .to_string(
            index=False
        )
    )

    print(
        f"\nFeature count: "
        f"{len(all_feature_names)} "
        f"→ {n_refined}"
    )

    print(
        f"Feature reduction: "
        f"{reduction_percentage:.2f}%"
    )

    # -------------------------------------------------
    # Save feature ranking
    # -------------------------------------------------

    ranking.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            f"{DATASET}_shap_feature_ranking.csv"
        ),
        index=False
    )

    print(
        "\nSHAP feature selection "
        "completed successfully."
    )

    return {
        'ranking':
            ranking,

        'selected_features':
            selected_features,

        'X_train_refined':
            X_train_refined,

        'X_val_refined':
            X_val_refined,

        'X_test_refined':
            X_test_refined,

        'reduction_percentage':
            reduction_percentage
    }
