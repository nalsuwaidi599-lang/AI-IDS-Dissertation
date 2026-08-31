import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier
)

from lime.lime_tabular import (
    LimeTabularExplainer
)

from config import (
    DATASET,
    RESULTS_FOLDER
)


def generate_lime_explanation(
    X_train_refined,
    X_test_refined,
    y_train,
    y_test,
    selected_features
):

    # -------------------------------------------------
    # Train Random Forest for LIME
    # -------------------------------------------------

    lime_model = (
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    )

    lime_model.fit(
        X_train_refined,
        y_train
    )

    # -------------------------------------------------
    # Create LIME explainer
    # -------------------------------------------------

    lime_explainer = (
        LimeTabularExplainer(
            X_train_refined,

            feature_names=
                selected_features,

            class_names=[
                "Normal",
                "Attack"
            ],

            mode=
                "classification",

            discretize_continuous=True,

            random_state=42
        )
    )

    # -------------------------------------------------
    # Find correctly detected attack
    # -------------------------------------------------

    probabilities = (
        lime_model
        .predict_proba(
            X_test_refined
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    correct_attacks = np.where(
        (
            np.asarray(
                y_test
            ) == 1
        )
        &
        (
            predictions == 1
        )
    )[0]

    if len(correct_attacks) == 0:

        print(
            "No correctly detected "
            "attack available for LIME."
        )

        return None

    sample_index = (
        correct_attacks[0]
    )

    # -------------------------------------------------
    # Explain individual prediction
    # -------------------------------------------------

    lime_result = (
        lime_explainer
        .explain_instance(

            X_test_refined[
                sample_index
            ],

            lime_model
            .predict_proba,

            num_features=min(
                10,
                len(
                    selected_features
                )
            )
        )
    )

    print(
        "True class: Attack"
    )

    print(
        "Predicted class:",
        (
            "Attack"
            if predictions[
                sample_index
            ] == 1
            else "Normal"
        )
    )

    print(
        "Attack probability:",
        round(
            probabilities[
                sample_index
            ],
            4
        )
    )

    print(
        "\nLIME feature contributions:"
    )

    for (
        feature,
        contribution
    ) in lime_result.as_list():

        print(
            f"{feature}: "
            f"{contribution:+.4f}"
        )

    # -------------------------------------------------
    # Save image
    # -------------------------------------------------

    lime_result.as_pyplot_figure()

    plt.tight_layout()

    lime_png = os.path.join(
        RESULTS_FOLDER,
        (
            f"{DATASET}_"
            "lime_attack_explanation.png"
        )
    )

    plt.savefig(
        lime_png,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    # -------------------------------------------------
    # Save HTML
    # -------------------------------------------------

    lime_html = os.path.join(
        RESULTS_FOLDER,
        (
            f"{DATASET}_"
            "lime_attack_explanation.html"
        )
    )

    lime_result.save_to_file(
        lime_html
    )

    print(
        "LIME PNG saved:",
        lime_png
    )

    print(
        "LIME HTML saved:",
        lime_html
    )

    return lime_resultth}")
