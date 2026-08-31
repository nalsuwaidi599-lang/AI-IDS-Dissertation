import os

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

from config import (
    DATASET,
    RESULTS_FOLDER
)


def plot_roc_curves(
    roc_store,
    y_test
):

    plt.figure(
        figsize=(10, 7)
    )

    for (
        feature_set,
        model_name
    ), probabilities in roc_store.items():

        fpr_values, (
            tpr_values
        ), _ = roc_curve(
            y_test,
            probabilities
        )

        auc_value = (
            roc_auc_score(
                y_test,
                probabilities
            )
        )

        style = (
            '-'
            if feature_set
            == 'SHAP refined'
            else '--'
        )

        plt.plot(
            fpr_values,
            tpr_values,
            linestyle=style,
            label=(
                f'{model_name} | '
                f'{feature_set} '
                f'(AUC={auc_value:.3f})'
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        'k:',
        label='Random'
    )

    plt.xlabel(
        'False Positive Rate'
    )

    plt.ylabel(
        'True Positive Rate'
    )

    plt.title(
        f'Corrected ROC comparison: '
        f'{DATASET.upper()}'
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    roc_path = os.path.join(
        RESULTS_FOLDER,
        f'{DATASET}_corrected_roc.png'
    )

    plt.savefig(
        roc_path,
        dpi=200
    )

    plt.show()

    return roc_path
