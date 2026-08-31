import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from config import (
    DATASET,
    RESULTS_FOLDER,
    EPOCHS,
    BATCH_SIZE
)

from models.classifiers import (
    build_logistic_regression,
    build_random_forest,
    build_ffnn,
    build_cnn,
    build_lstm
)

from utils.reproducibility import set_seed


SEED = 42


def get_sklearn_auc_scores(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    """
    Train one sklearn model and calculate ROC-AUC
    separately on training, validation and test data.
    """

    model.fit(
        X_train,
        y_train
    )

    train_probabilities = (
        model.predict_proba(
            X_train
        )[:, 1]
    )

    validation_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    training_auc = roc_auc_score(
        y_train,
        train_probabilities
    )

    validation_auc = roc_auc_score(
        y_val,
        validation_probabilities
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    return (
        training_auc,
        validation_auc,
        test_auc
    )


def get_keras_auc_scores(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    """
    Train one TensorFlow model and calculate ROC-AUC
    separately on training, validation and test data.
    """

    Xtr = X_train[..., np.newaxis]
    Xv = X_val[..., np.newaxis]
    Xte = X_test[..., np.newaxis]

    callback = (
        tf.keras.callbacks
        .EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    )

    model.fit(
        Xtr,
        np.asarray(y_train),

        validation_data=(
            Xv,
            np.asarray(y_val)
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            callback
        ],

        verbose=0
    )

    train_probabilities = (
        model.predict(
            Xtr,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    validation_probabilities = (
        model.predict(
            Xv,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    test_probabilities = (
        model.predict(
            Xte,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    training_auc = roc_auc_score(
        y_train,
        train_probabilities
    )

    validation_auc = roc_auc_score(
        y_val,
        validation_probabilities
    )

    test_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    return (
        training_auc,
        validation_auc,
        test_auc
    )


def run_partition_auc_analysis(
    X_train_all,
    X_val_all,
    X_test_all,

    X_train_refined,
    X_val_refined,
    X_test_refined,

    y_train,
    y_val,
    y_test
):

    print(
        "\n========================================"
    )

    print(
        "SEED 42 TRAIN / VALIDATION / TEST AUC"
    )

    print(
        "========================================"
    )

    results = []

    feature_sets = {

        'All features': (
            X_train_all,
            X_val_all,
            X_test_all
        ),

        'SHAP refined': (
            X_train_refined,
            X_val_refined,
            X_test_refined
        )
    }

    for (
        feature_set_name,
        (
            X_train,
            X_val,
            X_test
        )
    ) in feature_sets.items():

        print(
            f"\nFeature set: "
            f"{feature_set_name}"
        )

        print(
            f"Feature count: "
            f"{X_train.shape[1]}"
        )

        # =============================================
        # LOGISTIC REGRESSION
        # =============================================

        set_seed(SEED)

        model = (
            build_logistic_regression(
                SEED
            )
        )

        (
            training_auc,
            validation_auc,
            test_auc

        ) = get_sklearn_auc_scores(

            model,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        results.append({

            'Dataset':
                DATASET.upper(),

            'Feature set':
                feature_set_name,

            'Feature count':
                X_train.shape[1],

            'Seed':
                SEED,

            'Model':
                'Logistic Regression',

            'Training AUC':
                training_auc,

            'Validation AUC':
                validation_auc,

            'Test AUC':
                test_auc,

            'Train–Test Gap':
                training_auc
                -
                test_auc
        })

        # =============================================
        # RANDOM FOREST
        # =============================================

        set_seed(SEED)

        model = (
            build_random_forest(
                SEED
            )
        )

        (
            training_auc,
            validation_auc,
            test_auc

        ) = get_sklearn_auc_scores(

            model,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        results.append({

            'Dataset':
                DATASET.upper(),

            'Feature set':
                feature_set_name,

            'Feature count':
                X_train.shape[1],

            'Seed':
                SEED,

            'Model':
                'Random Forest',

            'Training AUC':
                training_auc,

            'Validation AUC':
                validation_auc,

            'Test AUC':
                test_auc,

            'Train–Test Gap':
                training_auc
                -
                test_auc
        })

        # =============================================
        # FFNN
        # =============================================

        set_seed(SEED)

        model = (
            build_ffnn(
                SEED
            )
        )

        (
            training_auc,
            validation_auc,
            test_auc

        ) = get_sklearn_auc_scores(

            model,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        results.append({

            'Dataset':
                DATASET.upper(),

            'Feature set':
                feature_set_name,

            'Feature count':
                X_train.shape[1],

            'Seed':
                SEED,

            'Model':
                'FFNN',

            'Training AUC':
                training_auc,

            'Validation AUC':
                validation_auc,

            'Test AUC':
                test_auc,

            'Train–Test Gap':
                training_auc
                -
                test_auc
        })

        # =============================================
        # CNN
        # =============================================

        tf.keras.backend.clear_session()

        set_seed(SEED)

        model = (
            build_cnn(
                X_train.shape[1],
                SEED
            )
        )

        (
            training_auc,
            validation_auc,
            test_auc

        ) = get_keras_auc_scores(

            model,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        results.append({

            'Dataset':
                DATASET.upper(),

            'Feature set':
                feature_set_name,

            'Feature count':
                X_train.shape[1],

            'Seed':
                SEED,

            'Model':
                'CNN',

            'Training AUC':
                training_auc,

            'Validation AUC':
                validation_auc,

            'Test AUC':
                test_auc,

            'Train–Test Gap':
                training_auc
                -
                test_auc
        })

        # =============================================
        # LSTM
        # =============================================

        tf.keras.backend.clear_session()

        set_seed(SEED)

        model = (
            build_lstm(
                X_train.shape[1],
                SEED
            )
        )

        (
            training_auc,
            validation_auc,
            test_auc

        ) = get_keras_auc_scores(

            model,

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test
        )

        results.append({

            'Dataset':
                DATASET.upper(),

            'Feature set':
                feature_set_name,

            'Feature count':
                X_train.shape[1],

            'Seed':
                SEED,

            'Model':
                'LSTM',

            'Training AUC':
                training_auc,

            'Validation AUC':
                validation_auc,

            'Test AUC':
                test_auc,

            'Train–Test Gap':
                training_auc
                -
                test_auc
        })

    # =============================================
    # CREATE RESULTS TABLE
    # =============================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nSEED 42 PARTITION AUC RESULTS"
    )

    print(
        results_df.round(6)
    )

    # =============================================
    # SAVE CSV
    # =============================================

    output_path = os.path.join(
        RESULTS_FOLDER,
        (
            f"{DATASET}_"
            "train_validation_test_auc_seed42.csv"
        )
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nPartition AUC results saved to:"
    )

    print(
        output_path
    )

    return results_df
