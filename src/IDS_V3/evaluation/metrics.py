import time

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score
)

from config import (
    QUICK_MODE,
    EPOCHS,
    BATCH_SIZE
)


def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    return {

        'Accuracy':
            accuracy_score(
                y_true,
                predictions
            ),

        'Precision':
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        'Recall':
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        'F1':
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        'F2':
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                zero_division=0
            ),

        'FPR':
            (
                fp / (fp + tn)
                if fp + tn
                else 0.0
            ),

        'FNR':
            (
                fn / (fn + tp)
                if fn + tp
                else 0.0
            ),

        'AUC':
            roc_auc_score(
                y_true,
                probabilities
            ),

        'TN':
            int(tn),

        'FP':
            int(fp),

        'FN':
            int(fn),

        'TP':
            int(tp)
    }


def evaluate_sklearn(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    # -------------------------------
    # Training time
    # -------------------------------

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_seconds = (
        time.perf_counter()
        - start
    )

    # -------------------------------
    # Inference time
    # -------------------------------

    start = time.perf_counter()

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    inference_seconds = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities
    )

    return (
        metrics,
        train_seconds,
        inference_seconds,
        probabilities
    )


def evaluate_keras(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    # CNN/LSTM expect:
    # samples × features × 1

    Xtr = (
        X_train[
            ...,
            np.newaxis
        ]
    )

    Xv = (
        X_val[
            ...,
            np.newaxis
        ]
    )

    Xte = (
        X_test[
            ...,
            np.newaxis
        ]
    )

    callback = (
        tf.keras.callbacks
        .EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    )

    # -------------------------------
    # Training time
    # -------------------------------

    start = time.perf_counter()

    model.fit(
        Xtr,
        np.asarray(y_train),

        validation_data=(
            Xv,
            np.asarray(y_val)
        ),

        epochs=(
            5
            if QUICK_MODE
            else EPOCHS
        ),

        batch_size=BATCH_SIZE,

        callbacks=[
            callback
        ],

        verbose=0
    )

    train_seconds = (
        time.perf_counter()
        - start
    )

    # -------------------------------
    # Inference
    # -------------------------------

    start = time.perf_counter()

    probabilities = (
        model.predict(
            Xte,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    inference_seconds = (
        time.perf_counter()
        - start
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities
    )

    return (
        metrics,
        train_seconds,
        inference_seconds,
        probabilities
    )
