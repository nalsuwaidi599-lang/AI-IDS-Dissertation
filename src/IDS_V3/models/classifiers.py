import tensorflow as tf

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.neural_network import (
    MLPClassifier
)

from utils.reproducibility import (
    set_seed
)

from config import QUICK_MODE


def build_logistic_regression(seed):

    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        class_weight='balanced'
    )


def build_random_forest(seed):

    return RandomForestClassifier(
        n_estimators=(
            200
            if not QUICK_MODE
            else 50
        ),
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced'
    )


def build_ffnn(seed):

    return MLPClassifier(
        hidden_layer_sizes=(
            128,
            64,
            32
        ),
        activation='relu',
        max_iter=300,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15
    )


def build_cnn(
    n_features,
    seed
):

    set_seed(seed)

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                n_features,
                1
            )
        ),

        tf.keras.layers.Conv1D(
            32,
            kernel_size=3,
            padding='same',
            activation='relu'
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv1D(
            16,
            kernel_size=3,
            padding='same',
            activation='relu'
        ),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(
            32,
            activation='relu'
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm(
    n_features,
    seed
):

    set_seed(seed)

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(
                n_features,
                1
            )
        ),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dropout(
            0.30
        ),

        tf.keras.layers.Dense(
            16,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
