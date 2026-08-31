import os
import sys
import json
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import shap

from config import (
    DATASET,
    RESULTS_FOLDER,
    QUICK_MODE,
    SEEDS
)

from data.loader import (
    load_dataset
)

from data.preprocessing import (
    preprocess_dataset
)

from features.shap_selection import (
    select_shap_features
)

from models.classifiers import (
    build_logistic_regression,
    build_random_forest,
    build_ffnn,
    build_cnn,
    build_lstm
)

from evaluation.metrics import (
    evaluate_sklearn,
    evaluate_keras
)

from evaluation.plots import (
    plot_roc_curves
)

from evaluation.partition_auc import (
    run_partition_auc_analysis
)

from explainability.lime_explainer import (
    generate_lime_explanation
)

from utils.reproducibility import (
    set_seed
)


warnings.filterwarnings(
    'ignore'
)

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


def main():

    # =================================================
    # ENVIRONMENT
    # =================================================

    print(
        'Dataset:',
        DATASET
    )

    print(
        'Mode:',
        (
            'QUICK CHECK ONLY'
            if QUICK_MODE
            else 'FINAL EXPERIMENT'
        )
    )

    print(
        'Python:',
        sys.version.split()[0]
    )

    print(
        'NumPy:',
        np.__version__
    )

    print(
        'Pandas:',
        pd.__version__
    )

    print(
        'Scikit-learn:',
        sklearn.__version__
    )

    print(
        'TensorFlow:',
        tf.__version__
    )

    print(
        'SHAP:',
        shap.__version__
    )

    print(
        'Platform:',
        platform.platform()
    )

    print(
        'CPU:',
        (
            platform.processor()
            or
            'Reported by runtime'
        )
    )

    print(
        'GPU:',
        tf.config.list_physical_devices(
            'GPU'
        )
    )


    # =================================================
    # 1. LOAD DATASET
    # =================================================

    df, source_files = (
        load_dataset(
            DATASET
        )
    )

    raw_rows, raw_columns = (
        df.shape
    )

    print(
        f'Raw records: '
        f'{raw_rows:,}; '
        f'raw columns: '
        f'{raw_columns}'
    )

    print(
        'Source files:',
        len(source_files)
    )

    print(
        df[
            'binary_label'
        ]
        .value_counts()
        .rename(
            index={
                0: 'Normal',
                1: 'Attack'
            }
        )
    )


    # =================================================
    # 2. PREPROCESS
    # =================================================

    prep = (
        preprocess_dataset(
            df
        )
    )

    X_train_all = (
        prep[
            'X_train_all'
        ]
    )

    X_val_all = (
        prep[
            'X_val_all'
        ]
    )

    X_test_all = (
        prep[
            'X_test_all'
        ]
    )

    y_train = (
        prep[
            'y_train'
        ]
    )

    y_val = (
        prep[
            'y_val'
        ]
    )

    y_test = (
        prep[
            'y_test'
        ]
    )

    all_feature_names = (
        prep[
            'all_feature_names'
        ]
    )


    # =================================================
    # 3. SHAP FEATURE REFINEMENT
    # =================================================

    shap_results = (
        select_shap_features(

            X_train_all,
            X_val_all,
            X_test_all,

            y_train,

            all_feature_names
        )
    )

    X_train_refined = (
        shap_results[
            'X_train_refined'
        ]
    )

    X_val_refined = (
        shap_results[
            'X_val_refined'
        ]
    )

    X_test_refined = (
        shap_results[
            'X_test_refined'
        ]
    )

    selected_features = (
        shap_results[
            'selected_features'
        ]
    )

    reduction_percentage = (
        shap_results[
            'reduction_percentage'
        ]
    )


    # =================================================
    # 4. TRAIN + EVALUATE MODELS
    # =================================================

    all_rows = []

    roc_store = {}

    experiment_seeds = (
        [42]
        if QUICK_MODE
        else SEEDS
    )

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
            Xtr,
            Xv,
            Xte
        )
    ) in feature_sets.items():

        print(
            f'\n===== '
            f'{feature_set_name}: '
            f'{Xtr.shape[1]} '
            f'features ====='
        )

        for seed in experiment_seeds:

            print(
                'Seed:',
                seed
            )

            set_seed(seed)

            # -----------------------------------------
            # Classical models + FFNN
            # -----------------------------------------

            sklearn_models = {

                'Logistic Regression':
                    build_logistic_regression(
                        seed
                    ),

                'Random Forest':
                    build_random_forest(
                        seed
                    ),

                'FFNN':
                    build_ffnn(
                        seed
                    )
            }

            for (
                model_name,
                model
            ) in sklearn_models.items():

                (
                    metrics,
                    train_time,
                    infer_time,
                    probabilities

                ) = evaluate_sklearn(

                    model,

                    Xtr,
                    y_train,

                    Xte,
                    y_test
                )

                row = {

                    'Dataset':
                        DATASET.upper(),

                    'FeatureSet':
                        feature_set_name,

                    'FeatureCount':
                        Xtr.shape[1],

                    'Seed':
                        seed,

                    'Model':
                        model_name,

                    'TrainingSeconds':
                        train_time,

                    'InferenceSeconds':
                        infer_time,

                    **metrics
                }

                all_rows.append(
                    row
                )

                if (
                    seed
                    ==
                    experiment_seeds[0]
                ):

                    roc_store[
                        (
                            feature_set_name,
                            model_name
                        )
                    ] = probabilities


            # -----------------------------------------
            # CNN + LSTM
            # -----------------------------------------

            deep_models = [

                (
                    'CNN',
                    build_cnn
                ),

                (
                    'LSTM',
                    build_lstm
                )
            ]

            for (
                model_name,
                builder
            ) in deep_models:

                tf.keras.backend.clear_session()

                model = builder(
                    Xtr.shape[1],
                    seed
                )

                (
                    metrics,
                    train_time,
                    infer_time,
                    probabilities

                ) = evaluate_keras(

                    model,

                    Xtr,
                    y_train,

                    Xv,
                    y_val,

                    Xte,
                    y_test
                )

                row = {

                    'Dataset':
                        DATASET.upper(),

                    'FeatureSet':
                        feature_set_name,

                    'FeatureCount':
                        Xtr.shape[1],

                    'Seed':
                        seed,

                    'Model':
                        model_name,

                    'TrainingSeconds':
                        train_time,

                    'InferenceSeconds':
                        infer_time,

                    **metrics
                }

                all_rows.append(
                    row
                )

                if (
                    seed
                    ==
                    experiment_seeds[0]
                ):

                    roc_store[
                        (
                            feature_set_name,
                            model_name
                        )
                    ] = probabilities


    # =================================================
    # 5. SAVE INDIVIDUAL RUN RESULTS
    # =================================================

    results_runs = pd.DataFrame(
        all_rows
    )

    print(
        '\nINDIVIDUAL RUN RESULTS'
    )

    print(
        results_runs.round(4)
    )

    results_runs.to_csv(

        os.path.join(
            RESULTS_FOLDER,
            (
                f'{DATASET}_'
                'individual_runs.csv'
            )
        ),

        index=False
    )

    print(
        'Individual-run results saved.'
    )


    # =================================================
    # 6. STATISTICAL SUMMARY
    # =================================================

    metric_columns = [

        'Accuracy',
        'Precision',
        'Recall',
        'F1',
        'F2',
        'FPR',
        'FNR',
        'AUC',
        'TrainingSeconds',
        'InferenceSeconds'
    ]

    group_columns = [

        'Dataset',
        'FeatureSet',
        'FeatureCount',
        'Model'
    ]

    summary_parts = []

    for (
        keys,
        group
    ) in results_runs.groupby(
        group_columns
    ):

        row = dict(
            zip(
                group_columns,
                keys
            )
        )

        n = len(group)

        row[
            'Runs'
        ] = n

        for metric in metric_columns:

            mean = (
                group[metric]
                .mean()
            )

            std = (
                group[metric]
                .std(ddof=1)
                if n > 1
                else 0.0
            )

            ci = (
                1.96
                *
                std
                /
                np.sqrt(n)
                if n > 1
                else 0.0
            )

            row[
                f'{metric}_Mean'
            ] = mean

            row[
                f'{metric}_SD'
            ] = std

            row[
                f'{metric}_CI95_Lower'
            ] = (
                mean - ci
            )

            row[
                f'{metric}_CI95_Upper'
            ] = (
                mean + ci
            )

        summary_parts.append(
            row
        )

    summary = pd.DataFrame(
        summary_parts
    )

    main_columns = (
        group_columns
        +
        ['Runs']
        +
        [
            f'{m}_Mean'
            for m
            in metric_columns
        ]
    )

    print(
        '\nSTATISTICAL SUMMARY'
    )

    print(
        summary[
            main_columns
        ].round(4)
    )

    summary.to_csv(

        os.path.join(
            RESULTS_FOLDER,
            (
                f'{DATASET}_'
                'statistical_summary.csv'
            )
        ),

        index=False
    )


    # =================================================
    # 7. EGFRF ABLATION COMPARISON
    # =================================================

    means = (
        summary
        .set_index(
            [
                'FeatureSet',
                'Model'
            ]
        )
    )

    ablation_rows = []

    for model_name in (
        summary[
            'Model'
        ].unique()
    ):

        full = means.loc[
            (
                'All features',
                model_name
            )
        ]

        refined = means.loc[
            (
                'SHAP refined',
                model_name
            )
        ]

        ablation_rows.append({

            'Model':
                model_name,

            'FeaturesBefore':
                int(
                    full[
                        'FeatureCount'
                    ]
                ),

            'FeaturesAfter':
                int(
                    refined[
                        'FeatureCount'
                    ]
                ),

            'ReductionPercent':
                100
                *
                (
                    1
                    -
                    refined[
                        'FeatureCount'
                    ]
                    /
                    full[
                        'FeatureCount'
                    ]
                ),

            'AccuracyChange':
                refined[
                    'Accuracy_Mean'
                ]
                -
                full[
                    'Accuracy_Mean'
                ],

            'F1Change':
                refined[
                    'F1_Mean'
                ]
                -
                full[
                    'F1_Mean'
                ],

            'F2Change':
                refined[
                    'F2_Mean'
                ]
                -
                full[
                    'F2_Mean'
                ],

            'FPRChange':
                refined[
                    'FPR_Mean'
                ]
                -
                full[
                    'FPR_Mean'
                ],

            'FNRChange':
                refined[
                    'FNR_Mean'
                ]
                -
                full[
                    'FNR_Mean'
                ],

            'AUCChange':
                refined[
                    'AUC_Mean'
                ]
                -
                full[
                    'AUC_Mean'
                ],

            'TrainingTimeChangeSeconds':
                refined[
                    'TrainingSeconds_Mean'
                ]
                -
                full[
                    'TrainingSeconds_Mean'
                ],

            'InferenceTimeChangeSeconds':
                refined[
                    'InferenceSeconds_Mean'
                ]
                -
                full[
                    'InferenceSeconds_Mean'
                ]
        })

    ablation = pd.DataFrame(
        ablation_rows
    )

    print(
        '\nEGFRF ABLATION'
    )

    print(
        ablation.round(4)
    )

    ablation.to_csv(

        os.path.join(
            RESULTS_FOLDER,
            (
                f'{DATASET}_'
                'egfrf_ablation.csv'
            )
        ),

        index=False
    )


    # =================================================
    # 8. ROC CURVES
    # =================================================

    plot_roc_curves(
        roc_store,
        y_test
    )


    # =================================================
    # 9. REPRODUCIBILITY RECORD
    # =================================================

    reproducibility = {

        'dataset':
            DATASET,

        'source_files': [
            os.path.basename(f)
            for f
            in source_files
        ],

        'raw_rows':
            int(raw_rows),

        'rows_after_exact_duplicate_removal_before_sampling':
            int(
                prep[
                    'before_duplicates'
                ]
                -
                prep[
                    'duplicates_removed'
                ]
            ),

        'exact_duplicates_removed':
            int(
                prep[
                    'duplicates_removed'
                ]
            ),

        'duplicate_numeric_feature_vectors_removed':
            int(
                prep[
                    'feature_duplicates_removed'
                ]
            ),

        'records_used':
            int(
                len(
                    prep[
                        'df'
                    ]
                )
            ),

        'train_records':
            int(
                len(
                    prep[
                        'X_train_raw'
                    ]
                )
            ),

        'validation_records':
            int(
                len(
                    prep[
                        'X_val_raw'
                    ]
                )
            ),

        'test_records':
            int(
                len(
                    prep[
                        'X_test_raw'
                    ]
                )
            ),

        'train_test_identical_feature_overlap':
            int(
                prep[
                    'overlap_count'
                ]
            ),

        'all_feature_count':
            int(
                len(
                    all_feature_names
                )
            ),

        'refined_feature_count':
            int(
                len(
                    selected_features
                )
            ),

        'feature_reduction_percent':
            float(
                reduction_percentage
            ),

        'seeds':
            experiment_seeds,

        'python':
            sys.version,

        'numpy':
            np.__version__,

        'pandas':
            pd.__version__,

        'scikit_learn':
            sklearn.__version__,

        'tensorflow':
            tf.__version__,

        'shap':
            shap.__version__,

        'platform':
            platform.platform(),

        'gpu': [
            str(x)
            for x
            in tf.config.list_physical_devices(
                'GPU'
            )
        ]
    }

    with open(

        os.path.join(
            RESULTS_FOLDER,
            (
                f'{DATASET}_'
                'reproducibility.json'
            )
        ),

        'w'

    ) as file:

        json.dump(
            reproducibility,
            file,
            indent=2
        )

    print(
        '\nREPRODUCIBILITY RECORD'
    )

    print(
        json.dumps(
            reproducibility,
            indent=2
        )
    )


    # =================================================
    # 10. SEED-42 TRAIN / VALIDATION / TEST AUC CHECK
    # =================================================

    partition_auc_results = (
        run_partition_auc_analysis(

            X_train_all,
            X_val_all,
            X_test_all,

            X_train_refined,
            X_val_refined,
            X_test_refined,

            y_train,
            y_val,
            y_test
        )
    )


    # =================================================
    # 11. LIME LOCAL EXPLANATION
    # =================================================

    generate_lime_explanation(

        X_train_refined,
        X_test_refined,

        y_train,
        y_test,

        selected_features
    )


    # =================================================
    # FINISHED
    # =================================================

    print(
        '\n========================================'
    )

    print(
        'EXPERIMENT COMPLETED'
    )

    print(
        '========================================'
    )

    print(
        'All results are saved in:',
        RESULTS_FOLDER
    )


if __name__ == '__main__':
    main()
