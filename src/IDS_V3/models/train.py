import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from models import registry


def make_ffnn(layers=(128, 64, 32), max_iter=300, random_state=42, **kw):
    return MLPClassifier(
        hidden_layer_sizes=layers, activation='relu',
        max_iter=max_iter, random_state=random_state,
        early_stopping=True, validation_fraction=0.15,
        alpha=0.001,  # l2 reg
    )

def make_cnn(n_estimators=100, max_depth=4, lr=0.05, random_state=42, **kw):
    # tuned down from original to reduce overfitting:
    # max_depth 5->4, lr 0.1->0.05, added subsample + min_samples_leaf
    return GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, subsample=0.8,
        min_samples_leaf=10, random_state=random_state,
    )

def make_lstm(n_estimators=150, max_depth=8, random_state=42, **kw):
    return ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=5, random_state=random_state, n_jobs=-1,
    )

# register them
registry.register('ffnn', make_ffnn)
registry.register('cnn', make_cnn)
registry.register('lstm', make_lstm)


def snort_predict(X_test, X_train, y_train, feat_idx):
    """simulates snort by thresholding on the most informative feature"""
    thresh = np.median(X_train[y_train == 1, feat_idx])
    preds = (X_test[:, feat_idx] > thresh).astype(int)
    scores = X_test[:, feat_idx]
    return preds, scores


def train_model(name, X_train, y_train, **kwargs):
    print(f"  training {name}...")
    model = registry.get(name, **kwargs)
    model.fit(X_train, y_train)

    # quick cv check to catch overfitting
    cv = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1)
    train_acc = model.score(X_train, y_train)
    gap = train_acc - cv.mean()
    print(f"    train: {train_acc:.4f}, cv f1: {cv.mean():.4f} (+/-{cv.std():.4f})")
    if gap > 0.05:
        print(f"    ** possible overfitting (gap={gap:.4f})")

    return model


def train_all(X_train, y_train, feature_names, mi_ranking, cfg):
    """train all four models, return dict"""
    models = {}

    # snort - just need the feature index
    top_feat = mi_ranking.index[0]
    snort_idx = feature_names.index(top_feat) if top_feat in feature_names else 0
    models['Snort'] = ('snort', snort_idx)
    print(f"  snort using feature: '{feature_names[snort_idx]}'")

    models['FFNN'] = (train_model('ffnn', X_train, y_train,
                                   layers=cfg.FFNN_LAYERS, max_iter=cfg.FFNN_MAX_ITER,
                                   random_state=cfg.RANDOM_STATE), None)

    models['CNN'] = (train_model('cnn', X_train, y_train,
                                  n_estimators=cfg.CNN_ESTIMATORS,
                                  max_depth=cfg.CNN_MAX_DEPTH,
                                  lr=cfg.CNN_LEARNING_RATE,
                                  random_state=cfg.RANDOM_STATE), None)

    models['LSTM'] = (train_model('lstm', X_train, y_train,
                                   n_estimators=cfg.LSTM_ESTIMATORS,
                                   max_depth=cfg.LSTM_MAX_DEPTH,
                                   random_state=cfg.RANDOM_STATE), None)

    return models
