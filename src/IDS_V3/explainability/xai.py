import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_shap(model, X_test, feat_names, path, n=200):
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping")
        return None

    print("  running SHAP...")
    exp = shap.TreeExplainer(model)
    sv = exp.shap_values(X_test[:n])
    shap.summary_plot(sv, X_test[:n], feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")
    return sv


def run_lime(model, X_train, X_test, y_test, feat_names, path):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("  lime not installed, skipping")
        return None

    print("  running LIME...")
    explainer = LimeTabularExplainer(
        X_train, feature_names=feat_names,
        class_names=['Normal', 'Attack'], mode='classification'
    )
    # pick first attack sample
    idx = np.where(y_test == 1)[0][0]
    exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)
    exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")

    print("  LIME feature weights:")
    for feat, w in exp.as_list():
        print(f"    {feat}: {w:+.4f}")
    return exp


def compare_shap(model_std, model_hyb, Xte_std, Xte_hyb, fn_std, fn_hyb, path, n=200):
    """side by side SHAP to show how hybrid changes what the model looks at"""
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping comparison")
        return

    print("  comparing SHAP between methods...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plt.sca(ax1)
    sv1 = shap.TreeExplainer(model_std).shap_values(Xte_std[:n])
    shap.summary_plot(sv1, Xte_std[:n], feature_names=fn_std, show=False, plot_size=None)
    ax1.set_title("Standard")

    plt.sca(ax2)
    sv2 = shap.TreeExplainer(model_hyb).shap_values(Xte_hyb[:n])
    shap.summary_plot(sv2, Xte_hyb[:n], feature_names=fn_hyb, show=False, plot_size=None)
    ax2.set_title("Hybrid (ours)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")
