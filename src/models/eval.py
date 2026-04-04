from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def plot_roc(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def plot_roc_curve_and_auc(y, y_proba, plot_roc_curve=False):
    auc = roc_auc_score(y, y_proba)

    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(y, y_proba)
        plot_roc(fpr, tpr, auc)
    
    return auc