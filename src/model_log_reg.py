from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,fbeta_score,roc_auc_score,roc_curve,precision_score,recall_score, precision_recall_curve, PrecisionRecallDisplay,average_precision_score
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
import numpy as np

def build_logreg_pipeline(preprocessor, max_iter=1000, random_state=42):
    """
    Creates a pipeline with preprocessing and logistic regression classifier.

    Args:
        preprocessor (Transformer): A fitted or unfitted preprocessing transformer.
        max_iter (int): Maximum number of iterations for LogisticRegression.
        random_state (int): Random state for reproducibility.

    Returns:
        Pipeline: A scikit-learn pipeline.
    """
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])
    return pipeline

def evaluate_model(
    y_true: Union[np.ndarray, list],
    y_pred: Optional[Union[np.ndarray, list]] = None,
    y_proba: Optional[Union[np.ndarray, list]] = None,
    threshold: float = 0.10,
    beta: float = 1.0,
    average: str = 'binary',
    plot_roc: bool = False,
    plot_pr: bool = False,
    min_recall: float = None
) -> Dict[str, Union[float, str, dict]]:
    """
    Evaluates a classification model using a custom threshold if probabilities are provided.
    Can also find the best threshold given a minimum recall requirement.
    """
    if y_proba is not None:
        y_pred = (np.array(y_proba) >= threshold).astype(int)

    results = {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average=average),
        f'F{beta}-Score': fbeta_score(y_true, y_pred, beta=beta, average=average)
    }

    if y_proba is not None:
        try:
            # ROC AUC
            roc_auc = roc_auc_score(y_true, y_proba)
            results['ROC AUC'] = roc_auc

            if plot_roc:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            # Precision-Recall
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            results['Average Precision'] = avg_precision

            # Find best threshold for minimum recall
            best_data = None
            if min_recall is not None:
                best_acc = -1
                for p, r, t in zip(precision[:-1], recall[:-1], pr_thresholds):
                    if r >= min_recall:
                        preds = (np.array(y_proba) >= t).astype(int)
                        acc = accuracy_score(y_true, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_data = {
                                "Best Threshold": t,
                                "Precision": p,
                                "Recall": r,
                                "Accuracy": acc
                            }
                results['Best Threshold for Min Recall'] = best_data

            if plot_pr:
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.2f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.grid(True)

                # Mark the best threshold point
                if min_recall is not None and best_data is not None:
                    plt.scatter(best_data["Recall"], best_data["Precision"],
                                color="red", s=80, zorder=5, label="Best ≥ Min Recall")
                    plt.annotate(f"t={best_data['Best Threshold']:.2f}\nAcc={best_data['Accuracy']:.2f}",
                                 (best_data["Recall"], best_data["Precision"]),
                                 textcoords="offset points", xytext=(5, -10), fontsize=8)

                plt.legend()
                plt.tight_layout()
                plt.show()

        except Exception as e:
            results['ROC AUC'] = f'Error: {str(e)}'

    return results
