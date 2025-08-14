from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,fbeta_score,roc_auc_score,roc_curve,precision_score,recall_score, precision_recall_curve, PrecisionRecallDisplay,average_precision_score
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
import numpy as np

def build_xgboost_pipeline(
    preprocessor,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
    **kwargs
):
    """
    Creates a pipeline with preprocessing and XGBoost classifier.

    Args:
        preprocessor (Transformer): A fitted or unfitted preprocessing transformer.
        n_estimators (int): Number of gradient boosted trees.
        learning_rate (float): Boosting learning rate.
        max_depth (int): Maximum tree depth for base learners.
        subsample (float): Subsample ratio of the training instances.
        colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        random_state (int): Random state for reproducibility.
        **kwargs: Additional XGBoost parameters.

    Returns:
        Pipeline: A scikit-learn pipeline with XGBoost classifier.
    """

    # Set up XGBoost classifier with common parameters
    xgb_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': random_state,
        'eval_metric': 'logloss',  # Suppress warnings
        **kwargs
    }

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ])
    return pipeline

def evaluate_xgboost_model(
    y_true: Union[np.ndarray, list],
    y_pred: Optional[Union[np.ndarray, list]] = None,
    y_proba: Optional[Union[np.ndarray, list]] = None,
    threshold: float = 0.10,
    beta: float = 1.0,
    average: str = 'binary',
    plot_roc: bool = False,
    plot_pr: bool = False,
    min_recall: float = None,
    model_name: str = "XGBoost"
) -> Dict[str, Union[float, str, dict]]:
    """
    Evaluates an XGBoost classification model using a custom threshold if probabilities are provided.
    Can also find the best threshold given a minimum recall requirement.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels (optional if y_proba provided).
        y_proba: Predicted probabilities for positive class.
        threshold: Classification threshold for converting probabilities to predictions.
        beta: Beta parameter for F-beta score calculation.
        average: Averaging strategy for multi-class ('binary', 'macro', 'micro', 'weighted').
        plot_roc: Whether to plot ROC curve.
        plot_pr: Whether to plot Precision-Recall curve.
        min_recall: Minimum recall threshold for finding optimal threshold.
        model_name: Name to display in plots.

    Returns:
        Dictionary containing evaluation metrics and optimal threshold info.
    """
    if y_proba is not None:
        y_pred = (np.array(y_proba) >= threshold).astype(int)

    results = {
        'Model': model_name,
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
                plt.plot(fpr, tpr, label=f"{model_name} ROC (AUC = {roc_auc:.3f})")
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"{model_name} ROC Curve")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

            # Precision-Recall Analysis
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            avg_precision = average_precision_score(y_true, y_proba)
            results['Average Precision'] = avg_precision

            # Find best threshold for minimum recall
            best_data = None
            if min_recall is not None:
                best_acc = -1
                valid_indices = recall[:-1] >= min_recall

                if np.any(valid_indices):
                    for p, r, t in zip(precision[:-1][valid_indices],
                                     recall[:-1][valid_indices],
                                     pr_thresholds[valid_indices]):
                        preds = (np.array(y_proba) >= t).astype(int)
                        acc = accuracy_score(y_true, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_data = {
                                "Best Threshold": t,
                                "Precision": p,
                                "Recall": r,
                                "Accuracy": acc,
                                "F1 Score": f1_score(y_true, preds),
                                f"F{beta} Score": fbeta_score(y_true, preds, beta=beta)
                            }

                results['Best Threshold for Min Recall'] = best_data if best_data else "No threshold found for min recall"

            if plot_pr:
                plt.figure(figsize=(6, 5))
                plt.plot(recall, precision, label=f"{model_name} PR (AP = {avg_precision:.3f})")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"{model_name} Precision-Recall Curve")
                plt.grid(True, alpha=0.3)

                # Mark the best threshold point
                if min_recall is not None and isinstance(best_data, dict):
                    plt.scatter(best_data["Recall"], best_data["Precision"],
                                color="red", s=100, zorder=5,
                                label=f"Best @ Recall≥{min_recall:.2f}")
                    plt.annotate(f"t={best_data['Best Threshold']:.3f}\nAcc={best_data['Accuracy']:.3f}",
                                 (best_data["Recall"], best_data["Precision"]),
                                 textcoords="offset points", xytext=(10, -15),
                                 fontsize=9, ha='left',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                plt.legend()
                plt.tight_layout()
                plt.show()

        except Exception as e:
            results['ROC AUC'] = f'Error: {str(e)}'

    return results

def compare_models(models_results: list, metrics: list = None):
    """
    Compare multiple model evaluation results.

    Args:
        models_results: List of dictionaries from evaluate_xgboost_model
        metrics: List of metrics to compare (default: common classification metrics)
    """
    if metrics is None:
        metrics = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

    import pandas as pd

    comparison_data = []
    for result in models_results:
        row = {}
        row['Model'] = result.get('Model', 'Unknown')
        row['Threshold'] = result.get('Threshold', 'N/A')
        for metric in metrics:
            row[metric] = result.get(metric, 'N/A')
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    return df
