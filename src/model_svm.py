from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,fbeta_score,roc_auc_score,roc_curve,precision_score,recall_score, precision_recall_curve, PrecisionRecallDisplay,average_precision_score
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict
import numpy as np

def build_sgd_pipeline(
    preprocessor,
    loss='log_loss',
    penalty='l2',
    alpha=0.0001,
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    tol=1e-3,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    random_state=42,
    **kwargs
):
    """
    Creates a pipeline with preprocessing and SGD classifier.

    Args:
        preprocessor (Transformer): A fitted or unfitted preprocessing transformer.
        loss (str): Loss function ('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron').
        penalty (str): Regularization term ('l1', 'l2', 'elasticnet', None).
        alpha (float): Constant that multiplies the regularization term.
        learning_rate (str): Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive').
        eta0 (float): Initial learning rate for 'constant', 'invscaling' or 'adaptive' schedules.
        max_iter (int): Maximum number of passes over the training data.
        tol (float): Stopping criterion tolerance.
        early_stopping (bool): Whether to use early stopping to terminate training.
        validation_fraction (float): Proportion of training data to set aside for early stopping.
        n_iter_no_change (int): Number of iterations with no improvement to wait before stopping.
        class_weight (dict, 'balanced' or None): Weights associated with classes.
        random_state (int): Random state for reproducibility.
        **kwargs: Additional SGD parameters.

    Returns:
        Pipeline: A scikit-learn pipeline with SGD classifier.
    """

    # Set up SGD classifier with common parameters
    sgd_params = {
        'loss': loss,
        'penalty': penalty,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'eta0': eta0,
        'max_iter': max_iter,
        'tol': tol,
        'early_stopping': early_stopping,
        'validation_fraction': validation_fraction,
        'n_iter_no_change': n_iter_no_change,
        'class_weight': class_weight,
        'random_state': random_state,
        **kwargs
    }

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', SGDClassifier(**sgd_params))
    ])
    return pipeline

def evaluate_sgd_model(
    y_true: Union[np.ndarray, list],
    y_pred: Optional[Union[np.ndarray, list]] = None,
    y_proba: Optional[Union[np.ndarray, list]] = None,
    threshold: float = 0.10,
    beta: float = 1.0,
    average: str = 'binary',
    plot_roc: bool = False,
    plot_pr: bool = False,
    min_recall: float = None,
    model_name: str = "SGD"
) -> Dict[str, Union[float, str, dict]]:
    """
    Evaluates an SGD classification model using a custom threshold if probabilities are provided.
    Can also find the best threshold given a minimum recall requirement.

    Note: SGD with 'hinge' loss doesn't provide predict_proba. Use 'log_loss' or 'modified_huber'
    for probability estimates, or use decision_function scores instead.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels (optional if y_proba provided).
        y_proba: Predicted probabilities for positive class (or decision function scores).
        threshold: Classification threshold for converting probabilities/scores to predictions.
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
        models_results: List of dictionaries from evaluate_sgd_model or evaluate_xgboost_model
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

# Example usage:
"""
# Basic usage
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Create preprocessor
preprocessor = StandardScaler()

# Build SGD pipeline
sgd_pipeline = build_sgd_pipeline(
    preprocessor,
    loss='log_loss',  # For probability estimates
    penalty='l2',
    alpha=0.001,
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=2000,
    early_stopping=True,
    class_weight='balanced'
)

# Fit the model
sgd_pipeline.fit(X_train, y_train)

# Get predictions and probabilities
y_pred = sgd_pipeline.predict(X_test)
y_proba = sgd_pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class

# Evaluate the model
results = evaluate_sgd_model(
    y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    threshold=0.3,
    min_recall=0.8,
    plot_roc=True,
    plot_pr=True,
    model_name="SGD Classifier"
)

print(results)
"""
