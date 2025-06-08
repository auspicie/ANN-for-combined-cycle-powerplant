# src/evaluation.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained model and returns common regression metrics.

    Args:
        model: Trained scikit-learn model pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for printing.

    Returns:
        tuple: (dict of metrics, numpy.ndarray of predictions)
    """
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    return metrics, y_pred

def plot_predictions(y_test, y_pred, model_name="Model", save_path=None):
    """
    Plots actual vs. predicted values.

    Args:
        y_test (pd.Series): Actual test target values.
        y_pred (numpy.ndarray): Predicted values.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Directory to save the plot. If None, displays.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Line ($y=x$)')
    plt.xlabel("Actual Electrical Energy Output (MW)", fontsize=12)
    plt.ylabel("Predicted Electrical Energy Output (MW)", fontsize=12)
    plt.title(f"{model_name}: Actual vs. Predicted Values", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_actual_vs_predicted.png'))
        print(f"Plot saved to {os.path.join(save_path, f'{model_name}_actual_vs_predicted.png')}")
        plt.close()
    else:
        plt.show()

def plot_residuals(y_test, y_pred, model_name="Model", save_path=None):
    """
    Plots residuals (actual - predicted) against predicted values.

    Args:
        y_test (pd.Series): Actual test target values.
        y_pred (numpy.ndarray): Predicted values.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Directory to save the plot. If None, displays.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residuals')
    plt.xlabel("Predicted Electrical Energy Output (MW)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.title(f"{model_name}: Residual Plot", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{model_name}_residuals.png'))
        print(f"Plot saved to {os.path.join(save_path, f'{model_name}_residuals.png')}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, model_name="Model", save_path=None):
    """
    Plots feature importance for models that support it (e.g., tree-based).

    Args:
        model: Trained scikit-learn pipeline (should have a 'regressor' step).
        feature_names (list): List of feature names.
        model_name (str): Name of the model for the plot title.
        save_path (str, optional): Directory to save the plot. If None, displays.
    """
    # Check if the regressor step has feature_importances_ attribute
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        features_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        features_df = features_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(x='Importance', y='Feature', data=features_df)
        plt.title(f"{model_name}: Feature Importance", fontsize=14)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{model_name}_feature_importance.png'))
            print(f"Plot saved to {os.path.join(save_path, f'{model_name}_feature_importance.png')}")
            plt.close()
        else:
            plt.show()
    else:
        print(f"Feature importance not available for {model_name}.")

if __name__ == '__main__':
    # This module is typically run via main.py for proper evaluation
    print("This module is for evaluation functions and is typically run via main.py.")