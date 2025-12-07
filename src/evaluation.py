import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate regression performance metrics
    
    Parameters:
    -----------
    y_true : array
        Actual values
    y_pred : array
        Predicted values
    model_name : str
        Name for printing
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
        
    Why these metrics?
    - MAE (Mean Absolute Error): Average error in same units as target
      * Easy to interpret: "Off by X points on average"
    - RMSE (Root Mean Squared Error): Penalizes large errors more
      * Useful when big mistakes are worse than small ones
    - R² Score: Proportion of variance explained (0-1)
      * 1.0 = perfect predictions
      * 0.0 = no better than predicting mean
      * Negative = worse than predicting mean
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== {model_name} PERFORMANCE ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics

def plot_predictions(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot actual vs predicted values
    
    Parameters:
    -----------
    y_true : array
        Actual values
    y_pred : array
        Predicted values
    model_name : str
        Title for plot
    save_path : str
        Path to save figure (optional)
        
    Why this visualization?
    - Shows if predictions are accurate (points near diagonal line)
    - Reveals systematic errors (all points above/below line)
    - Identifies outliers (points far from line)
    - Visual assessment of model quality
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of predictions
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line (diagonal)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot residuals (errors) distribution
    
    Parameters:
    -----------
    y_true : array
        Actual values
    y_pred : array
        Predicted values
    model_name : str
        Title for plot
    save_path : str
        Path to save figure (optional)
        
    Why residual plot?
    - Residual = actual - predicted (the error)
    - Good model: residuals randomly scattered around zero
    - Bad patterns:
      * Funnel shape: variance increases (heteroscedasticity)
      * Curved pattern: non-linear relationship missed
      * Clusters: missing important features
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'{model_name}: Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name}: Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_feature_importance(feature_names, importance_values, top_n=15, 
                           model_name="Model", save_path=None):
    """
    Plot feature importance scores
    
    Parameters:
    -----------
    feature_names : list
        Names of features
    importance_values : array
        Importance scores
    top_n : int
        Number of top features to show
    model_name : str
        Title for plot
    save_path : str
        Path to save figure (optional)
        
    Why feature importance?
    - Shows which features most influence predictions
    - Helps understand model decisions
    - Identifies key performance indicators
    - Can guide feature selection for simpler models
    - For Linear Regression: coefficient magnitudes
    - For Random Forest: Gini importance or permutation importance
    """
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # Sort and get top N
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'], align='center')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'{model_name}: Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest at top
    plt.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def compare_models(results_dict, save_path=None):
    """
    Create comparison visualizations for multiple models
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
        Example: {'Linear Regression': {'mae': 2.5, 'rmse': 3.2, 'r2': 0.85}, ...}
    save_path : str
        Path to save figure (optional)
        
    Why compare models?
    - Shows which model performs best
    - Reveals trade-offs (accuracy vs speed, interpretability)
    - Helps justify model selection in report
    - Visual comparison easier than tables
    """
    models = list(results_dict.keys())
    metrics = ['MAE', 'RMSE', 'R²']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['mae', 'rmse', 'r2']):
        values = [results_dict[model][metric] for model in models]
        
        axes[idx].bar(models, values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[idx].set_ylabel(metrics[idx], fontsize=12)
        axes[idx].set_title(f'{metrics[idx]} Comparison', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

import pandas as pd  # Add this import at the top