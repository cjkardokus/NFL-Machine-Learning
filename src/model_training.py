"""
NFL Model Training Script
Trains and compares multiple linear regression models to predict team win percentage
"""

import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Sklearn models
from sklearn.linear_model import LinearRegression, Ridge

# Sklearn metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sklearn preprocessing and validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def load_data(data_path):
    """
    Load and return the clean dataset

    Parameter: 
    ----------
    data_path : Path
        Path to the CSV file

    Return:
    -------
    df : Dataframe 
        The resulting dataframe from the CSV file
    """
    
    df = pd.read_csv(data_path)

    print(f"Loaded data shape: {df.shape}")
    print(df.head())

    return df


def determine_path(target_dir):
    """
    Determine the correct path to a target directory from current working directory
    
    Parameters:
    -----------
    target_dir : str
        Name of target directory relative to project root (e.g., 'outputs', 'models', 'data/processed')
    
    Returns:
    --------
    Path : Path object pointing to the target directory
    
    Example:
    --------
    output_dir = determine_path('outputs')
    models_dir = determine_path('models')
    data_dir = determine_path('data/processed')
    """
    current_dir = Path.cwd()
    
    # Determine project root
    if 'src' in str(current_dir):
        project_root = current_dir.parent
    else:
        project_root = current_dir
    
    # Build path to target directory
    target_path = project_root / target_dir
    
    # Create directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    return target_path


def clean_filename(model_name):
    """
    Convert model name to clean filename
    
    Parameters:
    -----------
    model_name : str
        Model name (e.g., "Model 1: Minimal")
    
    Returns:
    --------
    str : Cleaned filename (e.g., "model_1_minimal")
    """
    return model_name.replace(':', '').replace(' ', '_').lower()


def calculate_standardized_coefficients(model, features, X_train):
    """
    Calculate standardized coefficients for feature importance
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with coef_ attribute
    features : list
        List of feature names
    X_train : DataFrame
        Training data (used to calculate standard deviations)
    
    Returns:
    --------
    DataFrame : DataFrame with features, raw coefficients, and standardized coefficients,
                sorted by absolute standardized coefficient (descending)
    """
    coefficients = model.coef_
    
    # Calculate standardized coefficients
    standardized_coefs = []
    for i, feature in enumerate(features):
        std_feature = X_train[feature].std()
        std_coef = coefficients[i] * std_feature
        standardized_coefs.append(std_coef)
    
    # Create importance dataframe and sort by absolute value
    importance_df = pd.DataFrame({
        'feature': features,
        'raw_coefficient': coefficients,
        'standardized_coefficient': standardized_coefs
    }).sort_values('standardized_coefficient', key=abs, ascending=False)
    
    return importance_df


def create_scatter_plot(y_test, y_pred, r2, mae, model_name, output_dir):
    """
    Create and save predicted vs actual scatter plot
    
    Parameters:
    -----------
    y_test : Series
        Actual test values
    y_pred : array
        Predicted values
    r2 : float
        R² score
    mae : float
        Mean absolute error
    model_name : str
        Name of the model for title
    output_dir : Path
        Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1.5)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Predictions')
    
    plt.xlabel('Actual Win Percentage', fontsize=12)
    plt.ylabel('Predicted Win Percentage', fontsize=12)
    plt.title(f'{model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    clean_name = clean_filename(model_name)
    plt.savefig(output_dir / f'{clean_name}_predictions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Scatter plot saved: {clean_name}_predictions_scatter.png")


def create_feature_importance_plot(model_name, feature_importance_data, output_dir):
    """
    Create and save feature importance bar chart
    
    Parameters:
    -----------
    model_name : str
        Name of the model for title
    feature_importance_data : list of dict
        List of dictionaries with 'feature', 'raw_coefficient', 'standardized_coefficient'
    output_dir : Path
        Directory to save the plot
    """
    # Convert to DataFrame and sort
    importance_data = pd.DataFrame(feature_importance_data)
    importance_data = importance_data.sort_values('standardized_coefficient', key=abs, ascending=True)
    
    # Determine figure height based on number of features
    num_features = len(importance_data)
    fig_height = max(6, num_features * 0.4)
    
    plt.figure(figsize=(10, fig_height))
    
    # Color bars: red for negative, blue for positive
    colors = ['red' if x < 0 else 'blue' for x in importance_data['standardized_coefficient']]
    plt.barh(importance_data['feature'], importance_data['standardized_coefficient'], color=colors)
    
    plt.xlabel('Standardized Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name}: Feature Importance', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Save plot
    clean_name = clean_filename(model_name)
    plt.savefig(output_dir / f'{clean_name}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Created feature importance plot: {clean_name}_feature_importance.png")


def create_comparison_dashboard(model_names, r2_scores, mae_scores, rmse_scores, cv_means, cv_stds, output_dir):
    """
    Create and save 2x2 model comparison dashboard
    
    Parameters:
    -----------
    model_names : list
        List of model names
    r2_scores : list
        Test R² scores for each model
    mae_scores : list
        Test MAE scores for each model
    rmse_scores : list
        Test RMSE scores for each model
    cv_means : list
        Cross-validation mean R² for each model
    cv_stds : list
        Cross-validation std R² for each model
    output_dir : Path
        Directory to save the plot
    """
    # Shorten model names for x-axis
    short_names = ['Model 1\n(3 feat)', 'Model 2\n(5 feat)', 'Model 3\n(Ridge)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² comparison
    bars1 = axes[0, 0].bar(short_names, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('R² Score', fontsize=12)
    axes[0, 0].set_title('Test R² Score', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    # Highlight best model
    best_idx = r2_scores.index(max(r2_scores))
    bars1[best_idx].set_edgecolor('gold')
    bars1[best_idx].set_linewidth(3)
    
    # MAE comparison
    bars2 = axes[0, 1].bar(short_names, mae_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[0, 1].set_title('Test MAE (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    # Highlight best model (lowest MAE)
    best_idx = mae_scores.index(min(mae_scores))
    bars2[best_idx].set_edgecolor('gold')
    bars2[best_idx].set_linewidth(3)
    
    # RMSE comparison
    bars3 = axes[1, 0].bar(short_names, rmse_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Root Mean Squared Error', fontsize=12)
    axes[1, 0].set_title('Test RMSE (Lower is Better)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Cross-validation comparison with error bars
    axes[1, 1].bar(short_names, cv_means, yerr=cv_stds, capsize=10, 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1, 1].set_ylabel('Cross-Validation R²', fontsize=12)
    axes[1, 1].set_title('CV Performance (Mean ± Std)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Created model comparison plot: model_comparison.png")

def create_summary_table(model_names, all_results, best_model_name, output_dir):
    """
    Create and save model performance summary table as image
    
    Parameters:
    -----------
    model_names : list
        List of model names
    all_results : dict
        Dictionary containing results for all models
    best_model_name : str
        Name of the best performing model
    output_dir : Path
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table data
    table_data = []
    for model_name in model_names:
        data = all_results[model_name]
        table_data.append([
            model_name,
            f"{data['metrics']['r2']:.4f}",
            f"{data['metrics']['mae']:.4f}",
            f"{data['metrics']['rmse']:.4f}",
            f"{data['metrics']['cv_mean']:.4f} ± {data['metrics']['cv_std']:.4f}",
            len(data['features'])
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Test R²', 'Test MAE', 'Test RMSE', 'CV R² (Mean ± Std)', 'Features'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.12, 0.22, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Highlight best model row
    best_row_idx = list(model_names).index(best_model_name) + 1  # +1 for header
    for col_idx in range(6):
        table[(best_row_idx, col_idx)].set_facecolor('#90EE90')
        table[(best_row_idx, col_idx)].set_text_props(weight='bold')
    
    # Style header
    for col_idx in range(6):
        table[(0, col_idx)].set_facecolor('#4CAF50')
        table[(0, col_idx)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / 'model_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Created summary table: model_summary_table.png")


def split_data(df):
    """
    Split into training and testing data

    Parameter: 
    ----------
    df : Dataframe
        The entire DataFrame from the CSV containing 2015-2024 nfl statistics

    Returns:
    ---------
    X_train : DataFrame
        2015-2023 features for training
    y_train : Series
        win percentages for training
    X_test : DataFrame
        2024 season features for testing
    y_test : Series
        2024 season winning percentages
    """
    # Filter by season
    train_df = df[df['season'] < 2024]
    test_df = df[df['season'] == 2024]

    # Get feature columns (exclude identifiers and target)
    feature_cols = [col for col in df.columns if col not in ['season', 'team', 'win_percentage']]

    # Separate X and y
    X_train = train_df[feature_cols]
    y_train = train_df['win_percentage']
    X_test = test_df[feature_cols]
    y_test = test_df['win_percentage']

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, features, use_ridge=False, alpha=1.0):
    """
    Train a linear regression model with the provided list of features
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
    features : list
        List of feature names to use
    use_ridge : bool
        If True, use Ridge regression with regularization
    alpha : float
        Regularization strength for Ridge (only used if use_ridge=True)
    
    Returns:
    --------
    model : trained model
    scaler : StandardScaler (only returned if use_ridge=True, else None)
    """
    n = len(features)
    print(f"\nTraining Model with {n} features")
    print(f"  Features: {features}")
    print(f"  Model type: {'Ridge Regression' if use_ridge else 'Linear Regression'}")
    
    # Subset to selected features
    X_train_subset = X_train[features]
    
    if use_ridge:
        # Ridge requires feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_subset)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        print(f"  Regularization (alpha): {alpha}")
        print("Training completed")
        return model, scaler
    else:
        # Standard Linear Regression
        model = LinearRegression()
        model.fit(X_train_subset, y_train)
        
        print("Training completed")
        return model, None
    

def evaluate_model(model, features, X_train, y_train, X_test, y_test, model_name, scaler=None):
    """
    Evaluate a trained model with cross-validation and test metrics
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to evaluate
    features : list
        List of feature names used by the model
    X_train, y_train : training data
    X_test, y_test : test data
    model_name : str
        Name of the model for display
    scaler : StandardScaler or None
        Scaler used for Ridge models (None for Linear Regression)
    
    Returns:
    --------
    dict : Dictionary containing all performance metrics
    """
    print("\n" + "="*60)
    print(f"Evaluating {model_name}")
    print("="*60)
    
    # Subset data to model's features
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    # Scale if Ridge model
    if scaler is not None:
        X_train_scaled = scaler.transform(X_train_subset)
        X_test_scaled = scaler.transform(X_test_subset)
    else:
        X_train_scaled = X_train_subset
        X_test_scaled = X_test_subset
    
    # Cross-validation on training data (5-fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\nCross-Validation R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate test metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest Set Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Feature importance with standardized coefficients
    print(f"\nFeature Importance (Standardized Coefficients):")
    importance_df = calculate_standardized_coefficients(model, features, X_train)
    print(importance_df.to_string(index=False))

    # Create scatter plot
    output_dir = determine_path('outputs')
    create_scatter_plot(y_test, y_pred, r2, mae, model_name, output_dir)
    
    # Package metrics into dictionary
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': importance_df.to_dict('records')
    }
    
    return metrics


def create_visualizations(all_results, X_test, y_test, best_model_name):
    """
    Generate and save all performance visualizations
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all models
    X_test : DataFrame
        Test features
    y_test : Series
        Test target values
    best_model_name : str
        Name of the best performing model
    """
    
    # Determine output path
    output_dir = determine_path('outputs')
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Store predictions for comparison plot
    model_predictions = {}
    
    # 1. Individual model plots (Predicted vs Actual, Feature Importance)
    for model_name, data in all_results.items():
        # Get predictions (need to recreate from stored data)
        features = data['features']
        X_test_subset = X_test[features]
        
        # Note: We'd need to pass models to this function OR recalculate predictions
        # For now, let's create feature importance plots using stored data
        
        # Feature Importance Bar Chart
        create_feature_importance_plot(model_name, data['metrics']['feature_importance'], output_dir)
    
    # 2. Model Comparison Bar Charts
    model_names = list(all_results.keys())
    r2_scores = [all_results[m]['metrics']['r2'] for m in model_names]
    mae_scores = [all_results[m]['metrics']['mae'] for m in model_names]
    rmse_scores = [all_results[m]['metrics']['rmse'] for m in model_names]
    cv_means = [all_results[m]['metrics']['cv_mean'] for m in model_names]
    cv_stds = [all_results[m]['metrics']['cv_std'] for m in model_names]

    create_comparison_dashboard(model_names, r2_scores, mae_scores, rmse_scores, cv_means, cv_stds, output_dir)
    
    # 3. Summary table as image
    create_summary_table(model_names, all_results, best_model_name, output_dir)
    
    print("\n✓ All visualizations saved to outputs/ directory")


def save_results(all_results):
    """
    Save model training results to JSON file
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all models
        Format: {
            'Model 1: Minimal': {
                'features': [...],
                'metrics': {...}
            },
            ...
        }
    """
    # Determine output path
    output_dir = determine_path('outputs')
    
    # Structure the output
    results_summary = {
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_years': '2015-2023',
        'test_year': '2024',
        'models': {}
    }
    
    # Add each model's results
    for model_name, data in all_results.items():
        results_summary['models'][model_name] = {
            'test_r2': round(data['metrics']['r2'], 4),
            'test_mae': round(data['metrics']['mae'], 4),
            'test_rmse': round(data['metrics']['rmse'], 4),
            'cv_r2_mean': round(data['metrics']['cv_mean'], 4),
            'cv_r2_std': round(data['metrics']['cv_std'], 4),
            'num_features': len(data['features']),
            'features': data['features'],
            'feature_importance': data['metrics']['feature_importance']
        }
    
    # Save to JSON
    output_path = output_dir / 'model_training_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✓ Results saved to {output_path}")
    print("="*60)


def compare_models(all_results):
    """
    Compare models and identify the best performer
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results for all models
    
    Returns:
    --------
    str : Name of the best performing model (highest test R²)
    """
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    
    # Find best model by R² score
    best_model = None
    best_r2 = -1
    
    for model_name, data in all_results.items():
        r2 = data['metrics']['r2']
        if r2 > best_r2:
            best_r2 = r2
            best_model = model_name
    
    # Print simple comparison
    print(f"\n{'Model':<25} {'Test R²':<12} {'MAE':<12} {'Features'}")
    print("-" * 60)
    
    for model_name, data in all_results.items():
        metrics = data['metrics']
        num_features = len(data['features'])
        marker = " ⭐ BEST" if model_name == best_model else ""
        
        print(f"{model_name:<25} {metrics['r2']:<12.4f} {metrics['mae']:<12.4f} {num_features}{marker}")
    
    print("\n" + "="*60)
    print(f"Best Model: {best_model} (R² = {best_r2:.4f})")
    print("="*60)
    
    return best_model


def save_model(model, features, model_name, scaler=None):
    """
    Save trained model (and scaler if Ridge) to disk
    
    Parameters:
    -----------
    model : sklearn model
        Trained model to save
    features : list
        List of feature names used by the model
    model_name : str
        Name of the model for labeling saved files
    scaler : StandardScaler or None
        Scaler object if Ridge model, None otherwise
    """
    # Determine output path
    models_dir = determine_path('models')
    
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    
    # Save the model
    model_path = models_dir / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save the scaler if Ridge model
    if scaler is not None:
        scaler_path = models_dir / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature list (important for making predictions later!)
    features_path = models_dir / 'model_features.json'
    with open(features_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'features': features,
            'uses_scaler': scaler is not None
        }, f, indent=2)
    print(f"✓ Features saved: {features_path}")
    
    print(f"\n✓ {model_name} ready for deployment!")
    print("="*60)


def main():
    """Orchestrate the entire training pipeline"""
    
    print("="*60)
    print("NFL MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Load and split data
    data_path = determine_path('data/processed') / 'nfl_features_clean.csv'
    df = load_data(data_path)
    X_train, y_train, X_test, y_test = split_data(df)
    
    print("\n" + "="*60)
    print("Train/Test Split:")
    print(f"  Training samples: {len(X_train)} (2015-2023)")
    print(f"  Test samples: {len(X_test)} (2024)")
    print("="*60)
    
    # Define feature sets for each model
    features_model1 = [
        'epa_per_play_offense',
        'epa_per_play_defense',
        'turnover_differential_per_game'
    ]
    
    features_model2 = [
        'epa_per_play_offense',
        'epa_per_play_defense',
        'turnover_differential_per_game',
        'yards_per_game_offense',
        'yards_per_game_defense'
    ]
    
    features_model3 = [
        'epa_per_play_offense',
        'epa_per_play_defense',
        'turnover_differential_per_game',
        'yards_per_game_offense',
        'yards_per_game_defense',
        'success_rate_offense',
        'success_rate_defense',
        'rush_attempts_per_game',
        'pass_attempts_per_game'
    ]
    
    # Train Model 1: Minimal (3 features)
    model1, scaler1 = train_model(X_train, y_train, features_model1, use_ridge=False)
    metrics1 = evaluate_model(model1, features_model1, X_train, y_train, X_test, y_test, 
                              "Model 1: Minimal", scaler1)
    
    # Train Model 2: Core (5 features)
    model2, scaler2 = train_model(X_train, y_train, features_model2, use_ridge=False)
    metrics2 = evaluate_model(model2, features_model2, X_train, y_train, X_test, y_test,
                              "Model 2: Core", scaler2)
    
    # Train Model 3: Ridge (9 features)
    model3, scaler3 = train_model(X_train, y_train, features_model3, use_ridge=True, alpha=1.0)
    metrics3 = evaluate_model(model3, features_model3, X_train, y_train, X_test, y_test,
                              "Model 3: Ridge", scaler3)
    
    # Collect all results
    all_results = {
        'Model 1: Minimal': {
            'features': features_model1,
            'metrics': metrics1
        },
        'Model 2: Core': {
            'features': features_model2,
            'metrics': metrics2
        },
        'Model 3: Ridge': {
            'features': features_model3,
            'metrics': metrics3
        }
    }
    
    # Compare models and identify best
    best_model_name = compare_models(all_results)
    
    # Save results to JSON
    save_results(all_results)
    
    # Create all visualizations
    create_visualizations(all_results, X_test, y_test, best_model_name)
    
    # Save the best model
    if best_model_name == 'Model 1: Minimal':
        save_model(model1, features_model1, best_model_name, scaler1)
    elif best_model_name == 'Model 2: Core':
        save_model(model2, features_model2, best_model_name, scaler2)
    else:  # Model 3: Ridge
        save_model(model3, features_model3, best_model_name, scaler3)
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  8 visualization images in outputs/")
    print("  model_training_results.json in outputs/")
    print("  Best model saved in models/")
    print("\nReady for next phase: Generate team strength ratings!")
    print("="*60)

   
if __name__ == "__main__":
    main()