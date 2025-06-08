# src/models.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor # Uncomment if you install xgboost
# from lightgbm import LGBMRegressor # Uncomment if you install lightgbm
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
import joblib
import os

def train_model(model_name, X_train, y_train, preprocessor, param_grid=None, random_state=42):
    """
    Trains a specified machine learning model, optionally with hyperparameter tuning.

    Args:
        model_name (str): Name of the model ('LinearRegression', 'RandomForest', 'GradientBoosting', 'SVR', 'MLP').
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        preprocessor (sklearn.pipeline.Pipeline): The preprocessing pipeline to apply.
        param_grid (dict, optional): Dictionary of hyperparameters for GridSearchCV.
        random_state (int): Seed for random processes for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: Trained model encapsulated in a pipeline.
    """
    print(f"--- Training {model_name} ---")

    model = None
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=random_state)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=random_state)
    # elif model_name == 'XGBoost': # Uncomment if using XGBoost
    #     model = XGBRegressor(random_state=random_state, objective='reg:squarederror', n_jobs=-1)
    # elif model_name == 'LightGBM': # Uncomment if using LightGBM
    #     model = LGBMRegressor(random_state=random_state, n_jobs=-1)
    elif model_name == 'SVR':
        model = SVR()
    elif model_name == 'MLP':
        # MLP can be sensitive to scaling and might need more iterations for convergence
        model = MLPRegressor(random_state=random_state, max_iter=500, early_stopping=True, n_iter_no_change=50)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Create a pipeline that first preprocesses, then applies the model
    pipeline = Pipeline([
        ('preprocessor', preprocessor), # Re-uses the preprocessor from preprocessing.py
        ('regressor', model)
    ])

    if param_grid:
        print(f"Performing GridSearchCV for {model_name}...")
        # KFold for cross-validation during hyperparameter tuning
        # Use a consistent random_state for reproducible splits
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation RMSE for {model_name}: {-grid_search.best_score_**0.5:.4f}")
        return grid_search.best_estimator_
    else:
        print(f"Training {model_name} without hyperparameter tuning...")
        pipeline.fit(X_train, y_train)
        return pipeline

def save_model(model, filename, path='trained_models'):
    """Saves a trained model to a file."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename, path='trained_models'):
    """Loads a trained model from a file."""
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    else:
        print(f"Error: Model file not found at {filepath}")
        return None
# --- New function for training Voting Regressor ---
def train_voting_regressor(X_train, y_train, preprocessor, best_individual_models, param_grid=None, random_state=42):
    """
    Trains a VotingRegressor, combining predictions from base estimators.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        preprocessor (sklearn.pipeline.Pipeline): The preprocessing pipeline.
        best_individual_models (dict): Dictionary of best trained individual models.
                                       Expected keys: 'RandomForest', 'GradientBoosting', 'SVR' (or others).
        param_grid (dict, optional): Hyperparameter grid for VotingRegressor.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: Trained VotingRegressor encapsulated in a pipeline.
    """
    print("\n--- Training VotingRegressor ---")

    # Use the best estimators found from previous individual model training
    # Ensure these keys exist in best_individual_models
    estimators = [
        ('rf', best_individual_models['RandomForest'].named_steps['regressor']),
        ('gb', best_individual_models['GradientBoosting'].named_steps['regressor']),
        ('svr', best_individual_models['SVR'].named_steps['regressor'])
        # You can add more estimators here
    ]

    voting_model = VotingRegressor(estimators=estimators, n_jobs=-1)

    # The pipeline for VotingRegressor still needs the preprocessor,
    # but the base estimators *inside* VotingRegressor receive already preprocessed data.
    # So, we apply preprocessor *before* the VotingRegressor.
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('voting_regressor', voting_model)
    ])

    if param_grid:
        print("Performing GridSearchCV for VotingRegressor...")
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for VotingRegressor: {grid_search.best_params_}")
        print(f"Best cross-validation RMSE for VotingRegressor: {-grid_search.best_score_**0.5:.4f}")
        return grid_search.best_estimator_
    else:
        print("Training VotingRegressor without hyperparameter tuning...")
        pipeline.fit(X_train, y_train)
        return pipeline

# --- New function for training Stacking Regressor ---
def train_stacking_regressor(X_train, y_train, preprocessor, best_individual_models, param_grid=None, random_state=42):
    """
    Trains a StackingRegressor, combining predictions from base estimators
    with a meta-regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        preprocessor (sklearn.pipeline.Pipeline): The preprocessing pipeline.
        best_individual_models (dict): Dictionary of best trained individual models.
                                       Expected keys: 'RandomForest', 'GradientBoosting', 'SVR' (or others).
        param_grid (dict, optional): Hyperparameter grid for StackingRegressor.
        random_state (int): Seed for reproducibility.

    Returns:
        sklearn.pipeline.Pipeline: Trained StackingRegressor encapsulated in a pipeline.
    """
    print("\n--- Training StackingRegressor ---")

    # Use the best estimators found from previous individual model training
    estimators = [
        ('rf', best_individual_models['RandomForest'].named_steps['regressor']),
        ('gb', best_individual_models['GradientBoosting'].named_steps['regressor']),
        ('svr', best_individual_models['SVR'].named_steps['regressor'])
        # You can add more estimators here
    ]

    # Define the meta-regressor
    # A simple Linear Regression or a more complex model like RandomForest can be used.
    meta_regressor = LinearRegression() # Or RandomForestRegressor(random_state=random_state)

    stacking_model = StackingRegressor(estimators=estimators, final_estimator=meta_regressor, cv=5, n_jobs=-1)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking_regressor', stacking_model)
    ])

    if param_grid:
        print("Performing GridSearchCV for StackingRegressor...")
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters for StackingRegressor: {grid_search.best_params_}")
        print(f"Best cross-validation RMSE for StackingRegressor: {-grid_search.best_score_**0.5:.4f}")
        return grid_search.best_estimator_
    else:
        print("Training StackingRegressor without hyperparameter tuning...")
        pipeline.fit(X_train, y_train)
        return pipeline

if __name__ == '__main__':
    # This block needs X_train, y_train and preprocessor from main.py or dummy data
    # To run this example standalone, you'd need to mock the data and preprocessor
    print("This module is typically run via main.py to get proper data and preprocessor instances.")
    # Example of how you might define param_grids in main.py:
    # rf_param_grid = {
    #     'regressor__n_estimators': [100, 200],
    #     'regressor__max_depth': [10, 20]
    # }