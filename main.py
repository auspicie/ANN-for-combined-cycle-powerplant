# main.py
import os
import pandas as pd
from src.data_loader import load_ccpp_data
from src.preprocessing import split_data, create_preprocessing_pipeline
from src.models import train_model, save_model, load_model, train_voting_regressor, train_stacking_regressor # Make sure train_voting_regressor and train_stacking_regressor are imported
from src.evaluation import evaluate_model, plot_predictions, plot_residuals, plot_feature_importance
from src.utils import set_plot_style


def main():
    """
    Main function to run the Combined Cycle Power Plant prediction project.
    """
    print("--- Starting Combined Cycle Power Plant Prediction Project ---")

    # Set consistent plot style
    set_plot_style()

    # --- Configuration ---
    DATA_FILEPATH = 'data/Folds5x2_pp.xlsx'
    TARGET_COLUMN = 'PE'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42 # For reproducibility

    # Moved inside main()
    MODELS_TO_TRAIN = {
        'LinearRegression': {},
        'RandomForest': {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__max_features': [0.6, 0.8, 1.0],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_leaf': [1, 2, 4],
        },
        'GradientBoosting': {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.15],
            'regressor__max_depth': [3, 4, 5, 6],
            'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
            'regressor__max_features': [0.6, 0.8, 1.0],
        },
        'SVR': {
            'regressor__C': [10, 100, 1000],
            'regressor__epsilon': [0.5, 1, 2],
        },
        'MLP': {
            'regressor__hidden_layer_sizes': [(50,), (100, 50), (100, 100)],
            'regressor__activation': ['relu', 'tanh'],
            'regressor__alpha': [0.0001, 0.001, 0.01]
        }
    }

    # --- Configuration for Ensemble Models ---
    ENSEMBLE_MODELS_TO_TRAIN = {
        'VotingRegressor': {
            'voting_regressor__weights': [[0.3, 0.3, 0.4], [0.2, 0.4, 0.4], [0.4, 0.3, 0.3]]
        },
        'StackingRegressor': {
            'stacking_regressor__final_estimator__n_estimators': [50, 100],
            'stacking_regressor__final_estimator__max_depth': [5, 10]
        }
    }

    # --- 1. Load Data ---
    print("\n--- Phase 1: Data Loading ---")
    df = load_ccpp_data(filepath=DATA_FILEPATH, sheet_name=0)
    if df is None:
        print("Exiting due to data loading failure.")
        return

    # --- 2. Data Preprocessing and Splitting ---
    print("\n--- Phase 2: Data Preprocessing & Splitting ---")
    X_train, X_test, y_train, y_test = split_data(df, target_column=TARGET_COLUMN,
                                                  test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preprocessor = create_preprocessing_pipeline() # Moved inside main()

    # --- 3. Model Training ---
    print("\n--- Phase 3: Model Training (Individual Models) ---")
    trained_models = {}
    for model_name, param_grid in MODELS_TO_TRAIN.items():
        try:
            model = train_model(model_name, X_train, y_train, preprocessor,
                                param_grid=param_grid, random_state=RANDOM_STATE)
            trained_models[model_name] = model
            save_model(model, f'{model_name.lower().replace(" ", "_")}_model.pkl')
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    # --- 3b. Model Training (Ensemble Models) ---
    print("\n--- Phase 3b: Model Training (Ensemble Models) ---")
    required_base_models = ['RandomForest', 'GradientBoosting', 'SVR']
    if all(model_name in trained_models for model_name in required_base_models):
        try:
            # Train Voting Regressor
            voting_model = train_voting_regressor(X_train, y_train, preprocessor,
                                                  trained_models,
                                                  param_grid=ENSEMBLE_MODELS_TO_TRAIN.get('VotingRegressor'),
                                                  random_state=RANDOM_STATE)
            trained_models['VotingRegressor'] = voting_model
            save_model(voting_model, 'voting_regressor_model.pkl')

            # Train Stacking Regressor
            stacking_model = train_stacking_regressor(X_train, y_train, preprocessor,
                                                      trained_models,
                                                      param_grid=ENSEMBLE_MODELS_TO_TRAIN.get('StackingRegressor'),
                                                      random_state=RANDOM_STATE)
            trained_models['StackingRegressor'] = stacking_model
            save_model(stacking_model, 'stacking_regressor_model.pkl')

        except Exception as e:
            print(f"Error training ensemble models: {e}")
            print("Please ensure RandomForest, GradientBoosting, and SVR are trained successfully.")
    else:
        print(f"Skipping ensemble model training: Not all required base models ({required_base_models}) were trained successfully.")

    # --- 4. Model Evaluation ---
    print("\n--- Phase 4: Model Evaluation ---")
    results = {}
    PLOT_SAVE_PATH = 'plots'
    os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

    for model_name, model in trained_models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, model_name=model_name)
        results[model_name] = metrics

        plot_predictions(y_test, y_pred, model_name=model_name, save_path=PLOT_SAVE_PATH)
        plot_residuals(y_test, y_pred, model_name=model_name, save_path=PLOT_SAVE_PATH)

        if hasattr(model.named_steps.get('regressor'), 'feature_importances_'):
            plot_feature_importance(model, X_train.columns.tolist(), model_name=model_name, save_path=PLOT_SAVE_PATH)
        elif hasattr(model.named_steps.get('voting_regressor'), 'estimators_'):
            print(f"Feature importance not directly applicable for {model_name} as a whole. You can inspect base estimators.")

    print("\n--- All Model Results ---")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n--- Project Finished ---")
    print(f"Plots saved to the '{PLOT_SAVE_PATH}' directory.")
    print("Trained models saved to the 'trained_models' directory.")

if __name__ == "__main__":
    main()