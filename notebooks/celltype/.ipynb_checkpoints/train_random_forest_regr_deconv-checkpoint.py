"""
Train and evaluate a Random Forest Regressor with hyperparameter tuning
to predict cell-type proportions from pseudobulk latent embeddings.

Example usage:
---------------
python train_rf_deconv_tuned.py \
    --train_latent /path/to/train_latent.csv \
    --train_prop /path/to/train_prop.csv \
    --test_latent /path/to/test_latent.csv \
    --out_val_pred /path/to/output_val_predictions.csv \
    --out_test_pred /path/to/output_test_predictions.csv \
    --out_model /path/to/best_model.pkl
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def load_csv(path, name):
    """Load a CSV file and validate existence."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found at: {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded {name} with shape: {df.shape}")
    return df


def evaluate_model(y_true, y_pred, label="Validation"):
    """Compute performance metrics."""
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n[{label} Performance]")
    print(f"  R²: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser(description="Random Forest cell-type deconvolution with hyperparameter tuning.")
    parser.add_argument("--train_latent", required=True, help="Path to training latent pseudobulk embeddings CSV.")
    parser.add_argument("--train_prop", required=True, help="Path to training cell-type proportion CSV.")
    parser.add_argument("--test_latent", required=True, help="Path to test latent pseudobulk embeddings CSV.")
    parser.add_argument("--out_val_pred", required=True, help="Path to save validation predictions CSV.")
    parser.add_argument("--out_test_pred", required=True, help="Path to save test predictions CSV.")
    parser.add_argument("--out_model", required=True, help="Path to save best Random Forest model (pickle).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of training data for validation.")
    args = parser.parse_args()

    # Load data
    X = load_csv(args.train_latent, "Training latent embeddings")
    y = load_csv(args.train_prop, "Training cell-type proportions")

    # Ensure alignment
    #common_samples = X.index.intersection(y.index)
    #X, y = X.loc[common_samples], y.loc[common_samples]
    #print(f"Aligned training data: {X.shape[0]} samples")

    # Split into train/validation for tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    # Initialize base model
    rf_base = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)

    print("\nPerforming grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("\nBest hyperparameters found:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    evaluate_model(y_val, y_val_pred, label="Validation")

    # Save validation predictions
    val_pred_df = pd.DataFrame(y_val_pred, columns=y.columns, index=X_val.index)
    val_pred_df.to_csv(args.out_val_pred)
    print(f"Saved validation predictions to {args.out_val_pred}")

    y_val.to_csv(os.path.join(os.path.dirname(args.out_val_pred), "validation_proportions.csv"))
    print(f"Saved validation proportions")

    # Refit best model on entire training + validation set
    print("\nRefitting best model on full training data...")
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    best_model.fit(X_full, y_full)

    # Save the trained model
    with open(args.out_model, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved best model to {args.out_model}")

    # Load test set
    X_test = load_csv(args.test_latent, "Test latent embeddings")
    X_test = X_test[X.columns]  # Ensure feature alignment

    # Predict on test data
    print("\nPredicting on test data...")
    y_test_pred = best_model.predict(X_test)
    test_pred_df = pd.DataFrame(y_test_pred, columns=y.columns, index=X_test.index)
    test_pred_df.to_csv(args.out_test_pred)
    print(f"Saved test predictions to {args.out_test_pred}")

    print("\n✅ Training, tuning, and prediction completed successfully.")


if __name__ == "__main__":
    main()