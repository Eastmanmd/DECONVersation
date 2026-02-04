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
    --log True
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import logging
import pickle
import psutil
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# Function to load train and test files
def load_csv(path, name):
    """Load a CSV file and validate existence."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found at: {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded {name} with shape: {df.shape}")
    return df

# Function to evaluate model performance
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

# Function to calculate RAM usage
process = psutil.Process(os.getpid())
def mem_mb():
    return process.memory_info().rss / 1024**2


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
    parser.add_argument("--log", type=bool, default=True, help="verbose: output to log file")
    parser.add_argument("--log_file", default="./logs/logLfile.log", help="Log output file")
    args = parser.parse_args()

    # Load data
    print("Splitting data into training and validation sets")

    # Initialize time logger
    log = args.log    

    # If log == True (initialize logger)
    if log:
        logging.basicConfig(format='%(asctime)s %(message)s', 
                        filename=args.log_file,
                        filemode='w', force = True)

    # Load Train data
    if log:
        logging.warning('Load data')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")
    
    X = load_csv(args.train_latent, "Training latent embeddings")
    y = load_csv(args.train_prop, "Training cell-type proportions")

    
    # Ensure alignment
    #common_samples = X.index.intersection(y.index)
    #X, y = X.loc[common_samples], y.loc[common_samples]
    #print(f"Aligned training data: {X.shape[0]} samples")

    # Log split data
    if log:
        logging.warning('Split data into train and val')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Split into train/validation for tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Define hyperparameter grid
    print("Initializing grid parameters")

    # Define parameters
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']}

    # Log: initialize base model
    if log:
        logging.warning('Initialize base model')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Initialize base model
    rf_base = RandomForestRegressor(random_state=args.random_state, n_jobs=-1)

    # Log: Perform grid search
    if log:
        logging.warning('Performing grid search')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Initialize grid search
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=2)

    # Fit grid seach to training data
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Log: grid search done
    if log:
        logging.warning('Grid search done')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")
    
    print("\nBest hyperparameters found:")

    # Print out best parameters
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    
    # Log: Evaluate on validation set
    if log:
        logging.warning('Evaluate on validation data')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Predict on validation set
    y_val_pred = best_model.predict(X_val)

    # Print out validation metrics
    evaluate_model(y_val, y_val_pred, label="Validation")

    # Save validation predictions
    val_pred_df = pd.DataFrame(y_val_pred, columns=y.columns, index=X_val.index)
    val_pred_df.to_csv(args.out_val_pred)
    
    print(f"Saved validation predictions to {args.out_val_pred}")

    # Log: Saving validation predictions
    if log:
        logging.warning('Saving val predictions')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    y_val.to_csv(os.path.join(os.path.dirname(args.out_val_pred), "validation_proportions.csv"))

    # Log: Saved validation predictions
    if log:
        logging.warning('Saved val proportion')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Refit best model on entire training + validation set
    print("\nRefitting best model on full training data...")

    # Log: Refit model on full train data
    if log:
        logging.warning('Refit model on full train data')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Concat tran and validation splits
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    # Fut model on full data
    best_model.fit(X_full, y_full)
    
    # Log: Done: RF model re-fit on train data'
    if log:
        logging.warning('Done: RF model re-fit on train data')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Log: Saving trained models
    if log:
        logging.warning('Saving trained model')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Save the trained model
    with open(args.out_model, "wb") as f:
        pickle.dump(best_model, f)
        
    print(f"Saved best model to {args.out_model}")

    # Log: Saved trained models
    if log:
        logging.warning('Done: Saved trained model')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Load test set
    X_test = load_csv(args.test_latent, "Test latent embeddings")
    X_test = X_test[X.columns]  # Ensure feature alignment

    # Log: Predict on trained model
    if log:
        logging.warning('Predict on test data')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Predict on test data
    y_test_pred = best_model.predict(X_test)

    # Log: Prediction done
    if log:
        logging.warning('Prediction done')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")

    # Convert to dataframe
    test_pred_df = pd.DataFrame(y_test_pred, columns=y.columns, index=X_test.index)

    if log:
        logging.warning('Saving test predictions')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")
            
    # Output to csv
    test_pred_df.to_csv(args.out_test_pred)
    print(f"Saved test predictions to {args.out_test_pred}")

    print("\n✅ Training, tuning, and prediction completed successfully.")

    if log:
        logging.warning('✅ Saved test predictions')
        logging.warning('Memory used: ' + str(mem_mb()) + " MB")
        

if __name__ == "__main__":
    main()