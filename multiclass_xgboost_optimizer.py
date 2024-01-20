#!/usr/bin/env python
"""
As features are selected and hyperparameters are optimized on the entire dataset without a separated validation set, 
this script is designed for an XGBoost classifier in the context of SHAP analysis only.  

Requirements:
- Python 3.9 or higher.
- Libraries: pandas, numpy, scikit-learn, xgboost, optuna, sklearnex (optional, for Intel CPU acceleration).
- A classification dataset in CSV format with a 'target' column.

Script Arguments:
- --dataset: Path to the CSV dataset file.
- --target_list: Comma-separated string specifying the target classes.
- --n_trials: Number of trials for Optuna optimization.
- --timeout: Time limit in seconds for the optimization process.

Running the Script:
Navigate to the directory containing the script and run it with the necessary parameters. Example:
    python multiclass_xgboost_optimizer.py --fn_data data.csv --target_list "class1,class2,class3" --n_trials 300 --timeout 600

Output:
Saves the selected features to CSV and the parameters to JSON.
"""

import pandas as pd
import numpy as np
import argparse
import re
import json
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import optuna
from sklearnex import patch_sklearn

# Accelerate sklearn operations on Intel CPUs
patch_sklearn()

# Ignore warnings for cleaner output
warnings.simplefilter(action='ignore')  

def filter_classes(df: pd.DataFrame, target_list: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Filter the DataFrame to keep only specified target classes."""
    og_classes = sorted(df.target.unique().tolist())
    classes_to_keep = [
        target 
        for arg_cls in target_list 
        for target in og_classes 
        if re.search(arg_cls, target, re.IGNORECASE)
        ]
    classes_to_keep.sort()

    print(f"Keeping classes: {classes_to_keep}")
    return df[df.target.isin(classes_to_keep)], classes_to_keep

def feature_selection(X: pd.DataFrame, y: pd.Series) -> RFECV:
    """Perform feature selection using RFECV with XGBClassifier."""
    rfecv = RFECV(
        estimator=XGBClassifier(),
        step=1,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=-1
    )
    rfecv.fit(X, y)
    return rfecv

def objective(trial: optuna.Trial, data: pd.DataFrame, target: pd.Series) -> float:
    """Objective function for Optuna optimization."""
    params = {
        "verbosity": 0,
        "objective": "multi:softprob",
        "num_class": len(np.unique(target)),
        "tree_method": "exact",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "max_depth": trial.suggest_int("max_depth", 2, 20, step=2),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300, step=20, log=False),
        "gamma": trial.suggest_loguniform('gamma', 1e-8, 1.0),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    target_pred = cross_val_predict(XGBClassifier(**params), data, target, cv=cv, n_jobs=-1)
    accuracy = accuracy_score(target, target_pred)
    return accuracy

def main(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.fn_data, index_col="stimulus_id")
    df, classes_to_keep = filter_classes(df, args.target_list.split(","))

    X = df.drop("target", axis=1)
    y = LabelEncoder().fit_transform(df["target"])

    rfecv = feature_selection(X, y)
    print(f"Optimal number of features: {rfecv.n_features_}")
    selected_features = X.columns[rfecv.support_].tolist()
    print(f"Selected features: {selected_features}")

    # Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X[selected_features], y), 
        n_trials=args.n_trials, 
        timeout=args.timeout
    )

    print("\nNumber of finished trials: ", len(study.trials))
    print("\nBest trial:")
    trial = study.best_trial
    print("\tValue: ", trial.value)
    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")

    fn = f"xgboost_params_{args.n_trials}_{classes_to_keep[0]}_{classes_to_keep[1]}.json"
    print(f"\nSaving best trial to {fn}")
    with open(fn, "w") as f:
        json.dump(study.best_params, f)

    X_selected = df[selected_features + ["target"]]
    clsstr = "_".join(classes_to_keep)
    fn = f"selected_features_{clsstr}.csv"
    print(f"Saving {fn}")
    X_selected.to_csv(fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_data", help="File path for the features dataset", type=str, default="features_mean_with_target.csv")
    parser.add_argument("--target_list", help='e.g., --target_list "class1,class2,class3"', type=str, required=True)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()

    main(args)
