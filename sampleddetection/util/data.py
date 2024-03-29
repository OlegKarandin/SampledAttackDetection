"""
Utility functions for loading datasets
"""
from typing import List

import hyperopt
import joblib as joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from ..common_lingo import ATTACK_TO_STRING, Attack


def clean_dataset(
    df: pd.DataFrame,
    selected_features: List,
    attacks_to_detect: List[Attack],
):
    """
    Params:
    ~~~~~~~
        - selected_features: list of features to retrieve from csv
        - csv_location: csv location
        - attacks_to_detect: list of attacks to retrieve filter for
        - bool_classification: if true changes labeled attacks to 1, bening to 0
    """
    # clean up
    new_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=new_cols, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # df.drop_duplicates(inplace=True)
    # for feature in selected_features:
    #     df = df[df[feature] >= 0]
    df = df[selected_features]

    # keep the chosen attacks
    labels = [ATTACK_TO_STRING[attack] for attack in attacks_to_detect]
    labels.append(ATTACK_TO_STRING[Attack.BENIGN])
    relabelled_df = df[df["label"].isin(labels)]

    # Set labels to 0/1
    temp_dict = {ATTACK_TO_STRING[att_name]: 1 for att_name in attacks_to_detect}
    temp_dict[ATTACK_TO_STRING[Attack.BENIGN]] = 0
    relabelled_df.replace({"label": temp_dict}, inplace=True)
    # Reset the index of the DataFrame after filtering
    relabelled_df = relabelled_df.reset_index(drop=True)

    # Balance the dataset
    balanced_df = pd.DataFrame()
    samples_per_class = 1200
    for label in [0, 1]:
        class_df = relabelled_df[relabelled_df["label"] == label]
        random_indices = np.random.choice(
            class_df.index, samples_per_class, replace=False
        )
        balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    print(balanced_df["label"].value_counts())

    return balanced_df


def train_classifier_XGBoost(X_train, y_train, X_val, y_val):
    obj = "binary:logistic"
    ev_metric = ["error", "logloss"]
    xgb_feature_space = {
        "n_estimators": hp.choice("n_estimators", np.arange(10, 101, 10)),
        "max_depth": hp.choice("max_depth", np.arange(5, 51, 5)),
        "subsample": hp.choice("subsample", [0.7, 0.8, 0.9, 1]),
        "gamma": hp.choice("gamma", np.arange(0, 5, 1)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
    }

    def hyperopt_train_test(params):
        model = XGBClassifier(
            verbosity=0, objective=obj, eval_metric=ev_metric, **params
        )
        return cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

    def f(params):
        score_vect = hyperopt_train_test(params)
        return {
            "status": STATUS_OK,
            "loss": -np.mean(score_vect),
            "loss_variance": np.var(score_vect, ddof=1),
        }

    trials = Trials()
    best_params = fmin(
        f,
        xgb_feature_space,
        algo=tpe.suggest,
        max_evals=25,
        early_stop_fn=no_progress_loss(5),
        trials=trials,
    )
    best_params = hyperopt.space_eval(xgb_feature_space, best_params)
    print("Best parameters:", best_params)
    xgb_cl = XGBClassifier(
        verbosity=0, objective=obj, eval_metric=ev_metric, **best_params
    )
    xgb_cl.fit(X_train, y_train)
    # Evaluate the classifier on the validation set
    y_pred = xgb_cl.predict(X_val)
    y_pred_proba = xgb_cl.predict_proba(X_val)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "log_loss": log_loss(y_val, y_pred_proba),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }
    return xgb_cl, metrics
