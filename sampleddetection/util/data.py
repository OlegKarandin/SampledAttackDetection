"""
Utility functions for loading datasets
"""

from typing import List, Tuple

import hyperopt
import joblib as joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

# TODO: if ever needed again make sure to get it compatible to new framework
# def relabel_df(
#     df: pd.DataFrame, group_attacks: bool, attacks_to_detect: List[Attack]
# ) -> Tuple[pd.DataFrame, List[int]]:
#     # keep the chosen attacks.
#     labels = [ATTACK_TO_STRING[attack] for attack in attacks_to_detect]
#     labels.append(ATTACK_TO_STRING[Attack.BENIGN])
#     relabelled_df = df[df["label"].isin(labels)]
#
#     # Set labels to 0/1
#     if not group_attacks:
#         temp_dict = {
#             ATTACK_TO_STRING[att_name]: i + 1
#             for i, att_name in enumerate(attacks_to_detect)
#         }
#     else:
#         temp_dict = {ATTACK_TO_STRING[att_name]: 1 for att_name in attacks_to_detect}
#
#     temp_dict[ATTACK_TO_STRING[Attack.BENIGN]] = 0
#     labels = list(temp_dict.values())
#     relabelled_df["label"] = relabelled_df["label"].map(temp_dict)
#     relabelled_df = relabelled_df.reset_index(drop=True)
#
#     return relabelled_df, labels


# def clean_dataset(
#     df: pd.DataFrame,
#     selected_features: List,
#     attacks_to_detect: List[Attack],
#     group_attacks: bool,
# ):
#     """
#     Params:
#     ~~~~~~~
#         - selected_features: list of features to retrieve from csv
#         - csv_location: csv location
#         - attacks_to_detect: list of attacks to retrieve filter for
#         - bool_classification: if true changes labeled attacks to 1, bening to 0
#         - group_attacks: take all attacks as a single binary label. Attack or no attack
#     """
#     old_df = df.copy()
#     # clean up
#     new_cols = {col: col.strip() for col in df.columns}
#     df.rename(columns=new_cols, inplace=True)
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.dropna(inplace=True)
#     # df.drop_duplicates(inplace=True)
#     # for feature in selected_features:
#     #     df = df[df[feature] >= 0]
#     df = df[selected_features]
#
#     # Relabel the dataset
#     relabelled_df, labels = relabel_df(df, group_attacks, attacks_to_detect)
#
#     return relabelled_df
#


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


def train_multinary_classier_XGBoost(X_train, y_train, X_val, y_val):
    """
    Same as above but we make for multiple categories
    """

    obj = "multi:softprob"
    ev_metric = ["merror", "mlogloss"]
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
    y_pred_proba = xgb_cl.predict_proba(X_val)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "log_loss": log_loss(y_val, y_pred_proba),
        # "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }
    return xgb_cl, metrics
