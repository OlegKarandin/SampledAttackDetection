import pickle

import hyperopt
import joblib as joblib
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from sampleddetection.util.data import train_multinary_classier_XGBoost
from sampleddetection.utils import pretty_print

# import matplotlib.pyplot as plt


def train_classifier_RF(X_train, y_train):
    rf_feature_space = {
        "n_estimators": hp.choice("n_estimators", np.arange(10, 101, 10)),
        "max_depth": hp.choice("max_depth", np.arange(5, 51, 5)),
        "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 10, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "min_samples_split": hp.choice("min_samples_split", np.arange(2, 10, 1)),
        "max_features": hp.choice("max_features", ["log2", "sqrt"]),
    }

    def hyperopt_train_test(params):
        model = RandomForestClassifier(**params, n_jobs=-1)
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
        rf_feature_space,
        algo=tpe.suggest,
        max_evals=25,
        early_stop_fn=no_progress_loss(5),
        trials=trials,
    )
    best_params = hyperopt.space_eval(rf_feature_space, best_params)

    print(best_params)

    clf = RandomForestClassifier(**best_params, n_jobs=-1)
    clf.fit(X_train, y_train)

    return clf


def train_classifier_XGBoost(X_train, y_train):
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

    print(best_params)

    xgb_cl = XGBClassifier(
        verbosity=0, objective=obj, eval_metric=ev_metric, **best_params
    )
    xgb_cl.fit(X_train, y_train)

    return xgb_cl

    """evalset = [(X_train, y_train), (X_test, y_test)]
    xgb_cl.fit(X_train, y_train, eval_set=evalset, verbose = 0)
    
    results = xgb_cl.evals_result()
    epochs = len(results['validation_0'][ev_metric[0]])
    x_axis = range(0, epochs)
    
    for metric in ev_metric:

        fig, ax = plt.subplots()

        ax.plot(x_axis, results['validation_1'][metric], label='Test')
        ax.legend()
        plt.xlabel('Number of trees')
        plt.ylabel(metric)
        plt.title('XGBoost {}'.format(metric))
        plt.savefig('XGBoost_training_{}.png'.format(metric), bbox_inches='tight', dpi=600)
        plt.close()
        #plt.show(False)
    
    return xgb_cl"""


def read_DDOS_dataset(selected_features, binary=True):
    df = pd.read_csv(
        "./data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
        delimiter=",",
    )
    new_cols = {col: col.strip() for col in df.columns}
    df.rename(columns=new_cols, inplace=True)

    # clean up
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    for feature in selected_features:
        df = df[df[feature] >= 0]

    selected_features.append("Label")
    df = df[selected_features]

    # keep the chosen attacks
    attacks_to_detect = [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
    ]
    labels_to_keep = attacks_to_detect
    labels_to_keep.append("BENIGN")
    relabelled_df = df[df["Label"].isin(labels_to_keep)]

    # set labels to 0/1
    rel_dict = {}
    if binary:
        rel_dict = {att_name: 1 for att_name in attacks_to_detect}
        rel_dict["BENIGN"] = 0
        relabelled_df.replace({"Label": rel_dict}, inplace=True)
    else:
        rel_dict = {att_name: i + 1 for i, att_name in enumerate(attacks_to_detect)}
        rel_dict["BENIGN"] = 0
        relabelled_df.replace({"Label": rel_dict}, inplace=True)

    # Reset the index of the DataFrame after filtering
    relabelled_df = relabelled_df.reset_index(drop=True)

    # balance the dataset
    balanced_df = pd.DataFrame()

    samples_per_class = 5000
    for k, v in rel_dict.items():
        print(f"Going through {k}:{v}")
        class_df = relabelled_df[relabelled_df["Label"] == v]
        random_indices = np.random.choice(
            class_df.index, samples_per_class, replace=False
        )
        balanced_df = pd.concat([balanced_df, class_df.loc[random_indices]], axis=0)

    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    print(balanced_df["Label"].value_counts())

    return balanced_df


def accuracy_metrics(y_test, y_pred):
    print("Accuracy: {}".format(round(mt.accuracy_score(y_test, y_pred), 2)))
    print(
        "Precision: {}".format(round(mt.precision_score(y_test, y_pred, labels=[1]), 2))
    )
    print("Recall: {}".format(round(mt.recall_score(y_test, y_pred, labels=[1]), 2)))
    print("F1score: {}".format(round(mt.f1_score(y_test, y_pred, labels=[1]), 2)))

    # cm = mt.confusion_matrix(y_test, y_pred, labels = lab)
    # print(cm)


if __name__ == "__main__":
    features = [
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Mean",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Mean",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Min Packet Length",
        "Max Packet Length",
        "Packet Length Mean",
    ]

    df_ddos = read_DDOS_dataset(features, binary=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df_ddos.drop(columns=["Label"]), df_ddos["Label"], test_size=0.3
    )

    clf, metrics = train_multinary_classier_XGBoost(X_train, y_train, X_test, y_test)
    # clf = train_classifier_RF(X_train, y_train)

    print(f"Evaluation of model resulted in: {pretty_print(metrics)}")
    # accuracy_metrics(y_test, clf.predict(X_test))

    # Save the model for later
    print("Saving Model")
    joblib.dump(clf, "multinary_detection_model.joblib")
