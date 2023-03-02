import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from keras.optimizers import adam_v2
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)


def get_random_parameters(
    name: str,
    n_iter: int = 100,
    random_state: int = 42,
    out_path: Path = None,
):
    """
    create random parameters dictionary for algorithms.
    """
    np.random.seed(seed=random_state)
    hp = {
        "l1": {
            "penalty": ["l1"],
            "C": st.reciprocal(a=1e-5, b=0.9),
            "solver": ["saga"],
            "max_iter": [10000],
        },
        "RF": {
            "n_estimators": st.randint(50, 600),
            "max_features": st.truncnorm(a=0, b=1, loc=0.45, scale=0.1),
            "max_depth": st.randint(4, 15),
            # uniform distribution from 0.01 to 0.2 (1%-20% of obs)
            "min_samples_split": st.uniform(0.01, 0.2),
            # 'RF_min_samples_leaf': st.uniform(0.1, 0.5),
            "bootstrap": [True, False],
        },
        "XGBoost": {
            "n_estimators": st.randint(50, 600),
            "max_depth": st.randint(4, 15),
            # "min_samples_split": st.uniform(0.01, 0.2),
            # "min_samples_leaf": st.uniform(0.1, 0.5),
            # "max_features": st.truncnorm(a=0, b=1, loc=0.45, scale=0.1),
            "learning_rate": st.uniform(0.001, 0.59),
            "colsample_bytree": st.beta(10, 1),
            "subsample": st.beta(10, 1),
            "gamma": st.uniform(0, 10),
            "objective": ["binary:logistic"],
            "scale_pos_weight": st.randint(0, 2),
            "min_child_weight": st.expon(0, 50),
            "num_class": [1],
        },
        "NN": {
            "batch_size": st.randint(20, 200),
            "epochs": st.randint(10, 100),
            "learn_rate": st.uniform(0.000001, 0.29),
            # 'momentum': st.uniform(0.1, 0.8),
            "beta_1": st.uniform(0.7, 0.199),
            "beta_2": st.uniform(0.7, 0.199),
            "activation": ["softmax", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"],
            "dropout_rate": st.uniform(0.1, 0.4),
            "neurons1": st.randint(30, 100),
            "neurons2": [30],
            "n_layers": [1, 2],
        },
    }

    random_hp = {}
    for model_name in hp:
        random_hp[model_name] = list(ParameterSampler(hp[model_name], n_iter=n_iter, random_state=random_state))
    # add 'neurons2 based on neurons1
    for param_dict in random_hp["NN"]:
        param_dict["neurons2"] = round(param_dict["neurons1"] / 2)
        # param_dict["optimizer"] = adam_v2.Adam(
        #     learning_rate=param_dict["learn_rate"], beta_1=param_dict["beta_1"], beta_2=param_dict["beta_2"]
        # )
    (out_path / name).mkdir(parents=True, exist_ok=True)
    if not (out_path / name / "random_hp.pkl").exists():
        with open(out_path / name / "random_hp.pkl", "wb") as f:
            pickle.dump(random_hp, f)
    return random_hp


def get_data(csv_path: str):
    df = pd.read_csv(csv_path, parse_dates=["Culture.Time.Date"], index_col=["Culture.Time.Date"])
    cols_to_drop = ["TimeSinceEndLastUse365", "RunningDays", "T.Since.Comorbidity.Scores"]
    df = df.drop(cols_to_drop, axis=1)
    df = df.fillna(0)
    for col in df.columns:
        if 0 in df[col].unique() and 1 in df[col].unique() and df[col].nunique() == 2:
            df[col] = df[col].astype("int8")
    float16_cols = [
        "Age.At.Culture.Time.Date",
        "CumAnyUsage365",
        "Length.Days.Current.Hospitalization",
        "Length.Days.Hospitalized.Past.365.Days",
        "PrevAntiUsageBin181_365",
        "PrevAntiUsageBin60",
        "PrevAntiUsageBin61_180",
        "PrevAntiUsageFamilyBin181_365",
        "PrevAntiUsageFamilyBin60",
        "PrevAntiUsageFamilyBin61_180",
        "PrevAntiUsageOtherBin181_365",
        "PrevAntiUsageOtherBin60",
        "PrevAntiUsageOtherBin61_180",
        "PrevAntiUsageOtherFamilyBin181_365",
        "PrevAntiUsageOtherFamilyBin60",
        "PrevAntiUsageOtherFamilyBin61_180",
        "PrevAntiUsageSaneFamilyOtherAntiBin181_365",
        "PrevAntiUsageSaneFamilyOtherAntiBin60",
        "PrevAntiUsageSaneFamilyOtherAntiBin61_180",
        "PreviouslyResistantOtherAntiSameFamilyBin180_365",
        "SlidingResAny30",
        "SlidingResSame30",
        "SlidingResUnits30",
        "T.Since.Comorbidity.Scores",
        "SlidingResSameBacAnyAnti30",
    ]
    for col in float16_cols:
        try:
            df[col] = df[col].astype("float16")
        except KeyError:
            continue
    return df.sort_index()


def split_and_scale(df: pd.DataFrame):
    X = df.drop(["Resistance"], axis=1)
    # keep non-zero colunms
    features = []
    for col in X:
        if X[col].sum() > 0:
            features.append(col)
    X = X[features]
    y = df["Resistance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # keep column names befor standartizimg
    col_names = X_train.columns

    # standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, col_names


def get_transformed_X_y(train_index, test_index, X_train, y_train):
    """
    create scaled_x_train, y_train, scaled_X_test, y_test by fold limits.
    fold limits are from time series split (time series CV)
    """
    # get fold indices
    # fold_train_start = 0
    fold_train_end = train_index[-1] + 1
    fold_test_start = test_index[0]
    fold_test_end = test_index[-1] + 1
    scaled_X_train = X_train[0:fold_train_end]
    fold_y_train = y_train[0:fold_train_end]
    scaled_X_test = X_train[fold_test_start:fold_test_end]
    fold_y_test = y_train[fold_test_start:fold_test_end]
    return scaled_X_train, fold_y_train, scaled_X_test, fold_y_test
