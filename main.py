# imports
import inspect
import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# other
import scipy.stats as st

# from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from keras import layers
from keras.layers import BatchNormalization, Dropout
from keras.optimizers import adam_v2
from sklearn.ensemble import RandomForestClassifier

# models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# tuning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tensorflow import keras
from xgboost import plot_tree

from plots import plot_calibration, plot_desicion_curve, plot_roc
from utils import get_data, get_random_parameters, get_transformed_X_y, split_and_scale
import tensorflow as tf

warnings.filterwarnings("ignore")

import random

random_state = 42
np.random.seed(seed=random_state)
import os

os.environ["PYTHONHASHSEED"] = "0"
random.seed(random_state)
tf.random.set_seed(random_state)


def get_script_dir() -> Path:
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    return Path(path)


class EnsembleSingleAnti:
    """
    This class is intended to create an instance of ensemble learner for a dataframe of a single tested antibiotics.
    this is a workaround of an issue with other ensemble packages that can't deal with time series split
    (while cross validating to create the z_matrix the number of predictions is smaller than the whole data set
    because the first fold is used for training only
    """

    def __init__(self, name: str, df: pd.DataFrame, out_path: Path, n_iter=20):
        """
        n_iter is for Random Search of Hyper-Parameters
        """
        self.name = name
        self.df = df
        self.out_path = out_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.col_names = None
        #  create a dictionary of algorithms
        self.clfs = {
            "NN": None,
            "l1": LogisticRegression(),
            "RF": RandomForestClassifier(),
            "XGBoost": xgb.XGBClassifier(),
        }
        self.n_iter = n_iter
        self.cv_randomsearch_results_df = pd.DataFrame(columns=["model", "fold_n", "iteration", "params", "auc"])
        # Z dict (to be matrix) for saving predictions for training meta learner
        self.z_dict = {"NN": [], "l1": [], "RF": [], "XGBoost": [], "true": []}
        self.z_matrix = None
        self.test_dict = {"NN": [], "l1": [], "RF": [], "XGBoost": [], "true": []}
        self.test_df = None
        self.meta_learner = LogisticRegression(penalty="l1", solver="liblinear")
        self.best_params = {"NN": None, "l1": None, "RF": None, "XGBoost": None}
        self.meta_learner_coef_dict = {}
        # call all methods
        if (self.out_path / "z_matrix.csv").is_file():
            self.z_matrix = pd.read_csv(self.out_path / "z_matrix.csv")
            self.best_params = pickle.load(open(self.out_path / "best_params.pkl", "rb"))
        else:
            self.create_z_matrix()
        self.train_meta_learner()
        print(self.X_train.shape)
        self.train_all_models()
        self.save_models()

    def random_search(self, model, fold_n, _iter, scaled_X_train, fold_y_train, scaled_X_test, fold_y_test, params):
        """
        sets paarams to model, fits, trains, predicts, scores and save results in self.cv_randomsearch_results_df
        """
        if model == "NN":
            self.clfs[model] = keras.Sequential(
                [
                    keras.layers.Dense(
                        units=params["NN"][_iter]["neurons1"],
                        input_shape=(scaled_X_train.shape[1],),
                        activation=params["NN"][_iter]["activation"],
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                        bias_initializer=keras.initializers.Constant(value=0.1),
                    )
                ]
            )
            self.clfs[model].add(BatchNormalization())
            self.clfs[model].add(Dropout(params["NN"][_iter]["dropout_rate"]))
            if params["NN"][_iter]["n_layers"] == 2:
                self.clfs[model].add(
                    layers.Dense(
                        units=params["NN"][_iter]["neurons2"],
                        activation=params["NN"][_iter]["activation"],
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                        bias_initializer=keras.initializers.Constant(value=0.1),
                    )
                )
                self.clfs[model].add(Dropout(params["NN"][_iter]["dropout_rate"]))
            self.clfs[model].add(
                layers.Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                    bias_initializer=keras.initializers.Constant(value=0.1),
                )
            )

            opt = adam_v2.Adam(
                learning_rate=params["NN"][_iter]["learn_rate"],
                beta_1=params["NN"][_iter]["beta_1"],
                beta_2=params["NN"][_iter]["beta_1"],
            )
            self.clfs[model].compile(loss="binary_crossentropy", optimizer=opt, metrics=[keras.metrics.AUC()])

            # fit model
            self.clfs[model].fit(
                scaled_X_train,
                fold_y_train,
                batch_size=params["NN"][_iter]["batch_size"],
                epochs=params["NN"][_iter]["epochs"],
                shuffle=False,
                verbose=False,
            )
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            fold_pred = np.concatenate(fold_pred).ravel().tolist()
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[len(self.cv_randomsearch_results_df) + 1] = [
                model,
                fold_n,
                _iter,
                params["NN"][_iter],
                auc_score,
            ]
        elif model == "XGBoost":
            self.clfs[model].set_params(**params[model][_iter])
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train, eval_metric="auc")
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[len(self.cv_randomsearch_results_df) + 1] = [
                model,
                fold_n,
                _iter,
                self.clfs[model].get_params(),
                auc_score,
            ]
        else:
            self.clfs[model].set_params(**params[model][_iter])
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train)
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[len(self.cv_randomsearch_results_df) + 1] = [
                model,
                fold_n,
                _iter,
                self.clfs[model].get_params(),
                auc_score,
            ]

    def predict_fold(self, model, fold_n, scaled_X_train, fold_y_train, scaled_X_test):
        # get parameters with highest AUC
        best_iter_params = (
            self.cv_randomsearch_results_df.query("model==@model & fold_n==@fold_n")
            .sort_values(by="auc", ascending=False)["params"]
            .values[0]
        )
        if model == "NN":
            self.clfs[model] = keras.Sequential(
                [
                    keras.layers.Dense(
                        units=best_iter_params["neurons1"],
                        input_shape=(scaled_X_train.shape[1],),
                        activation=best_iter_params["activation"],
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                        bias_initializer=keras.initializers.Constant(value=0.1),
                    )
                ]
            )
            self.clfs[model].add(BatchNormalization())
            self.clfs[model].add(Dropout(best_iter_params["dropout_rate"]))
            if best_iter_params["n_layers"] == 2:
                self.clfs[model].add(
                    layers.Dense(
                        units=best_iter_params["neurons2"],
                        activation=best_iter_params["activation"],
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                        bias_initializer=keras.initializers.Constant(value=0.1),
                    )
                )
                self.clfs[model].add(Dropout(best_iter_params["dropout_rate"]))

            self.clfs[model].add(
                layers.Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                    bias_initializer=keras.initializers.Constant(value=0.1),
                )
            )
            opt = adam_v2.Adam(
                learning_rate=best_iter_params["learn_rate"],
                beta_1=best_iter_params["beta_1"],
                beta_2=best_iter_params["beta_1"],
            )
            self.clfs[model].compile(loss="binary_crossentropy", optimizer=opt, metrics=[keras.metrics.AUC()])

            # fit model
            self.clfs[model].fit(
                scaled_X_train,
                fold_y_train,
                batch_size=best_iter_params["batch_size"],
                epochs=best_iter_params["epochs"],
                shuffle=False,
                verbose=False,
            )
            # create predictions with best parameters for Z matrix
            # there won't be predictions for the first training fold due to TimeSeriesSplit CV
            fold_pred = self.clfs[model].predict(scaled_X_test)
            fold_pred = np.concatenate(fold_pred).ravel().tolist()
            self.z_dict[model].extend(x for x in fold_pred)
        elif model == "XGBoost":
            self.clfs[model].set_params(**best_iter_params)
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train, eval_metric="auc")
            fold_pred = self.clfs[model].predict_proba(scaled_X_test)
            self.z_dict[model].extend(x for x in fold_pred[:, 1])  # keep probabilities for the positive outcome only

        else:
            self.clfs[model].set_params(**best_iter_params)
            self.clfs[model].fit(scaled_X_train, fold_y_train)
            fold_pred = self.clfs[model].predict_proba(scaled_X_test)
            self.z_dict[model].extend(x for x in fold_pred[:, 1])  # keep probabilities for the positive outcome only

    def create_z_matrix(self):
        """
        create a z_matrix, based on Van der Laan et al (2007).
        Super Learner. doi:10.2202/1544-6115.1309

        :return: a dataframe (z matrix) of predictions and the true value
                used later for training the meta learner
        """

        print('creating a "z-matrix"')
        self.X_train, self.X_test, self.y_train, self.y_test, self.col_names = split_and_scale(self.df)
        # get random parameters (for manual random search)
        params = get_random_parameters(self.name, out_path=self.out_path, n_iter=self.n_iter)
        tscv = TimeSeriesSplit(n_splits=3)
        true_extend_count = 0
        # iterate over models to find best hyper parameters by random
        print("Iterating over models...")
        for model in self.clfs.keys():
            print("model:", model)
            # iterate over folds
            for fold_n, (train_index, test_index) in enumerate(tscv.split(self.X_train)):
                # print(' fold:', fold_n)
                # selecting X and y train and test, scaling
                scaled_X_train, fold_y_train, scaled_X_test, fold_y_test = get_transformed_X_y(
                    train_index, test_index, self.X_train, self.y_train
                )
                # iterate over random search parameters
                for _iter in range(self.n_iter):
                    self.random_search(
                        model, fold_n, _iter, scaled_X_train, fold_y_train, scaled_X_test, fold_y_test, params
                    )
                # predict fold with best params
                self.predict_fold(model, fold_n, scaled_X_train, fold_y_train, scaled_X_test)
                if true_extend_count < tscv.get_n_splits():
                    self.z_dict["true"].extend(x for x in fold_y_test.values)
                    true_extend_count += 1
        # create Z matrix from z_dict
        self.z_matrix = pd.DataFrame.from_dict(self.z_dict)
        (self.out_path / self.name).mkdir(parents=True, exist_ok=True)
        self.z_matrix.to_csv(self.out_path / self.name / "z_matrix.csv", index=False)
        for model in self.clfs.keys():
            self.best_params[model] = self.get_best_params(model)
        with open(self.out_path / self.name / "best_params.pkl", "wb") as f:
            pickle.dump(self.best_params, f)

    def train_meta_learner(self):
        print("training meta learner")
        meta_x = self.z_matrix.drop("true", axis=1)
        meta_y = self.z_matrix["true"]
        c_values = {"C": st.uniform(loc=0, scale=4)}
        clf = RandomizedSearchCV(self.meta_learner, c_values, scoring="roc_auc", random_state=random_state, n_jobs=-1)
        search = clf.fit(meta_x, meta_y)
        best_params = search.best_params_
        self.meta_learner.set_params(**best_params)
        self.meta_learner.fit(meta_x, meta_y)
        print("meta learner coefficients:")
        for coef, feat in zip(self.meta_learner.coef_[0, :], meta_x.columns):
            self.meta_learner_coef_dict[feat] = coef
        print("coef dict:", self.meta_learner_coef_dict)

    def get_best_params(self, model) -> dict:
        """
        from self.cv_randomsearch_results_df
        """
        grouped = self.cv_randomsearch_results_df.query("model==@model").groupby("iteration").mean()
        best_fold = grouped.sort_values(by="auc", ascending=False).index[0]
        best_params = self.cv_randomsearch_results_df.query("model==@model & iteration==@best_fold")["params"].values[
            0
        ]
        return best_params

    def train_all_models(self):
        """
        train all models on train set. this is step zero in Van der Laan's paper "Super Learner".
        doi:10.2202/1544-6115.1309
        :return:
        """
        print("training all models (step 0)")
        for model in self.clfs.keys():
            if model == "NN":
                self.clfs[model] = keras.Sequential(
                    [
                        keras.layers.Dense(
                            units=self.best_params[model]["neurons1"],
                            input_shape=(self.X_train.shape[1],),
                            activation=self.best_params[model]["activation"],
                            kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                            bias_initializer=keras.initializers.Constant(value=0.1),
                        )
                    ]
                )
                self.clfs[model].add(BatchNormalization())
                self.clfs[model].add(Dropout(self.best_params[model]["dropout_rate"]))
                if self.best_params[model]["n_layers"] == 2:
                    self.clfs[model].add(
                        layers.Dense(
                            units=self.best_params[model]["neurons2"],
                            activation=self.best_params[model]["activation"],
                            kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                            bias_initializer=keras.initializers.Constant(value=0.1),
                        )
                    )
                    self.clfs[model].add(Dropout(self.best_params[model]["dropout_rate"]))

                self.clfs[model].add(
                    layers.Dense(
                        1,
                        activation="sigmoid",
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_state),
                        bias_initializer=keras.initializers.Constant(value=0.1),
                    )
                )
                opt = adam_v2.Adam(
                    learning_rate=self.best_params[model]["learn_rate"],
                    beta_1=self.best_params[model]["beta_1"],
                    beta_2=self.best_params[model]["beta_1"],
                )
                self.clfs[model].compile(loss="binary_crossentropy", optimizer=opt, metrics=[keras.metrics.AUC()])

                # fit model
                self.clfs[model].fit(
                    self.X_train,
                    self.y_train,
                    batch_size=self.best_params[model]["batch_size"],
                    epochs=self.best_params[model]["epochs"],
                    shuffle=False,
                    verbose=False,
                )
            elif model == "XGBoost":
                self.clfs[model].set_params(**self.best_params[model])
                self.clfs[model].fit(self.X_train, self.y_train, eval_metric="auc")
            else:
                self.clfs[model].set_params(**self.best_params[model])
                self.clfs[model].fit(self.X_train, self.y_train)
                if model == "l1":
                    coeffs = self.clfs[model].coef_
                    coeffs_df = pd.DataFrame(columns=self.col_names, data=coeffs)
                    dir_name = self.out_path / self.name
                    dir_name.mkdir(parents=True, exist_ok=True)
                    coeffs_df.to_csv(dir_name / "l1_coeffs.csv")

    def save_models(self):
        for model in self.clfs.keys():
            if model == "NN":
                self.clfs[model].save(self.out_path / self.name / f"{model}.h5")
                self.clfs[model].save_weights(self.out_path / self.name / f"{model}_weights.h5")
            else:
                with open(self.out_path / self.name / f"{model}.pkl", "wb") as f:
                    pickle.dump(self.clfs[model], f)
        with open(self.out_path / self.name / "ensemble.pkl", "wb") as f:
            pickle.dump(self.meta_learner, f)


if "__main__" == __name__:
    csv_paths = {
        "Agnostic": get_script_dir() / "agnostics_22_06_22.csv",
        "Gnostic": get_script_dir() / "gnostics_22_06_22.csv",
    }

    out_path = get_script_dir() / "results"
    for name, csv_path in csv_paths.items():
        print(name)
        df = get_data(csv_path)
        by_bac = EnsembleSingleAnti(name, df, out_path, n_iter=300)

    print("**********************")
    print("models and ensemble saved!!!")
    print("**********************")
