# imports
import warnings
import pandas as pd
import numpy as np
# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# tuning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# other
from sklearn.model_selection import train_test_split
import scipy.stats as st
from sklearn.model_selection import ParameterSampler
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from xgboost import plot_tree
from sklearn.metrics import brier_score_loss
from scipy import stats
from pathlib import Path
import inspect
import os



warnings.filterwarnings("ignore")
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'font.size': 16})


import numpy as np



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

    def __init__(self, name, df, out_path, n_iter=20):
        """
        n_iter is set to 60 because of a research on the optimal number:
        Random Search for Hyper-Parameter Optimization
        https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
        :param df: pandas.core.frame.DataFrame
        :param n_iter: int
        :param name: str
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
        self.clfs = {'NN': None,
                     'l1': LogisticRegression(),
                     'RF': RandomForestClassifier(),
                     'XGBoost': xgb.XGBClassifier()}
        self.n_iter = n_iter
        self.cv_randomsearch_results_df = pd.DataFrame(columns=['model', 'fold_n', 'iteration', 'params', 'auc'])
        # Z dict (to be matrix) for saving predictions for training meta learner
        self.z_dict = {'NN': [], 'l1': [], 'RF': [], 'XGBoost': [], 'true': []}
        self.z_matrix = None
        self.test_dict = {'NN': [], 'l1': [], 'RF': [], 'XGBoost': [], 'true': []}
        self.test_df = None
        self.meta_learner = LogisticRegression(penalty='l1', solver='liblinear')
        self.best_params = {'NN': None, 'l1': None, 'RF': None, 'XGBoost': None}
        self.meta_learner_coef_dict = {}
        self.shap_values = None
        self.final_auc_results = {'NN': None, 'l1': None, 'RF': None, 'XGBoost': None, 'ensemble': None}
        self.final_fpr = {'NN': None, 'l1': None, 'RF': None, 'XGBoost': None, 'ensemble': None}
        self.final_tpr = {'NN': None, 'l1': None, 'RF': None, 'XGBoost': None, 'ensemble': None}
        self.ensemble_thresholds = None
        self.ensemble_y_true = None
        self.ensemble_y_score = None
        self.final_sensitivity = 0
        self.final_specificity = 0
        self.final_accuracy = 0
        self.final_ppv = 0
        # call all methods
        self.create_z_matrix()
        self.train_meta_learner()
        self.train_all_models()
        self.final_predictions_for_ensemble()
        self.test_models()
        self.save_data()
        self.plot_desicion_curve()
        self.plot_roc()
        self.plot_calibration()
        self.calc_shap_values()

    def get_random_parameters(self):
        """
        create random parameters dictionary for search space.
        :return: {'l1': [{'C': 0.001934,
                   'max_iter': 10000,
                   'penalty': 'l1',
                   'solver': 'saga'},
                  {'C': 0.23151244171,
                   'max_iter': 10000,
                   'penalty': 'l1',
                   'solver': 'saga'},
                 'RF':[{...}]
        :rtype: dict
        """
        # hyper parameters dict
        hp = {'l1': {'penalty': ['l1'],
                     'C': st.reciprocal(a=1e-5, b=0.9),
                     'solver': ['saga'],
                     'max_iter': [10000]},
              'RF': {'n_estimators': st.randint(50, 600),
                     # normally distributed, with mean .45 stddev 0.1, bounded between 0 and 1
                     'max_features': st.truncnorm(a=0, b=1, loc=0.45, scale=0.1),
                     'max_depth': st.randint(4, 15),
                     # uniform distribution from 0.01 to 0.2 (1%-20% of obs)
                     'min_samples_split': st.uniform(0.01, 0.2),
                     # 'RF__min_samples_leaf': st.uniform(0.1, 0.5),
                     'bootstrap': [True, False],
                     'n_jobs': [7]},
              'XGBoost': {'n_estimators': st.randint(50, 600),
                          'max_depth': st.randint(4, 15),
                          # 'min_samples_split': st.uniform(0.01, 0.2),
                          # 'min_samples_leaf': st.uniform(0.1, 0.5),
                          # 'max_features': st.truncnorm(a=0, b=1, loc=0.45, scale=0.1),
                          'learning_rate': st.uniform(0.001, 0.59),
                          "colsample_bytree": st.beta(10, 1),
                          "subsample": st.beta(10, 1),
                          "gamma": st.uniform(0, 10),
                          'objective': ['binary:logistic'],
                          'scale_pos_weight': st.randint(0, 2),
                          "min_child_weight": st.expon(0, 50),
                          "n_jobs": [7]
                          },
              'NN': {
                  'batch_size': st.randint(20, 200),
                  'epochs': st.randint(10, 100),
                  'learn_rate': st.uniform(0.000001, 0.29),
                  # 'momentum': st.uniform(0.1, 0.8),
                  'beta_1': st.uniform(0.7, 0.199),
                  'beta_2': st.uniform(0.7, 0.199),
                  'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',
                                 'linear'],
                  'dropout_rate': st.uniform(0.1, 0.4),
                  'neurons1': st.randint(30, 100),
                  'neurons2': [30],
                  'n_layers': [1, 2]}}

        random_hp = {}
        for model_name in hp:
            random_hp[model_name] = list(ParameterSampler(hp[model_name], n_iter=self.n_iter))
        # add 'neurons2 based on neurons1
        for param_dict in random_hp['NN']:
            param_dict['neurons2'] = round(param_dict['neurons1'] / 2)
            param_dict['optimizer'] = Adam(learning_rate=param_dict['learn_rate'],
                                           beta_1=param_dict['beta_1'],
                                           beta_2=param_dict['beta_2'])
        return random_hp

    def get_transformed_X_y(self, train_index, test_index):
        """
        create scaled_x_train, y_train, scaled_X_test, y_test by fold limits.
        fold limits are from time series split (time series CV)

        :param train_index: int
        :param test_index: int
        :return (scaled_X_train, fold_y_train, scaled_X_test, fold_y_test):
        :rtype: (pandas.core.frame.DataFrame, pandas.core.series.Series, pandas.core.frame.DataFrame, pandas.core.series.Series
        """
        # get fold indices
        # fold_train_start = 0
        fold_train_end = train_index[-1] + 1
        fold_test_start = test_index[0]
        fold_test_end = test_index[-1] + 1

        scaled_X_train = self.X_train[0:fold_train_end]
        fold_y_train = self.y_train[0:fold_train_end]
        scaled_X_test = self.X_train[fold_test_start:fold_test_end]
        fold_y_test = self.y_train[fold_test_start:fold_test_end]
        return scaled_X_train, fold_y_train, scaled_X_test, fold_y_test

    def random_search(self, model, fold_n, _iter, scaled_X_train, fold_y_train, scaled_X_test, fold_y_test, params):
        """
        sets paarams to model, fits, trains, predicts, scores and save results in self.cv_randomsearch_results_df
        :param model:
        :param fold_n: int
        :param _iter: int
        :param scaled_X_train: pandas.core.frame.DataFrame
        :param fold_y_train: pandas.core.series.Series
        :param scaled_X_test: pandas.core.frame.DataFrame
        :param fold_y_test: pandas.core.series.Series
        :param params: dict
        :return:
        """
        if model == 'NN':
            self.clfs[model] = keras.Sequential([keras.layers.Dense(units=params['NN'][_iter]['neurons1'],
                                                                    input_shape=(scaled_X_train.shape[1],),
                                                                    activation=params['NN'][_iter]['activation'])])
            self.clfs[model].add(BatchNormalization())
            self.clfs[model].add(Dropout(params['NN'][_iter]['dropout_rate']))
            if params['NN'][_iter]['n_layers'] == 2:
                self.clfs[model].add(layers.Dense(units=params['NN'][_iter]['neurons2'],
                                                  activation=params['NN'][_iter]['activation']))
                self.clfs[model].add(Dropout(params['NN'][_iter]['dropout_rate']))
            self.clfs[model].add(layers.Dense(1, activation="sigmoid"))

            opt = keras.optimizers.Adam(learning_rate=params['NN'][_iter]['learn_rate'],
                                        beta_1=params['NN'][_iter]['beta_1'],
                                        beta_2=params['NN'][_iter]['beta_1'])
            self.clfs[model].compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.AUC()])

            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train,
                                 batch_size=params['NN'][_iter]['batch_size'],
                                 epochs=params['NN'][_iter]['epochs'],
                                 shuffle=False,
                                 verbose=False)
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            fold_pred = np.concatenate(fold_pred).ravel().tolist()
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[
                len(self.cv_randomsearch_results_df) + 1] = [model,
                                                             fold_n, _iter,
                                                             params['NN'][_iter],
                                                             auc_score]
        elif model == 'XGBoost':
            self.clfs[model].set_params(**params[model][_iter])
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train, eval_metric='auc')
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[
                len(self.cv_randomsearch_results_df) + 1] = [model,
                                                             fold_n, _iter,
                                                             self.clfs[model].get_params(),
                                                             auc_score]
        else:
            self.clfs[model].set_params(**params[model][_iter])
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train)
            # predict
            fold_pred = self.clfs[model].predict(scaled_X_test)
            # calculate AUC
            auc_score = roc_auc_score(fold_y_test, fold_pred)
            # save results on next line
            self.cv_randomsearch_results_df.loc[
                len(self.cv_randomsearch_results_df) + 1] = [model,
                                                             fold_n, _iter,
                                                             self.clfs[model].get_params(),
                                                             auc_score]

    def predict_fold(self, model, fold_n, scaled_X_train, fold_y_train, scaled_X_test):
        """

        :param model:
        :param fold_n: int
        :param scaled_X_train: pandas.core.frame.DataFrame
        :param fold_y_train: pandas.core.series.Series
        :param scaled_X_test: pandas.core.frame.DataFrame
        :return:
        """
        # get parameters with highest AUC
        best_iter_params = self.cv_randomsearch_results_df.query(
            'model==@model & fold_n==@fold_n').sort_values(by='auc', ascending=False)['params'].values[0]
        if model == 'NN':
            self.clfs[model] = keras.Sequential([keras.layers.Dense(units=best_iter_params['neurons1'],
                                                                    input_shape=(scaled_X_train.shape[1],),
                                                                    activation=best_iter_params['activation'])])
            self.clfs[model].add(BatchNormalization())
            self.clfs[model].add(Dropout(best_iter_params['dropout_rate']))
            if best_iter_params['n_layers'] == 2:
                self.clfs[model].add(layers.Dense(units=best_iter_params['neurons2'],
                                                  activation=best_iter_params['activation']))
                self.clfs[model].add(Dropout(best_iter_params['dropout_rate']))

            self.clfs[model].add(layers.Dense(1, activation="sigmoid"))
            opt = keras.optimizers.Adam(learning_rate=best_iter_params['learn_rate'],
                                        beta_1=best_iter_params['beta_1'],
                                        beta_2=best_iter_params['beta_1'])
            self.clfs[model].compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.AUC()])

            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train,
                                 batch_size=best_iter_params['batch_size'],
                                 epochs=best_iter_params['epochs'],
                                 shuffle=False,
                                 verbose=False)
            # create predictions with best parameters for Z matrix
            # there won't be predictions for the first training fold due to TimeSeriesSplit CV
            fold_pred = self.clfs[model].predict(scaled_X_test)
            fold_pred = np.concatenate(fold_pred).ravel().tolist()
            self.z_dict[model].extend(x for x in fold_pred)
        elif model == 'XGBoost':
            self.clfs[model].set_params(**best_iter_params)
            # fit model
            self.clfs[model].fit(scaled_X_train, fold_y_train, eval_metric='auc')
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
        :rtype: pandas.core.frame.DataFrame
        """

        print('creating a "z-matrix"')
        # get random parameters (for manual random search)
        params = self.get_random_parameters()
        X = self.df.drop(['Resistance'], axis=1)
        # keep non-zero colunms
        features = []
        for col in X:
            if X[col].sum() > 0:
                features.append(col)
        X = X[features]
        y = self.df['Resistance']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=42)
        # keep column names befor standartizimg
        self.col_names = self.X_train.columns

        # standardization
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        tscv = TimeSeriesSplit(n_splits=5)
        true_extend_count = 0
        # iterate over models to find best hyper parameters by random
        print('Iterating over models...')
        for model in self.clfs.keys():
            print('model:', model)
            # iterate over folds
            for fold_n, (train_index, test_index) in enumerate(tscv.split(self.X_train)):
                # selecting X and y train and test, scaling
                scaled_X_train, fold_y_train, scaled_X_test, fold_y_test = self.get_transformed_X_y(
                    train_index, test_index)
                # iterate over random search parameters
                for _iter in range(self.n_iter):
                    self.random_search(model, fold_n, _iter, scaled_X_train, fold_y_train, scaled_X_test, fold_y_test,
                                       params)
                # predict fold with best params
                self.predict_fold(model, fold_n, scaled_X_train, fold_y_train, scaled_X_test)
                if true_extend_count < tscv.get_n_splits():
                    self.z_dict['true'].extend(x for x in fold_y_test.values)
                    true_extend_count += 1
        # create Z matrix from z_dict
        self.z_matrix = pd.DataFrame.from_dict(self.z_dict)

    def train_meta_learner(self):
        print('training meta learner')
        meta_x = self.z_matrix.drop('true', axis=1)
        meta_y = self.z_matrix['true']
        c_values = {'C': st.uniform(loc=0, scale=4)}
        clf = RandomizedSearchCV(self.meta_learner, c_values, scoring='roc_auc', random_state=1)
        search = clf.fit(meta_x, meta_y)
        best_params = search.best_params_
        self.meta_learner.set_params(**best_params)
        self.meta_learner.fit(meta_x, meta_y)
        print('meta learner coefficients:')
        for coef, feat in zip(self.meta_learner.coef_[0, :], meta_x.columns):
            self.meta_learner_coef_dict[feat] = coef
        print('coef dict:', self.meta_learner_coef_dict)

    def get_best_params(self, model):
        """
        from self.cv_randomsearch_results_df
        :param model:
        :return: dict
        """
        grouped = self.cv_randomsearch_results_df.query('model==@model').groupby('iteration').mean()
        best_fold = grouped.sort_values(by='auc', ascending=False).index[0]
        best_params = self.cv_randomsearch_results_df.query('model==@model & iteration==@best_fold')['params'].values[0]
        return best_params

    def train_all_models(self):
        """
        train all models on train set. this is step zero in Van der Laan's paper "Super Learner".
        doi:10.2202/1544-6115.1309
        :return:
        """
        print('training all models (step 0)')
        for model in self.clfs.keys():
            self.best_params[model] = self.get_best_params(model)
            if model == 'NN':
                self.clfs[model] = keras.Sequential([keras.layers.Dense(units=self.best_params[model]['neurons1'],
                                                                        input_shape=(self.X_train.shape[1],),
                                                                        activation=self.best_params[model][
                                                                            'activation'])])
                self.clfs[model].add(BatchNormalization())
                self.clfs[model].add(Dropout(self.best_params[model]['dropout_rate']))
                if self.best_params[model]['n_layers'] == 2:
                    self.clfs[model].add(layers.Dense(units=self.best_params[model]['neurons2'],
                                                      activation=self.best_params[model]['activation']))
                    self.clfs[model].add(Dropout(self.best_params[model]['dropout_rate']))

                self.clfs[model].add(layers.Dense(1, activation="sigmoid"))
                opt = keras.optimizers.Adam(learning_rate=self.best_params[model]['learn_rate'],
                                            beta_1=self.best_params[model]['beta_1'],
                                            beta_2=self.best_params[model]['beta_1'])
                self.clfs[model].compile(loss='binary_crossentropy', optimizer=opt, metrics=[keras.metrics.AUC()])

                # fit model
                self.clfs[model].fit(self.X_train, self.y_train,
                                     batch_size=self.best_params[model]['batch_size'],
                                     epochs=self.best_params[model]['epochs'],
                                     shuffle=False,
                                     verbose=False)
            elif model == 'XGBoost':
                self.clfs[model].set_params(**self.best_params[model])
                self.clfs[model].fit(self.X_train, self.y_train, eval_metric='auc')
            else:
                self.clfs[model].set_params(**self.best_params[model])
                self.clfs[model].fit(self.X_train, self.y_train)
                if model == 'l1':
                    coeffs = self.clfs[model].coef_
                    coeffs_df = pd.DataFrame(columns=self.col_names, data=coeffs)
                    dir_name = self.out_path / self.name
                    dir_name.mkdir(parents=True, exist_ok=True)
                    coeffs_df.to_csv(dir_name / 'l1_coeffs.csv')

    def final_predictions_for_ensemble(self):
        print('getting final predictions')
        for model in self.clfs.keys():
            if model != 'NN':
                predictions = self.clfs[model].predict_proba(self.X_test)[:, 1]
            else:
                predictions = self.clfs[model].predict(self.X_test)
                predictions = np.concatenate(predictions).ravel().tolist()
            self.test_dict[model].extend(x for x in predictions)
        self.test_dict['true'].extend(x for x in self.y_test.values)
        self.test_df = pd.DataFrame.from_dict(self.test_dict)
        self.test_df['ensemble_pred'] = self.meta_learner.predict_proba(self.test_df.drop('true', axis=1))[:, 1]

    def test_models(self):
        print('testing models...')
        for model in self.clfs.keys():
            probs = self.test_df[model]
            # calculate scores
            auc = roc_auc_score(self.test_df['true'], probs)
            fpr, tpr, thresholds = roc_curve(self.y_test.values, probs)
            self.final_fpr[model] = list(fpr)
            self.final_tpr[model] = list(tpr)
            
            self.ensemble_y_true = self.y_test.values
            self.ensemble_y_score = probs.values

            print(f'{model} AUC on test: {auc}')
            self.final_auc_results[model] = round(auc, 3)

        # keep probabilities for the positive outcome only
        probs = self.test_df['ensemble_pred']
        # calculate scores
        self.test_df['rounded_proba'] = list(map(lambda x: round(x), probs))
        self.test_df['TP'] = self.test_df.apply(lambda x: 1 if (x['rounded_proba'] == 1 and x['true'] == 1) else 0,
                                                axis=1)
        self.test_df['TN'] = self.test_df.apply(lambda x: 1 if (x['rounded_proba'] == 0 and x['true'] == 0) else 0,
                                                axis=1)
        self.test_df['FP'] = self.test_df.apply(lambda x: 1 if (x['rounded_proba'] == 1 and x['true'] == 0) else 0,
                                                axis=1)
        self.final_sensitivity = self.test_df['TP'].sum() / self.test_df['true'].sum()
        self.final_specificity = self.test_df['TN'].sum() / len(self.test_df[self.test_df['true'] == 0])
        self.final_accuracy = (self.test_df['TP'].sum() + self.test_df['TN'].sum()) / len(self.test_df)
        self.final_ppv = self.test_df['TP'].sum() / self.test_df['rounded_proba'].sum()
        print(f'sensitivity:, {self.final_sensitivity}, specificity:{self.final_specificity}')
        print(f'accuracy: {self.final_accuracy}, ppv:{self.final_ppv}')
        auc = roc_auc_score(self.test_df['true'], probs)
        fpr, tpr, thresholds = roc_curve(self.y_test.values, probs)
        self.final_fpr['ensemble'] = list(fpr)
        self.final_tpr['ensemble'] = list(tpr)
        self.ensemble_thresholds = thresholds
        print('ensemble AUC on test:', auc)
        self.final_auc_results['ensemble'] = round(auc, 3)

    def save_data(self):
        """
        saves a csv file with the data for comparing the models and plotting ROC-AUC curves
        :return:
        """
        rows = []
        for clf in self.clfs.keys():
            row = [self.name, clf, self.best_params[clf], 'None', self.final_auc_results[clf], 'None', 'None', 'None',
                   'None',
                   self.final_fpr[clf], self.final_tpr[clf], list(self.test_df[clf]), None, None, None]
            rows.append(row)
        ensemble_row = [self.name, 'ensemble', self.meta_learner.get_params(), self.meta_learner_coef_dict,
                        self.final_auc_results['ensemble'], self.final_sensitivity, self.final_specificity,
                        self.final_accuracy, self.final_ppv,
                        self.final_fpr['ensemble'], self.final_tpr['ensemble'],
                        self.test_df['ensemble_pred'].tolist(),
                        self.ensemble_thresholds,
                        self.ensemble_y_true,
                        self.ensemble_y_score,
                        ]
        rows.append(ensemble_row)
        cols = ['antibiotic', 'model', 'params', 'coefs', 'auc',
                'sensitivity', 'specificity', 'accuracy', 'ppv',
                'fpr', 'tpr', 'probabiliteis', 'ensemble_y_true', 'ensemble_y_score', 'ensemble_thresholds']
        data = pd.DataFrame(rows, columns=cols)
        data.to_csv(self.out_path / (str(self.name) + '.csv'))
        print('data saved')
        print('**********')


    def plot_roc(self):
        fig = plt.figure(figsize=(8, 8))
        colors = ['orange', 'cyan', 'r', 'dodgerblue', 'k']
        line_style = ['solid', 'solid', 'solid', 'solid', 'dashed']
        alpha = [0.4, 0.4, 0.4, 0.4, 1]
        model_names = ['NN', 'l1', 'RF', 'XGBoost', 'ensemble']
        model_labels = ['NN', 'l1', 'RF', 'XGBoost', 'Ensemble']
        for i, model in enumerate(model_names):
            plt.plot(self.final_fpr[model], self.final_tpr[model], color=colors[i], alpha=alpha[i],
                     label=f"{model_labels[i]}: {round(self.final_auc_results[model], ndigits=4)}")
        plt.title(self.name)
        plt.legend()
        plt.savefig(self.out_path / self.name / 'ROC.png')
        plt.close()

    def plot_calibration(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        colors = ['orange', 'cyan', 'r', 'dodgerblue', 'k']
        model_names = ['NN', 'l1', 'RF', 'XGBoost', 'ensemble_pred']
        model_labels = ['NN', 'l1', 'RF', 'XGBoost', 'Ensemble']
        for i, model in enumerate(model_names):
            prob_true, prob_pred = calibration_curve(self.test_df['true'], self.test_df[model], n_bins=10,
                                                     strategy='quantile')
            if model=='ensemble_pred':
                ax[0].plot(prob_pred, prob_true, marker=".", color=colors[i], linewidth=2, label=model_labels[i])
                brier_score = brier_score_loss(self.test_df['true'], self.test_df[model])
                with open(self.out_path / self.name / 'brier_score.txt', 'w') as f:
                    f.write(f"brier_score: {brier_score}")
                print("brier_score:", brier_score)
            else:
                ax[0].plot(prob_pred, prob_true, marker=".", color=colors[i], alpha=0.4, label=model_labels[i])
            ax[0].set_title('Calibration Plot')
            ax[0].plot([0, 1], [0, 1], linestyle='dashed', color='gray')
            ax[0].set_ylabel('Fraction of positives')
            ax[0].set_xlabel('Mean predicted probability')
            ax[0].set_xticks(np.linspace(0, 1, 11))
            kde = st.gaussian_kde(self.test_df[model].values)
            if model=='ensemble_pred':
                ax[1].plot(np.linspace(0, 1, 11), kde(np.linspace(0, 1, 11)), marker=".", color=colors[i], linewidth=2, label=model_labels[i])
            else:
                ax[1].plot(np.linspace(0, 1, 11), kde(np.linspace(0, 1, 11)), marker=".", color=colors[i], alpha=0.4, label=model_labels[i])
            ax[1].set_xlim(0, 1)
            ax[1].set_xticks(np.linspace(0, 1, 11))
            ax[1].set_title('Kernel Density Estimte')
            ax[1].set_xlabel('Predicted probability')
            ax[1].set_ylabel('Density')
            ax[1].legend()
        plt.savefig(self.out_path / self.name / 'calibration.png')
        plt.close()

    def calculate_net_benefit_model(self):
        net_benefit_model = np.array([])
        for thresh in self.ensemble_thresholds:
            y_pred_label = (self.ensemble_y_score > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.ensemble_y_true, y_pred_label).ravel()
            n = len(self.ensemble_y_true)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefit_model = np.append(net_benefit_model, net_benefit)
        return net_benefit_model

    def calculate_net_benefit_all(self):
        net_benefit_all = np.array([])
        print()
        best_index = np.argmax(np.array(self.final_tpr['ensemble']) - np.array(self.final_fpr['ensemble']))
        best_threshold = self.ensemble_thresholds[best_index]
        binary_y_score = self.ensemble_y_score >= best_threshold
        tn, fp, fn, tp = confusion_matrix(self.ensemble_y_true, binary_y_score).ravel()
        total = tp + tn
        for thresh in self.ensemble_thresholds:
            net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
            net_benefit_all = np.append(net_benefit_all, net_benefit)
        return net_benefit_all

    def plot_desicion_curve(self):
        net_benefit_model = self.calculate_net_benefit_model()
        net_benefit_all = self.calculate_net_benefit_all()
        fig, ax = plt.subplots()
        ax.plot(self.ensemble_thresholds, net_benefit_model, color='crimson', label='Model')
        ax.plot(self.ensemble_thresholds, net_benefit_all, color='black', label='Treat all')
        ax.plot((0, 1), (0, 0), color='black', linestyle=':', label='Treat none')
        # Fill, Shows that the model is better than treat all and treat none The good part
        y2 = np.maximum(net_benefit_all, 0)
        y1 = np.maximum(net_benefit_model, y2)
        ax.fill_between(self.ensemble_thresholds, y1, y2, color='crimson', alpha=0.2)
        # Figure Configuration, Beautify the details
        ax.set_xlim(0, 1)
        ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
        ax.set_xlabel(
            xlabel='Threshold Probability',
            fontdict={
                'family': 'Times New Roman', 'fontsize': 15}
        )
        ax.set_ylabel(
            ylabel='Net Benefit',
            fontdict={
                'family': 'Times New Roman', 'fontsize': 15}
        )
        ax.grid('major')
        ax.spines['right'].set_color((0.8, 0.8, 0.8))
        ax.spines['top'].set_color((0.8, 0.8, 0.8))
        ax.legend(loc='upper right')
        fig.savefig(self.out_path / self.name / 'net_benefit.png', bbox_inches='tight')
        plt.close()


    def calc_shap_values(self):
        def f(X):
            return (self.meta_learner_coef_dict['NN'] * self.clfs['NN'].predict(X)).tolist() + \
                   self.meta_learner_coef_dict['l1'] * self.clfs['l1'].predict_proba(X) + \
                   self.meta_learner_coef_dict['RF'] * self.clfs['RF'].predict_proba(X) + \
                   self.meta_learner_coef_dict['XGBoost'] * self.clfs['XGBoost'].predict_proba(X)

        print('calculating shap values...')
        x_train = pd.DataFrame(self.X_train, columns=self.col_names)
        x_test = pd.DataFrame(self.X_test, columns=self.col_names)
        print(self.col_names)
        explainer = shap.KernelExplainer(f, x_train.iloc[0:100])
        self.shap_values = explainer.shap_values(x_test)
        pd.DataFrame(self.shap_values[1], columns=self.col_names).to_csv(self.out_path / self.name / 'shap_table.csv')

        print('plotting and saving')
        shap.summary_plot(self.shap_values[1], x_test, plot_type='dot', show=False, color_bar=False)
        plt.savefig(self.out_path / self.name / 'dot_sum_plot.png', bbox_inches='tight')
        plt.close()
        shap.summary_plot(self.shap_values[1], x_test, plot_type='bar', show=False, color_bar=False)
        plt.savefig(self.out_path / self.name / 'bar_sum_plot.png', bbox_inches='tight')
        plt.close()
        

csv_paths= {
            'Agnostic': get_script_dir() / 'agnostics_22_06_22.csv',
            'Gnostic': get_script_dir() / 'gnostics_22_06_22.csv',
            }

out_path = get_script_dir() / 'results_200_iter'
for name, csv_path in csv_paths.items():
    print(name)
    # df = pd.read_excel(csv_path, parse_dates=[0], index_col=[0])
    df = pd.read_csv(csv_path, parse_dates=['Culture.Time.Date'], index_col=['Culture.Time.Date'])
    cols_to_drop = ['TimeSinceEndLastUse365', 'RunningDays']
    df = df.drop(cols_to_drop, axis=1)
    df = df.fillna(0)
    for col in df.columns:
        if 0 in df[col].unique() and 1 in df[col].unique() and df[col].nunique() == 2:
            df[col] = df[col].astype('int8')
    float16_cols = ['Age.At.Culture.Time.Date', 'CumAnyUsage365', 'Length.Days.Current.Hospitalization',
                    'Length.Days.Hospitalized.Past.365.Days', 'PrevAntiUsageBin181_365', 'PrevAntiUsageBin60',
                    'PrevAntiUsageBin61_180', 'PrevAntiUsageFamilyBin181_365', 'PrevAntiUsageFamilyBin60',
                    'PrevAntiUsageFamilyBin61_180', 'PrevAntiUsageOtherBin181_365', 'PrevAntiUsageOtherBin60',
                    'PrevAntiUsageOtherBin61_180', 'PrevAntiUsageOtherFamilyBin181_365', 'PrevAntiUsageOtherFamilyBin60',
                    'PrevAntiUsageOtherFamilyBin61_180', 'PrevAntiUsageSaneFamilyOtherAntiBin181_365',
                    'PrevAntiUsageSaneFamilyOtherAntiBin60', 'PrevAntiUsageSaneFamilyOtherAntiBin61_180',
                    'PreviouslyResistantOtherAntiSameFamilyBin180_365', 'SlidingResAny30', 'SlidingResSame30',
                    'SlidingResUnits30', 'T.Since.Comorbidity.Scores', 'SlidingResSameBacAnyAnti30']
    for col in float16_cols:
        try:
            df[col] = df[col].astype('float16')
        except KeyError:
            continue
    df = df.drop('T.Since.Comorbidity.Scores', axis=1)
    by_bac = EnsembleSingleAnti(name, df, out_path, n_iter=200)

print('**********************')
print('finished everything!!!')
print('**********************')
