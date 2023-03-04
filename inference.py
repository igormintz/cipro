import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from main import get_script_dir
from plots import *
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from utils import get_data, split_and_scale

plt.rcParams["savefig.dpi"] = 600
plt.rcParams.update({"font.size": 16})
model_names = ["NN", "l1", "RF", "XGBoost", "ensemble"]


def load_models(out_path: Path, name: str, model_names: list) -> dict:
    model_dict = {}
    for model in model_names:
        if model == "NN":
            model_dict[model] = load_model(out_path / name / f"{model}.h5")
            model_dict[model].load_weights(out_path / name / f"{model}_weights.h5")
        else:
            tmp = open(out_path / name / f"{model}.pkl", "rb")
            model_dict[model] = pickle.load(tmp)
    return model_dict


def make_predictions(models: dict, X_test: pd.DataFrame, y_test) -> pd.DataFrame:
    predictions = {}
    for model_name, model in models.items():
        if model_name == "NN":
            predictions[f"{model_name}"] = model.predict(X_test).flatten()
        elif model_name != "ensemble":
            predictions[f"{model_name}"] = model.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame(predictions)
    ensemble = model.predict_proba(predictions)[:, 1]
    predictions = predictions.assign(ensemble=ensemble, y_test=y_test.reset_index(drop=True))
    return predictions


def get_auc_ci(pred_df: pd.DataFrame) -> tuple:
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(pred_df), len(pred_df))
        if len(np.unique(pred_df["y_test"][indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(pred_df["y_test"][indices], pred_df["ensemble"][indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    auc_ci = (confidence_lower, confidence_upper)
    return auc_ci


def calc_sn_sp_acc_ppv(pred_df: pd.DataFrame) -> dict:
    sn_sp_acc_ppv = {}
    pred_df["rounded_proba"] = list(map(lambda x: round(x), pred_df["ensemble"]))
    pred_df["TP"] = pred_df.apply(lambda x: 1 if (x["rounded_proba"] == 1 and x["y_test"] == 1) else 0, axis=1)
    pred_df["TN"] = pred_df.apply(lambda x: 1 if (x["rounded_proba"] == 0 and x["y_test"] == 0) else 0, axis=1)
    pred_df["FP"] = pred_df.apply(lambda x: 1 if (x["rounded_proba"] == 1 and x["y_test"] == 0) else 0, axis=1)
    sn_sp_acc_ppv["sensitivity"] = pred_df["TP"].sum() / pred_df["y_test"].sum()
    sn_sp_acc_ppv["specificity"] = pred_df["TN"].sum() / len(pred_df[pred_df["y_test"] == 0])
    sn_sp_acc_ppv["accuracy"] = (pred_df["TP"].sum() + pred_df["TN"].sum()) / len(pred_df)
    sn_sp_acc_ppv["ppv"] = pred_df["TP"].sum() / pred_df["rounded_proba"].sum()
    return sn_sp_acc_ppv


def save_summary(predictions: pd.DataFrame, name: str, models: dict, out_path: Path) -> None:
    aucs = {}
    ensemble_auc_ci = get_auc_ci(predictions)
    ensemble_sn_sp_acc_ppv = calc_sn_sp_acc_ppv(predictions)
    for model in model_names:
        aucs[model] = roc_auc_score(predictions["y_test"], predictions[model])

    with open(out_path / name / "summary.txt", "w") as f:
        f.write(f"AUCs: {aucs}\n")
        f.write(f"ensemble AUC CI: {ensemble_auc_ci}\n")
        f.write(f"ensemble sensitivity: {ensemble_sn_sp_acc_ppv['sensitivity']}\n")
        f.write(f"ensemble specificity: {ensemble_sn_sp_acc_ppv['specificity']}\n")
        f.write(f"ensemble accuracy: {ensemble_sn_sp_acc_ppv['accuracy']}\n")
        f.write(f"ensemble PPV: {ensemble_sn_sp_acc_ppv['ppv']}\n")
        for model_name, model in models.items():
            f.write(f"{model_name}:\n")
            if model_name == "NN":
                f.write(f"{model.summary()}\n")
            elif model_name == "ensemble":
                f.write(f"coefs: {model.coef_}, intercept: {model.intercept_}\n")
            else:
                f.write(f"{model.get_params()}\n")


if "__main__" == __name__:
    csv_paths = {
        "Agnostic": get_script_dir() / "agnostics_22_06_22.csv",
        "Gnostic": get_script_dir() / "gnostics_22_06_22.csv",
    }

    out_path = get_script_dir() / "results"
    for name, csv_path in csv_paths.items():
        print(name)
        df = get_data(csv_path)
        X_train, X_test, y_train, y_test, col_names = split_and_scale(df)
        # X_test = pd.DataFrame(X_test, columns=col_names)
        models = load_models(out_path, name, model_names)
        predictions = make_predictions(models, X_test, y_test)
        predictions.to_csv(out_path / name / "predictions.csv")
        plot_roc(predictions, name, out_path)
        plot_calibration(predictions, name, out_path)
        plot_desicion_curve(predictions, name, out_path)
        save_summary(predictions, name, models, out_path)
