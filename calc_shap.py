from multiprocessing import Manager, Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference import load_models
from main import get_script_dir
from utils import get_data, split_and_scale

plt.rcParams["savefig.dpi"] = 600
plt.rcParams.update({"font.size": 16})
import shap

model_names = ["NN", "l1", "RF", "XGBoost", "ensemble"]


def calc_shap_values(
    X_train: pd.DataFrame, X_test: pd.DataFrame, models: dict, col_names: list, out_path: Path, name: str
):
    def f(X):
        return (
            (models["ensemble"].coef_[0][0] * models["NN"].predict(X)).tolist()
            + models["ensemble"].coef_[0][1] * models["l1"].predict_proba(X)
            + models["ensemble"].coef_[0][2] * models["RF"].predict_proba(X)
            + models["ensemble"].coef_[0][3] * models["XGBoost"].predict_proba(X)
        )

    # '''Fixed by removing link="logit" and wrapping shap.force_plot(float(explainer.expected_value[0]), shap_values[0][0,:], X_test.iloc[0,:], link="logit")`'''
    print("calculating shap values...")
    X_train = pd.DataFrame(X_train, columns=col_names)
    explainer = shap.KernelExplainer(f, X_train.iloc[0:20])
    shap_values = explainer.shap_values(X_test, nsamples=100)
    pd.DataFrame(shap_values[1], columns=col_names).to_csv(out_path / name / "shap_table.csv")

    print("plotting and saving")
    shap.summary_plot(shap_values[1], X_test, plot_type="dot", show=False, color_bar=False)
    plt.savefig(out_path / name / "dot_sum_plot.png", bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False, color_bar=False)
    plt.savefig(out_path / name / "bar_sum_plot.png", bbox_inches="tight")
    plt.close()


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
        X_train = pd.DataFrame(X_train, columns=col_names)
        X_test = pd.DataFrame(X_test, columns=col_names)
        models = load_models(out_path, name, model_names)
        calc_shap_values(X_train, X_test, models, col_names, out_path, name)
