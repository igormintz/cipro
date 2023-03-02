from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve

plt.rcParams["savefig.dpi"] = 600
plt.rcParams.update({"font.size": 16})


def plot_roc(pred_df: pd.DataFrame, name: str, out_path: Path):
    fig = plt.figure(figsize=(8, 8))
    colors = ["orange", "cyan", "r", "dodgerblue", "k"]
    # line_style = ["solid", "solid", "solid", "solid", "dashed"]
    alpha = [0.4, 0.4, 0.4, 0.4, 1]
    model_names = ["NN", "l1", "RF", "XGBoost", "ensemble"]
    model_labels = ["NN", "l1", "RF", "XGBoost", "Ensemble"]
    for i, model in enumerate(model_names):
        fpr, tpr, _ = roc_curve(pred_df["y_test"], pred_df[model])
        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            alpha=alpha[i],
            label=f"{model_labels[i]}: {round(auc(fpr, tpr), ndigits=4)}",
        )
    plt.title(name)
    plt.legend()
    plt.savefig(out_path / name / "ROC.png")
    plt.close()


def plot_calibration(df: pd.DataFrame, name: str, out_path: Path):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    colors = ["orange", "cyan", "r", "dodgerblue", "k"]
    model_names = ["NN", "l1", "RF", "XGBoost", "ensemble"]
    model_labels = ["NN", "l1", "RF", "XGBoost", "Ensemble"]
    for i, model in enumerate(model_names):
        prob_true, prob_pred = calibration_curve(df["y_test"], df[model], n_bins=10, strategy="quantile")
        if model == "ensemble":
            ax[0].plot(prob_pred, prob_true, marker=".", color=colors[i], linewidth=2, label=model_labels[i])
        else:
            ax[0].plot(prob_pred, prob_true, marker=".", color=colors[i], alpha=0.4, label=model_labels[i])
        ax[0].set_title("Calibration Plot")
        ax[0].plot([0, 1], [0, 1], linestyle="dashed", color="gray")
        ax[0].set_ylabel("Fraction of positives")
        ax[0].set_xlabel("Mean predicted probability")
        ax[0].set_xticks(np.linspace(0, 1, 11))
        kde = st.gaussian_kde(df[model].values)
        if model == "ensemble":
            ax[1].plot(
                np.linspace(0, 1, 11),
                kde(np.linspace(0, 1, 11)),
                marker=".",
                color=colors[i],
                linewidth=2,
                label=model_labels[i],
            )
        else:
            ax[1].plot(
                np.linspace(0, 1, 11),
                kde(np.linspace(0, 1, 11)),
                marker=".",
                color=colors[i],
                alpha=0.4,
                label=model_labels[i],
            )
        ax[1].set_xlim(0, 1)
        ax[1].set_xticks(np.linspace(0, 1, 11))
        ax[1].set_title("Kernel Density Estimte")
        ax[1].set_xlabel("Predicted probability")
        ax[1].set_ylabel("Density")
        ax[1].legend()
    plt.savefig(out_path / name / "calibration.png")
    plt.close()


def calculate_net_benefit_model(ensemble_y_true, ensemble_y_score):
    tpr, fpr, ensemble_thresholds = roc_curve(ensemble_y_true, ensemble_y_score)
    net_benefit_model = np.array([])
    for thresh in ensemble_thresholds:
        y_pred_label = (ensemble_y_score > thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(ensemble_y_true, y_pred_label).ravel()
        n = len(ensemble_y_true)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(ensemble_y_true, ensemble_y_score):
    tpr, fpr, ensemble_thresholds = roc_curve(ensemble_y_true, ensemble_y_score)
    net_benefit_all = np.array([])
    best_index = np.argmax(np.array(tpr) - np.array(fpr))
    best_threshold = ensemble_thresholds[best_index]
    binary_y_score = ensemble_y_score >= best_threshold
    tn, fp, fn, tp = confusion_matrix(ensemble_y_true, binary_y_score).ravel()
    total = tp + tn
    for thresh in ensemble_thresholds:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_desicion_curve(
    pred_df: pd.DataFrame,
    name: str,
    out_path: Path,
):
    net_benefit_model = calculate_net_benefit_model(pred_df["y_test"], pred_df["ensemble"])
    net_benefit_all = calculate_net_benefit_all(pred_df["y_test"], pred_df["ensemble"])
    fig, ax = plt.subplots()
    tpr, fpr, ensemble_thresholds = roc_curve(pred_df["y_test"], pred_df["ensemble"])
    ax.plot(ensemble_thresholds, net_benefit_model, color="crimson", label="Model")
    ax.plot(ensemble_thresholds, net_benefit_all, color="black", label="Treat all")
    ax.plot((0, 1), (0, 0), color="black", linestyle=":", label="Treat none")
    # Fill, Shows that the model is better than treat all and treat none The good part
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(ensemble_thresholds, y1, y2, color="crimson", alpha=0.2)
    # Figure Configuration, Beautify the details
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)  # adjustify the y axis limitation
    ax.set_xlabel(xlabel="Threshold Probability", fontdict={"family": "Times New Roman", "fontsize": 15})
    ax.set_ylabel(ylabel="Net Benefit", fontdict={"family": "Times New Roman", "fontsize": 15})
    ax.grid("major")
    ax.spines["right"].set_color((0.8, 0.8, 0.8))
    ax.spines["top"].set_color((0.8, 0.8, 0.8))
    ax.legend(loc="upper right")
    fig.savefig(out_path / name / "net_benefit.png", bbox_inches="tight")
    plt.close()
