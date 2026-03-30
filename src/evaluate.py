import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from config import CFG

warnings.filterwarnings("ignore")

_C = {
    "red":   f"#{CFG.RED}",
    "dark":  f"#{CFG.DARK}",
    "gold":  f"#{CFG.GOLD}",
    "gray":  f"#{CFG.GRAY}",
    "green": f"#{CFG.GREEN}",
    "bg":    "#F8F9FB",
    "line":  "#E2E8F0",
}
_MODEL_COLORS = [_C["red"], "#1A56DB", _C["gold"], "#7C3AED"]

plt.rcParams.update({
    "figure.facecolor": _C["bg"],
    "axes.facecolor":   _C["bg"],
    "axes.edgecolor":   _C["line"],
    "axes.grid":        True,
    "grid.color":       _C["line"],
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.dpi":       150,
})


def compute_metrics(y_true, y_pred, y_proba, model_name, threshold=0.5):
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr  = average_precision_score(y_true, y_proba)
    brier   = brier_score_loss(y_true, y_proba)
    f1      = f1_score(y_true, y_pred)
    report  = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    cm      = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "name":          model_name,
        "auc_roc":       round(auc_roc, 4),
        "auc_pr":        round(auc_pr, 4),
        "brier":         round(brier, 4),
        "f1_churn":      round(f1, 4),
        "precision_churn": round(report["Churn"]["precision"], 4),
        "recall_churn":    round(report["Churn"]["recall"], 4),
        "threshold":     threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "report":        report,
        "cm":            cm,
    }


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(( correct_a & ~correct_b).sum())
    c = int((~correct_a &  correct_b).sum())
    if b + c == 0:
        return 1.0, 0.0
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - stats.chi2.cdf(statistic, df=1)
    return round(p_value, 4), round(statistic, 4)


def lift_curve_data(y_true, y_proba, n_bins=20):
    df = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False).reset_index(drop=True)
    baseline = y_true.mean()
    records  = []
    for i in range(1, n_bins + 1):
        n      = int(len(df) * i / n_bins)
        subset = df.iloc[:n]
        rate   = subset["y"].mean()
        lift   = rate / baseline if baseline > 0 else 0
        records.append({"pct_targeted": i / n_bins, "lift": round(lift, 4), "churn_rate": round(rate, 4), "n": n})
    return records


def get_feature_names(pipeline):
    ct  = pipeline.named_steps["pre"]
    num = list(ct.transformers_[0][2])
    try:
        cat = list(ct.transformers_[1][1].get_feature_names_out(ct.transformers_[1][2]))
    except Exception:
        cat = []
    return num + cat


def plot_roc_pr(results, y_test, save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Curvas ROC y Precisión-Recall", fontsize=14, fontweight="bold", y=1.01)

    for (name, res), color in zip(results.items(), _MODEL_COLORS):
        m = res["metrics"]
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax1.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={m['auc_roc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax2.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={m['auc_pr']:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatorio")
    ax1.set_xlabel("Tasa Falsos Positivos")
    ax1.set_ylabel("Tasa Verdaderos Positivos")
    ax1.set_title("Curva ROC")
    ax1.legend(loc="lower right")

    baseline = y_test.mean()
    ax2.axhline(baseline, color="k", lw=1, ls="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precisión")
    ax2.set_title("Curva Precisión-Recall")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "01_roc_pr.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results, save=True):
    n     = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Matrices de Confusión", fontsize=14, fontweight="bold")

    for ax, (name, res), color in zip(axes, results.items(), _MODEL_COLORS):
        cm = res["metrics"]["cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds", ax=ax,
            linewidths=0.5, linecolor=_C["line"],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            cbar=False,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "02_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results, save=True):
    metrics = ["auc_roc", "auc_pr", "f1_churn", "recall_churn", "precision_churn"]
    labels  = ["AUC-ROC", "AUC-PR", "F1 Churn", "Recall Churn", "Precisión Churn"]
    names   = list(results.keys())
    x       = np.arange(len(metrics))
    w       = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, color) in enumerate(zip(names, _MODEL_COLORS)):
        vals = [results[name]["metrics"][m] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=name, color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de Modelos — Métricas en Test", fontsize=13)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "03_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(pipeline, X_test, y_test, model_name, top_n=18, save=True):
    try:
        clf = pipeline.named_steps["clf"]
        X_t = pipeline.named_steps["pre"].transform(X_test)
        perm = permutation_importance(clf, X_t, y_test, n_repeats=15, random_state=CFG.random_state, n_jobs=CFG.n_jobs, scoring="roc_auc")
        names = get_feature_names(pipeline)
        fi_df = pd.DataFrame({"feature": names[:len(perm.importances_mean)], "importance": perm.importances_mean, "std": perm.importances_std})
        fi_df = fi_df[fi_df["importance"] > 0].sort_values("importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], xerr=fi_df["std"][::-1], color=_C["red"], alpha=0.85, edgecolor="white", capsize=3)
        ax.set_xlabel("Permutation Importance (AUC-ROC drop)")
        ax.set_title(f"Top {top_n} Variables — {model_name}\n(caída en AUC-ROC al permutar, promedio ± std, 15 repeticiones)", fontsize=11)
        plt.tight_layout()
        if save:
            fig.savefig(CFG.figures_dir / "04_feature_importance.png", bbox_inches="tight")
        plt.close(fig)
        return fi_df
    except Exception:
        return pd.DataFrame()


def plot_threshold_analysis(threshold_results, model_name, save=True):
    data = list(threshold_results["all"].values())
    ts   = [d["threshold"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Análisis de Threshold — {model_name}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(ts, [d["f1"] for d in data],    color=_C["red"],  lw=2, label="F1")
    ax.plot(ts, [d["f2"] for d in data],    color="#1A56DB",  lw=2, label="F2 (recall-weighted)")
    ax.plot(ts, [d["precision"] for d in data], color=_C["gold"], lw=2, label="Precisión")
    ax.plot(ts, [d["recall"] for d in data],    color=_C["green"], lw=2, label="Recall")
    ax.axvline(threshold_results["best_f1"]["threshold"],   color=_C["red"],  lw=1.2, ls="--", alpha=0.7)
    ax.axvline(threshold_results["best_cost"]["threshold"],  color="#7C3AED",  lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("F1 / F2 / Precisión / Recall")
    ax.legend()

    ax = axes[1]
    costs = [d["business_cost"] for d in data]
    ax.plot(ts, costs, color="#7C3AED", lw=2.5)
    ax.axvline(threshold_results["best_cost"]["threshold"], color="#7C3AED", lw=1.5, ls="--", alpha=0.8,
               label=f'Óptimo: {threshold_results["best_cost"]["threshold"]:.2f} (costo ${threshold_results["best_cost"]["business_cost"]:,.0f})')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Costo total ($)")
    ax.set_title(f"Costo de Negocio\n(FN=${CFG.fn_cost:.0f} · FP=${CFG.fp_cost:.0f})")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "05_threshold_analysis.png", bbox_inches="tight")
    plt.close(fig)


def plot_lift_curves(lift_data_dict, save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Curvas de Lift y Ganancia", fontsize=13, fontweight="bold")

    for (name, data), color in zip(lift_data_dict.items(), _MODEL_COLORS):
        pcts  = [d["pct_targeted"] for d in data]
        lifts = [d["lift"]         for d in data]
        gains = [d["churn_rate"]   for d in data]
        ax1.plot(pcts, lifts, lw=2, color=color, label=name)
        ax2.plot(pcts, gains, lw=2, color=color, label=name)

    ax1.axhline(1.0, color="k", lw=1, ls="--", alpha=0.5, label="Sin modelo (lift=1)")
    ax1.set_xlabel("% clientes contactados")
    ax1.set_ylabel("Lift")
    ax1.set_title("Lift Curve\n(cuánto mejor que aleatorio)")
    ax1.legend()

    ax2.axhline(lift_data_dict[list(lift_data_dict.keys())[0]][0]["churn_rate"] * 0 + 0.265,
                color="k", lw=1, ls="--", alpha=0.5, label="Baseline (26.5%)")
    ax2.set_xlabel("% clientes contactados")
    ax2.set_ylabel("Tasa de churn real en segmento")
    ax2.set_title("Gain Curve\n(tasa de churn por segmento)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend()

    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "06_lift_gain.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results, y_test, save=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Calibración perfecta")

    for (name, res), color in zip(results.items(), _MODEL_COLORS):
        prob_true, prob_pred = calibration_curve(y_test, res["y_proba"], n_bins=10)
        brier = res["metrics"]["brier"]
        ax.plot(prob_pred, prob_true, "o-", lw=2, ms=5, color=color,
                label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Probabilidad predicha promedio")
    ax.set_ylabel("Fracción de positivos reales")
    ax.set_title("Calibración de Probabilidades\n(más cercano a la diagonal = mejor calibrado)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "07_calibration.png", bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(pipeline, X_train, y_train, model_name, save=True):
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=5, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=CFG.n_jobs,
        random_state=CFG.random_state,
    )

    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    va_mean, va_std = val_scores.mean(axis=1),   val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean, "o-", color=_C["red"],  lw=2, label="Train AUC-ROC")
    ax.plot(train_sizes, va_mean, "o-", color="#1A56DB", lw=2, label="Validation AUC-ROC")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.12, color=_C["red"])
    ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std, alpha=0.12, color="#1A56DB")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Curva de Aprendizaje — {model_name}")
    ax.legend()
    gap = tr_mean[-1] - va_mean[-1]
    ax.text(0.02, 0.05, f"Gap final: {gap:.3f}", transform=ax.transAxes, fontsize=9,
            color=_C["red"] if gap > 0.05 else _C["green"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=_C["line"]))
    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "08_learning_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_risk_segmentation(y_test, y_proba, best_name, save=True):
    bins   = [0, 0.25, 0.50, 0.75, 1.0]
    labels = ["Bajo\n<25%", "Medio\n25-50%", "Alto\n50-75%", "Crítico\n>75%"]
    colors = [_C["green"], _C["gold"], "#F97316", _C["red"]]

    seg = pd.cut(y_proba, bins=bins, labels=labels)
    df  = pd.DataFrame({"seg": seg, "y": y_test, "p": y_proba})
    summary = df.groupby("seg", observed=True).agg(
        n=("y", "count"), churn=("y", "sum"), rate=("y", "mean")
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Segmentación de Riesgo — {best_name}", fontsize=13, fontweight="bold")

    bars = ax1.bar(summary["seg"], summary["n"], color=colors, edgecolor="white", linewidth=0.5)
    for bar, row in zip(bars, summary.itertuples()):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{row.n:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Número de clientes")
    ax1.set_title("Volumen por Segmento")

    bars2 = ax2.bar(summary["seg"], summary["rate"] * 100, color=colors, edgecolor="white", linewidth=0.5)
    for bar, row in zip(bars2, summary.itertuples()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{row.rate:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.axhline(y_test.mean() * 100, color="k", lw=1.2, ls="--", alpha=0.6,
                label=f"Tasa global ({y_test.mean():.1%})")
    ax2.set_ylabel("Tasa de churn real (%)")
    ax2.set_title("Tasa de Churn Real por Segmento")
    ax2.legend()

    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "09_risk_segmentation.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_eda(df, save=True):
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["Churn"].value_counts()
    ax1.bar(["No Churn", "Churn"], counts.values, color=[_C["gray"], _C["red"]], edgecolor="white")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 30, f"{v:,}\n({v/len(df):.1%})", ha="center", fontsize=9)
    ax1.set_title("Distribución de Churn")
    ax1.set_ylabel("Clientes")

    ax2 = fig.add_subplot(gs[0, 1])
    cr  = df.groupby("Contract")["Churn"].mean().sort_values()
    ax2.barh(cr.index, cr.values * 100, color=_C["red"], alpha=0.85, edgecolor="white")
    for i, v in enumerate(cr.values):
        ax2.text(v + 0.3, i, f"{v:.1%}", va="center", fontsize=9)
    ax2.set_xlabel("Tasa de churn (%)")
    ax2.set_title("Churn por Tipo de Contrato")

    ax3 = fig.add_subplot(gs[0, 2])
    for val, color, label in [(0, _C["gray"], "No Churn"), (1, _C["red"], "Churn")]:
        ax3.hist(df[df["Churn"] == val]["tenure"], bins=30, alpha=0.65, color=color, label=label, edgecolor="white")
    ax3.set_xlabel("Antigüedad (meses)")
    ax3.set_ylabel("Frecuencia")
    ax3.set_title("Distribución de Tenure")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    for val, color, label in [(0, _C["gray"], "No Churn"), (1, _C["red"], "Churn")]:
        ax4.hist(df[df["Churn"] == val]["MonthlyCharges"], bins=30, alpha=0.65, color=color, label=label, edgecolor="white")
    ax4.set_xlabel("Cargo mensual ($)")
    ax4.set_title("Distribución de MonthlyCharges")
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    service_cr = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
    ax5.bar(service_cr.index, service_cr.values * 100, color=[_C["red"], _C["gold"], _C["green"]], edgecolor="white")
    for i, v in enumerate(service_cr.values):
        ax5.text(i, v + 0.5, f"{v:.1%}", ha="center", fontsize=9)
    ax5.set_ylabel("Tasa de churn (%)")
    ax5.set_title("Churn por Servicio de Internet")

    ax6 = fig.add_subplot(gs[1, 2])
    pay_cr = df.groupby("PaymentMethod")["Churn"].mean().sort_values()
    short  = [p.replace("(automatic)", "(auto)").replace("Bank transfer", "Bank transf.") for p in pay_cr.index]
    ax6.barh(short, pay_cr.values * 100, color=_C["dark"], alpha=0.85, edgecolor="white")
    for i, v in enumerate(pay_cr.values):
        ax6.text(v + 0.3, i, f"{v:.1%}", va="center", fontsize=9)
    ax6.set_xlabel("Tasa de churn (%)")
    ax6.set_title("Churn por Método de Pago")

    fig.suptitle("Análisis Exploratorio — Dataset Telco Churn", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(CFG.figures_dir / "00_eda.png", bbox_inches="tight")
    plt.close(fig)
