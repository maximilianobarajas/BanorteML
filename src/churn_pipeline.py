import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG
from evaluate import (
    compute_metrics,
    mcnemar_test,
    lift_curve_data,
    plot_roc_pr,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_lift_curves,
    plot_calibration,
    plot_learning_curves,
    plot_risk_segmentation,
    plot_eda,
)
from features import load_raw, engineer
from models import build_all, optimize_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("banorte.churn")


def setup_dirs():
    for d in (CFG.output_dir, CFG.figures_dir, CFG.models_dir):
        d.mkdir(parents=True, exist_ok=True)


def run(args):
    setup_dirs()
    np.random.seed(CFG.random_state)

    log.info("=" * 65)
    log.info("  PREDICCIÓN DE CHURN — Banorte Prueba Técnica ML v2")
    log.info("=" * 65)

    log.info("Cargando y preparando datos…")
    df_raw = load_raw(args.data)
    df     = engineer(df_raw)

    log.info(f"  Filas: {len(df):,}  |  Churn: {df['Churn'].mean():.2%}")

    numeric_all  = list(CFG.NUMERIC_RAW) + list(CFG.NUMERIC_ENGINEERED)
    categorical  = list(CFG.CATEGORICAL)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state, stratify=y
    )
    log.info(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    if not args.no_plots:
        log.info("Generando EDA…")
        plot_eda(df)

    log.info("Construyendo modelos…")
    models = build_all(numeric_all, categorical)

    cv     = StratifiedKFold(n_splits=CFG.cv_folds, shuffle=True, random_state=CFG.random_state)
    cv_results: dict[str, dict] = {}
    trained:    dict[str, object] = {}

    for name, pipeline in models.items():
        log.info(f"  Entrenando: {name}")
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"auc_roc": "roc_auc", "auc_pr": "average_precision", "f1": "f1"},
            n_jobs=CFG.n_jobs, return_train_score=False,
        )
        cv_results[name] = {
            "cv_auc_roc_mean": round(scores["test_auc_roc"].mean(), 4),
            "cv_auc_roc_std":  round(scores["test_auc_roc"].std(),  4),
            "cv_auc_pr_mean":  round(scores["test_auc_pr"].mean(),  4),
            "cv_f1_mean":      round(scores["test_f1"].mean(),      4),
        }
        log.info(
            f"    CV AUC-ROC: {cv_results[name]['cv_auc_roc_mean']:.4f}"
            f" ± {cv_results[name]['cv_auc_roc_std']:.4f}"
        )
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

    test_results: dict[str, dict] = {}
    for name, pipeline in trained.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)
        m = compute_metrics(y_test, y_pred, y_proba, name)
        m["y_proba"] = y_proba
        m["y_pred"]  = y_pred
        test_results[name] = {"metrics": m, "pipeline": pipeline, "y_proba": y_proba, "y_pred": y_pred}
        log.info(
            f"  {name:<22}  AUC-ROC={m['auc_roc']:.4f}  "
            f"AUC-PR={m['auc_pr']:.4f}  F1={m['f1_churn']:.4f}  Brier={m['brier']:.4f}"
        )

    best_name = max(test_results, key=lambda k: test_results[k]["metrics"]["auc_roc"])
    log.info(f"\n✅ Mejor modelo: {best_name}  (AUC-ROC={test_results[best_name]['metrics']['auc_roc']:.4f})")

    log.info("Optimizando threshold…")
    best_proba    = test_results[best_name]["y_proba"]
    thresh_results = optimize_threshold(y_test.values, best_proba)
    log.info(f"  Mejor F1     → threshold={thresh_results['best_f1']['threshold']:.2f}  F1={thresh_results['best_f1']['f1']:.4f}")
    log.info(f"  Mejor F2     → threshold={thresh_results['best_f2']['threshold']:.2f}  recall={thresh_results['best_f2']['recall']:.4f}")
    log.info(f"  Menor costo  → threshold={thresh_results['best_cost']['threshold']:.2f}  costo=${thresh_results['best_cost']['business_cost']:,.0f}")

    opt_threshold = thresh_results["best_f1"]["threshold"]
    y_pred_opt    = (best_proba >= opt_threshold).astype(int)
    test_results[best_name]["y_pred_opt"] = y_pred_opt
    test_results[best_name]["opt_threshold"] = opt_threshold

    log.info("Pruebas estadísticas entre modelos (McNemar)…")
    names = list(trained.keys())
    mcnemar_matrix = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            p, stat = mcnemar_test(
                y_test.values,
                test_results[a]["y_pred"],
                test_results[b]["y_pred"],
            )
            mcnemar_matrix[f"{a} vs {b}"] = {"p_value": p, "statistic": stat, "significant": p < 0.05}
            sig = "✓ significativa" if p < 0.05 else "✗ no significativa"
            log.info(f"  {a} vs {b}: p={p:.4f} ({sig})")

    log.info("Calculando lift curves…")
    lift_data_dict = {
        name: lift_curve_data(y_test.values, res["y_proba"])
        for name, res in test_results.items()
    }

    if not args.no_plots:
        log.info("Generando figuras…")
        plot_roc_pr(test_results, y_test)
        plot_confusion_matrices(test_results)
        plot_model_comparison(test_results)
        plot_feature_importance(trained[best_name], X_test, y_test, best_name)
        plot_threshold_analysis(thresh_results, best_name)
        plot_lift_curves(lift_data_dict)
        plot_calibration(test_results, y_test)
        if not args.skip_learning_curve:
            log.info("  Curva de aprendizaje (puede tardar ~30s)…")
            plot_learning_curves(trained[best_name], X_train, y_train, best_name)
        plot_risk_segmentation(y_test.values, best_proba, best_name)
        log.info(f"  Figuras guardadas en: {CFG.figures_dir}/")

    log.info("Segmentación de riesgo…")
    bins   = [0, 0.25, 0.50, 0.75, 1.0]
    labels = ["Bajo (<25%)", "Medio (25-50%)", "Alto (50-75%)", "Crítico (>75%)"]
    seg    = pd.cut(best_proba, bins=bins, labels=labels)
    seg_df = pd.DataFrame({"segment": seg, "churn_real": y_test.values, "prob": best_proba})
    seg_summary = (
        seg_df.groupby("segment", observed=True)
        .agg(clientes=("churn_real", "count"), churn_real=("churn_real", "sum"), tasa=("churn_real", "mean"))
        .round(3)
    )
    log.info("\n" + seg_summary.to_string())

    log.info("Guardando modelo y resultados…")
    joblib.dump(trained[best_name], CFG.models_dir / "best_model.joblib")

    metrics_out = {
        name: {
            **cv_results[name],
            **{k: v for k, v in res["metrics"].items() if k not in ("report", "cm", "y_proba", "y_pred", "name")},
        }
        for name, res in test_results.items()
    }
    metrics_out["_meta"] = {
        "best_model":     best_name,
        "opt_threshold":  float(opt_threshold),
        "best_f1_thresh": thresh_results["best_f1"],
        "best_f2_thresh": thresh_results["best_f2"],
        "best_cost_thresh": {k: v for k, v in thresh_results["best_cost"].items()},
        "mcnemar":        mcnemar_matrix,
        "fn_cost":        CFG.fn_cost,
        "fp_cost":        CFG.fp_cost,
        "random_state":   CFG.random_state,
        "test_size":      CFG.test_size,
        "cv_folds":       CFG.cv_folds,
    }
    with open(CFG.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False, default=str)

    fi_cols = ["feature", "importance", "std"]
    try:
        from evaluate import get_feature_names
        clf  = trained[best_name].named_steps["clf"]
        X_t  = trained[best_name].named_steps["pre"].transform(X_test)
        from sklearn.inspection import permutation_importance as _pi
        perm = _pi(clf, X_t, y_test, n_repeats=10, random_state=CFG.random_state, n_jobs=CFG.n_jobs, scoring="roc_auc")
        fnames = get_feature_names(trained[best_name])
        fi_df = pd.DataFrame({
            "feature":    fnames[:len(perm.importances_mean)],
            "importance": perm.importances_mean.round(5),
            "std":        perm.importances_std.round(5),
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(CFG.output_dir / "feature_importance.csv", index=False)
    except Exception:
        pass

    seg_summary.to_csv(CFG.output_dir / "risk_segmentation.csv")

    log.info("=" * 65)
    log.info(f"✅ Pipeline completado.")
    log.info(f"   Mejor modelo  : {best_name}")
    log.info(f"   AUC-ROC (test): {test_results[best_name]['metrics']['auc_roc']:.4f}")
    log.info(f"   Threshold opt : {opt_threshold:.2f}  (F1={thresh_results['best_f1']['f1']:.4f})")
    log.info(f"   Modelo guardado: {CFG.models_dir}/best_model.joblib")
    log.info("=" * 65)


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de predicción de churn — Banorte")
    p.add_argument("--data",                default=str(CFG.data_path))
    p.add_argument("--no-plots",            action="store_true")
    p.add_argument("--skip-learning-curve", action="store_true", help="Omitir curva de aprendizaje (lenta)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
