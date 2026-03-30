import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import CFG


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), list(categorical_features)),
        ],
        remainder="drop",
    )


def build_all(numeric_features, categorical_features):
    pre = lambda: build_preprocessor(numeric_features, categorical_features)

    lr = Pipeline([
        ("pre", pre()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            C=0.3,
            max_iter=1000,
            solver="lbfgs",
            random_state=CFG.random_state,
        )),
    ])

    rf = Pipeline([
        ("pre", pre()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=CFG.random_state,
            n_jobs=CFG.n_jobs,
        )),
    ])

    gbt = Pipeline([
        ("pre", pre()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=CFG.random_state,
        )),
    ])

    ensemble = Pipeline([
        ("pre", pre()),
        ("clf", VotingClassifier(
            estimators=[
                ("lr",  LogisticRegression(class_weight="balanced", C=0.3, max_iter=1000, random_state=CFG.random_state)),
                ("rf",  RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=CFG.random_state, n_jobs=CFG.n_jobs)),
                ("gbt", GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=CFG.random_state)),
            ],
            voting="soft",
            weights=[1, 2, 2],
        )),
    ])

    return {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "Gradient Boosting":   gbt,
        "Voting Ensemble":     ensemble,
    }


def optimize_threshold(y_true, y_proba, fn_cost=None, fp_cost=None):
    fn_cost = fn_cost or CFG.fn_cost
    fp_cost = fp_cost or CFG.fp_cost

    thresholds = np.linspace(0.05, 0.95, 181)
    results = {}

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn  = int(((preds == 0) & (y_true == 0)).sum())
        fp  = int(((preds == 1) & (y_true == 0)).sum())
        fn  = int(((preds == 0) & (y_true == 1)).sum())
        tp  = int(((preds == 1) & (y_true == 1)).sum())

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cost   = fn * fn_cost + fp * fp_cost
        f2     = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0

        results[round(t, 3)] = dict(
            threshold=round(t, 3),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            f2=round(f2, 4),
            business_cost=round(cost, 2),
            tp=tp, fp=fp, tn=tn, fn=fn,
        )

    best_f1   = max(results.values(), key=lambda x: x["f1"])
    best_f2   = max(results.values(), key=lambda x: x["f2"])
    best_cost = min(results.values(), key=lambda x: x["business_cost"])

    return {
        "all": results,
        "best_f1":   best_f1,
        "best_f2":   best_f2,
        "best_cost": best_cost,
    }
