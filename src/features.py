import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import CFG


def load_raw(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or CFG.data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ChargePerMonth"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    df["LongTenure"]  = (df["tenure"] >= 24).astype(int)
    df["HighSpender"] = (df["MonthlyCharges"] >= df["MonthlyCharges"].median()).astype(int)

    bins   = [0, 12, 24, 48, np.inf]
    labels = [0, 1, 2, 3]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True).cat.codes

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer(pd.DataFrame(X)) if not isinstance(X, pd.DataFrame) else engineer(X)
