from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    data_path:   Path = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn_reto2.csv"
    output_dir:  Path = BASE_DIR / "output"
    figures_dir: Path = BASE_DIR / "figures"
    models_dir:  Path = BASE_DIR / "models"

    test_size:    float = 0.20
    random_state: int   = 42
    cv_folds:     int   = 5
    n_jobs:       int   = -1

    low_confidence_threshold: float = 0.60

    fn_cost: float = 500.0
    fp_cost: float = 50.0

    NUMERIC_RAW: tuple = (
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    )
    NUMERIC_ENGINEERED: tuple = (
        "ChargePerMonth", "ServiceCount", "LongTenure", "HighSpender",
        "TenureGroup",
    )
    CATEGORICAL: tuple = (
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    )

    RED:   str = "C8102E"
    DARK:  str = "1A1A2E"
    GOLD:  str = "C9A84C"
    GRAY:  str = "94A3B8"
    GREEN: str = "10B981"

CFG = Config()
