
# Solución Machine Learning — Predicción de Churn
**Prueba Técnica Banorte · 2026**

## Estructura

```
Solución Machine Learning/
├── src/
│   ├── config.py           ← Toda la configuración como dataclass
│   ├── features.py         ← Carga, limpieza y feature engineering
│   ├── models.py           ← Definición de modelos y optimización de threshold
│   ├── evaluate.py         ← Métricas, pruebas estadísticas y todas las figuras
│   └── churn_pipeline.py   ← Orquestador principal (punto de entrada)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn_reto2.csv
├── output/
│   ├── metrics_summary.json
│   ├── feature_importance.csv
│   └── risk_segmentation.csv
├── figures/
│   ├── 00_eda.png
│   ├── 01_roc_pr.png
│   ├── 02_confusion_matrices.png
│   ├── 03_model_comparison.png
│   ├── 04_feature_importance.png
│   ├── 05_threshold_analysis.png
│   ├── 06_lift_gain.png
│   ├── 07_calibration.png
│   ├── 08_learning_curve.png     ← Solo con --skip-learning-curve omitido
│   └── 09_risk_segmentation.png
├── models/
│   └── best_model.joblib         ← Modelo serializado listo para scoring
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Uso

```bash
# Ejecución completa (~3-4 min con curva de aprendizaje)
python src/churn_pipeline.py

# Sin curva de aprendizaje (~1 min)
python src/churn_pipeline.py --skip-learning-curve

# Dataset alternativo
python src/churn_pipeline.py --data ruta/al/dataset.csv

# Sin figuras (solo métricas y modelo)
python src/churn_pipeline.py --no-plots
```

## Resultados

| Modelo | CV AUC-ROC | Test AUC-ROC | AUC-PR | F1 (Churn) | Brier |
|--------|-----------|-------------|--------|-----------|-------|
| Logistic Regression | 0.845 ± 0.012 | 0.841 | 0.627 | 0.614 | 0.169 |
| Random Forest | 0.848 ± 0.010 | 0.845 | 0.655 | 0.636 | 0.159 |
| Gradient Boosting | 0.847 ± 0.009 | 0.845 | 0.658 | 0.587 | 0.136 |
| **Voting Ensemble** | **0.850 ± 0.011** | **0.847** | **0.657** | **0.626** | 0.144 |

**Modelo seleccionado: Voting Ensemble**
- Threshold óptimo (F1): **0.42** → F1 = 0.639
- Threshold óptimo (costo negocio): **0.17** → costo = $34,950

## Features construidas

Además de las variables originales, se ingeniería:

| Feature | Descripción |
|---------|-------------|
| `ChargePerMonth` | TotalCharges / tenure (costo real por mes) |
| `ServiceCount` | Número de servicios activos |
| `LongTenure` | Flag: antigüedad ≥ 24 meses |
| `HighSpender` | Flag: cargo mensual ≥ mediana |
| `TenureGroup` | Bins de antigüedad: 0-12, 12-24, 24-48, 48+ |

## Segmentación de riesgo (Voting Ensemble, threshold 0.42)

| Segmento | Clientes | Churn real | Tasa |
|----------|----------|-----------|------|
| Bajo (<25%) | 649 | 37 | 5.7% |
| Medio (25-50%) | 307 | 78 | 25.4% |
| Alto (50-75%) | 309 | 148 | 47.9% |
| Crítico (>75%) | 144 | 111 | **77.1%** |
