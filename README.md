# Heavy Equipment Selling Price Prediction

Kaggle competition notebook for predicting used heavy-equipment sale prices (`TargetValue`) from asset, spec, and transaction features.

**Notebook:** [`23f2004898-notebook-2026t2.ipynb`](./23f2004898-notebook-2026t2.ipynb)  
**Competition:** Heavy Equipment Selling Price Prediction Challenge  
**Primary metric:** RMSLE (Root Mean Squared Logarithmic Error)

---

## Dataset

| Split | Rows | Columns |
|-------|-----:|--------:|
| Train | 138,701 | 50 (incl. `TargetValue`) |
| Test  | 15,000 | 49 |

Key raw fields include manufacture year, operational hours, utilization tier, equipment specs (`Spec_*`), cabin/config columns, region, and transaction date.

Notable data issues handled in the notebook:
- Mixed dtypes on import
- Sentinel manufacture years (`1001` → missing)
- `"None or Unspecified"` treated as missing
- Columns with **>80% missing** dropped
- High cardinality categoricals present in both train and test

---

## Approach

### 1. Cleaning & imputation
- Drop sparse columns (>80% null)
- Median imputation for numerics; categorical fill / native CatBoost handling for cats
- Date parsing from `TransactionDate`

### 2. Feature engineering
| Feature | Description |
|---------|-------------|
| `TransactionYear` / `Month` / `Quarter` | Time components from sale date |
| `AssetAge` | `TransactionYear − ManufactureYear` |
| `AgeSquared` | Non-linear age effect |
| `HoursPerYear` | Usage intensity (`OperationalHoursMeter / AssetAge`) |
| `HasOperationalHours` | Missingness indicator for hours |
| `HasVariantModifier` | Presence of variant modifier |
| `DescriptorLength` / `BaseClassLength` | Spec string lengths |
| `DescPrefix` / `DescSuffix` / `BaseNumber` / etc. | Parsed descriptor tokens |

Target was modeled in **log space** (`log1p` / `expm1`) to stabilize skewed prices.

### 3. Modeling
Baseline comparison with 5-fold CV (sklearn pipelines + `TransformedTargetRegressor`), then native **CatBoost** with categorical features.

---

## Model comparison (5-fold CV)

Sorted by RMSLE (lower is better):

| Model | R² | RMSE | RMSLE |
|-------|---:|-----:|------:|
| **CatBoostRegressor** | **0.890** | **8,761** | **0.217** |
| XGBRegressor | 0.823 | 10,978 | 0.266 |
| RandomForestRegressor | 0.752 | 12,925 | 0.298 |
| GradientBoostingRegressor | 0.745 | 13,170 | 0.314 |
| LinearRegression | 0.518 | 17,127 | 0.350 |
| DecisionTreeRegressor | 0.662 | 15,148 | 0.375 |
| Ridge | 0.370 | 20,687 | 0.472 |

**Winner:** CatBoost — best R² and lowest RMSLE among baselines.

---

## Hyperparameter tuning (CatBoost)

Random search + 3-fold CV improved CV RMSLE to **0.206**.

| Parameter | Tuned value | Final train config |
|-----------|------------:|-------------------:|
| iterations | 3000 | 8000 |
| learning_rate | 0.08 | 0.02 |
| depth | 10 | 10 |
| l2_leaf_reg | 10 | 10 |
| random_strength | 1 | 1.0 |
| bagging_temperature | 1 | 1 |
| border_count | 64 | 64 |

Reported best score after final training setup: **~0.202 RMSLE**.

Final predictions written to `submission.csv` (`TransactionID`, `TargetValue`) for 15,000 test rows.

---

## Top predictive features (CatBoost importance)

| Rank | Feature | Importance |
|-----:|---------|-----------:|
| 1 | `FunctionalClassification` | 14.22 |
| 2 | `ManufactureYear` | 13.67 |
| 3 | `Spec_FullDescriptor` | 11.00 |
| 4 | `DescSuffix` | 9.12 |
| 5 | `AssetAge` | 8.51 |
| 6 | `TransactionYear` | 8.04 |
| 7 | `AgeSquared` | 6.45 |
| 8 | `CabinType` | 4.07 |
| 9 | `AssetScaleFactor` | 4.05 |
| 10 | `Spec_SubClass` | 3.63 |

Price is driven mainly by **equipment class**, **age/year**, and **spec descriptors**, with cabin/scale as secondary signals. Raw hours matter less than engineered age/intensity features.

---

## Tech stack

- Python, pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (pipelines, CV, baselines)
- XGBoost, CatBoost
- category_encoders (explored for high-cardinality paths)

---

## How to run

1. Open the notebook in Kaggle (or locally with the competition data mounted).
2. Ensure paths point to:
   - `train.csv`, `test.csv`, `sample_submission.csv`
3. Run all cells → produces `submission.csv`.

```bash
# Local (example)
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost category_encoders
```

---

## Project structure

```text
price-prediction-project/
├── README.md
└── 23f2004898-notebook-2026t2.ipynb   # Final Aug 15 analysis + submission pipeline
```

---

## Key takeaways

1. Log-transforming the target improves RMSLE for skewed sale prices.
2. Dropping >80% missing columns and engineering **age / intensity / date** features beat raw baselines.
3. **CatBoost** with native categoricals clearly outperformed linear, tree, RF, GBM, and XGB in CV.
4. Tuned CatBoost reduced RMSLE from **~0.217 → ~0.206** (CV), with final training score around **0.202**.
