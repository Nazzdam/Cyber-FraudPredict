# Cyber-FraudPredict: AI Agent Instructions

## Project Overview
End-to-end fraud detection system combining ML models (LightGBM + Logistic Regression), behavioral analytics, and market stress indicators to predict fraudulent/cyber-risky transactions.

## Key Architecture Patterns

### Data Pipeline
- **Source**: Kaggle `creditcard.csv` or synthetic transactions (20K records fallback)
- **Market Enrichment**: Synthetic daily features (USD/ZAR returns, VIX proxy, SARB repo rate) merged by transaction date
- **Feature Engineering**: 3-layer approach in [PredictorProgramme.ipynb](PredictorProgramme.ipynb):
  1. **Transactional**: `hour`, `weekday`, `AmountLog`, categorical features (DeviceType, MerchantCategory, Country)
  2. **Behavioral**: Rolling user stats (1h/24h transaction counts), historical amount mean/std, Z-score anomaly
  3. **Market Stress**: USD/ZAR volatility 7-day, VIX index, binary stress flag when abs(return) > 2% or VIX in 90th percentile

### Model Training Workflow
- **Preprocessing Pipeline**: `ColumnTransformer` + `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- **Class Imbalance**: SMOTE applied before model training (optional but recommended given ~0.2% fraud rate)
- **Primary Model**: LightGBM (production) with gradient boosting
- **Baseline**: Logistic Regression for comparison
- **Validation**: Stratified train/test split (seed=42), ROC-AUC, PR-AUC, confusion matrix, KS-statistic

### Explainability Integration
- **SHAP**: Global summary plots + per-transaction waterfall explanations
- **Purpose**: Transparency on feature contributions for regulatory compliance (FinTech/cybersecurity context)

## Critical Development Commands

### Running the Pipeline
```bash
# Execute notebook cells sequentially (Jupyter/VS Code)
# Cells are interdependent; run in order: Setup → Load Data → Market Enrichment → Features → Training → Evaluation → SHAP
```

### Required Dependencies
```python
# Core ML
lightgbm, scikit-learn, imbalanced-learn (SMOTE)

# Analysis & Visualization
pandas, numpy, matplotlib, seaborn, shap, joblib

# Install: pip install lightgbm scikit-learn imbalanced-learn shap
```

### Model Serialization
- Models saved as `.joblib` files in `models/` directory
- `preprocessor.joblib`: ColumnTransformer pipeline (critical for inference)
- `lgbm_model.joblib`, `logreg_baseline.joblib`: Trained estimators

## Project-Specific Conventions

### Naming & Typos (Document Current State)
- Column naming: snake_case (`is_fraud`, `user_tx_count_1h`, `UserAmountMean` — mixed convention)
- Common typos in notebook: `RandoomSeed` (should be `RandomSeed`), `scikitlearn` (should be `sklearn`)
- Market stress flag: triggered when `abs(UsdZarReturn) > 0.02` OR `VixIndex > 90th percentile`

### Feature Engineering Gotchas
- **Rolling Features**: Computed per-user window in seconds; left pointer approach for O(n) efficiency
- **Amount Anomaly**: Z-score computed as `(amount - UserAmountMean) / UserAmountStd`; fill NaN with 0 (no historical std)
- **Night Flag**: Condition is `(hour > 0 & hour < 6)` (typo: should use `and`, not `&` in Python logic)
- **Device-Country Mismatch**: Uses user's primary country (mode), not most recent

### Data Merge Order
1. Load transactions + synthetic categorical features
2. Create daily market stress series (date-indexed)
3. Merge market features onto transactions by `date` (many-to-one)
4. Forward-fill + backfill missing market data

## Integration Points

### External Data Sources
- **Kaggle creditcard.csv** expected path: `data/KaggleDataset.csv` (checks existence before synthetic fallback)
- **Market data simulation**: Self-contained; no external API calls

### Downstream Usage (Streamlit App)
- Expected pickle/joblib artifacts: preprocessor, LGBM model for real-time scoring
- App applies same feature engineering pipeline to new transactions before prediction
- Risk tier classification: Low/Medium/High based on model probability thresholds

## Common Workflows for AI Agents

1. **Debugging Cell Failures**: Check import statements and seed consistency (`np.random.seed(42)` at line 56)
2. **Feature Addition**: Modify feature engineering cell [7](PredictorProgramme.ipynb) before train/test split
3. **Model Tuning**: LightGBM hyperparameters in training cell [10]; SMOTE ratio in preprocessing
4. **Reproducibility**: Always set seed=42 at notebook start; regenerates synthetic data identically
5. **Performance Analysis**: ROC-AUC, PR-AUC curves in evaluation; SHAP waterfall for per-transaction audit

## Quick References
- **Main notebook**: [PredictorProgramme.ipynb](PredictorProgramme.ipynb)
- **Project README**: [README.md](README.md) (architecture diagram + feature list)
- **Data format**: 20K rows × ~15 features (transactional + market + behavioral)
- **Class distribution**: ~0.2% fraud (highly imbalanced)
