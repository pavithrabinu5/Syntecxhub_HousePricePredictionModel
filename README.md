# House Price Prediction

A supervised machine learning project that predicts residential property prices
using Ridge Regression trained on the USA Housing dataset. The project covers
the full ML pipeline from data exploration and cleaning through model training,
evaluation, and serialisation for inference.

---

## Overview

This project implements a complete end-to-end regression pipeline to predict
house prices based on area-level demographic and property features. The goal
was to build an interpretable baseline model using Linear/Ridge Regression
and evaluate it rigorously using held-out test data and k-fold
cross-validation.

---

## Dataset

**Source:** USA Housing Dataset (Kaggle)  
**URL:** https://www.kaggle.com/datasets/vedavyasv/usa-housing  
**Rows:** 5,000  
**Features:** 6 (5 numeric, 1 categorical dropped)

| Feature | Type | Description |
|---------|------|-------------|
| Avg. Area Income | float | Median income of residents in the area |
| Avg. Area House Age | float | Average age of houses in the area |
| Avg. Area Number of Rooms | float | Average number of rooms per house |
| Avg. Area Number of Bedrooms | float | Average number of bedrooms per house |
| Area Population | float | Population of the area |
| Price | float | Target variable — sale price of the house |

The `Address` column was dropped as it is a free-text identifier with no
predictive value.

---

## Project Structure
```
house-price-prediction/
│
├── USA_Housing.csv                 # Raw dataset
├── housing_clean.csv               # Cleaned dataset after preprocessing
│
├── step1_explore.py                # Data loading, cleaning, EDA
├── step2_train.py                  # Feature engineering, train/test split, model training
├── step3_evaluate.py               # Evaluation metrics, cross-validation, plots
├── step4_predict.py                # Inference on new examples, model bundle export
│
├── lr_model.pkl                    # Serialised Ridge Regression model
├── scaler.pkl                      # Serialised StandardScaler
├── house_price_predictor.pkl       # Full inference bundle (model + scaler)
│
├── heatmap.png                     # Correlation heatmap
├── price_dist.png                  # Target variable distribution
├── actual_vs_predicted.png         # Predicted vs actual scatter plot
├── residuals.png                   # Residual analysis plot
├── feature_importance.png          # Coefficient magnitude chart
│
└── README.md
```

---

## Installation

**Requirements:** Python 3.8 or higher

Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

Install dependencies:
```bash
pip3 install scikit-learn pandas numpy matplotlib seaborn joblib
```

Download the dataset from Kaggle and place `USA_Housing.csv` in the
project root before running any scripts.

---

## Usage

Run each script in order:

**Step 1 — Data exploration and cleaning:**
```bash
python3 step1_explore.py
```
Loads the raw CSV, removes duplicates, checks for missing values, prints
descriptive statistics, generates correlation heatmap and price distribution
plot, and saves the cleaned dataset to `housing_clean.csv`.

**Step 2 — Model training:**
```bash
python3 step2_train.py
```
Reads the cleaned dataset, defines feature matrix and target vector, performs
an 80/20 train/test split, applies StandardScaler, trains a Ridge Regression
model with alpha=1.0, prints intercept and coefficients, and serialises the
model and scaler to disk.

**Step 3 — Evaluation:**
```bash
python3 step3_evaluate.py
```
Loads the saved model and scaler, generates predictions on the held-out test
set, computes RMSE, R-squared, MAE, runs 5-fold cross-validation, and
produces three diagnostic plots: actual vs predicted, residuals, and feature
importance.

**Step 4 — Inference:**
```bash
python3 step4_predict.py
```
Loads the model bundle, runs predictions on five example properties with
realistic feature values, and saves the full inference bundle to
`house_price_predictor.pkl`.

**Loading the saved model for inference:**
```python
import joblib
import pandas as pd

bundle   = joblib.load('house_price_predictor.pkl')
model    = bundle['model']
scaler   = bundle['scaler']

feature_cols = [
    'Avg. Area Income',
    'Avg. Area House Age',
    'Avg. Area Number of Rooms',
    'Avg. Area Number of Bedrooms',
    'Area Population'
]

new_house = pd.DataFrame([[79545, 5.68, 7.01, 4.0, 23086]],
                          columns=feature_cols)
price = model.predict(scaler.transform(new_house))[0]
print(f"Predicted price: ${price:,.0f}")
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Algorithm | Ridge Regression |
| Regularisation | L2 (alpha = 1.0) |
| Preprocessing | StandardScaler (zero mean, unit variance) |
| Train/Test Split | 80% train / 20% test (random_state=42) |
| Cross-Validation | 5-fold on test set |

Ridge Regression was chosen over plain Linear Regression to apply mild L2
regularisation, which reduces variance on correlated features without
sacrificing interpretability. Given the moderate feature collinearity observed
in the correlation matrix, this provides a more stable set of coefficients.

---

## Results

| Metric | Value |
|--------|-------|
| R-squared (Test) | 0.9180 |
| RMSE (Test) | $100,444 |
| MAE (Test) | $80,877 |
| Mean CV R-squared | 0.9167 |
| CV Std | 0.01 |

The model explains 91.8% of the variance in house prices on unseen data.
The low standard deviation across cross-validation folds (0.01) confirms
the model generalises consistently and is not overfitting to the training set.

---

## Feature Importance

Coefficients are reported after StandardScaler transformation, so they
represent the change in predicted price per one standard deviation increase
in each feature, holding all others constant.

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| Avg. Area Income | +231,682 | Increases price |
| Avg. Area House Age | +163,538 | Increases price |
| Area Population | +152,196 | Increases price |
| Avg. Area Number of Rooms | +120,684 | Increases price |
| Avg. Area Number of Bedrooms | +3,011 | Increases price |

Area income is the dominant predictor by a significant margin, followed by
house age and population. Number of bedrooms has minimal marginal impact
once income and room count are controlled for.

---

## Limitations

- The dataset is synthetically generated and does not reflect real-world
  housing market complexity such as location granularity, property condition,
  or market timing effects.
- Linear models assume a linear relationship between features and the target.
  Non-linear interactions are not captured.
- Predictions outside the training distribution (e.g. very low income values)
  can produce negative price estimates. Input validation should be applied
  before serving the model.
- No hyperparameter search was performed for the Ridge alpha value.

---

## Future Work

- Implement hyperparameter tuning for Ridge alpha using GridSearchCV
- Evaluate non-linear models such as Gradient Boosting or Random Forest
- Add input validation and clipping in the inference pipeline
- Build a simple REST API using FastAPI to serve predictions
- Experiment with polynomial feature expansion for non-linear relationships
- Replace the synthetic dataset with real MLS or Zillow data for
  production-grade evaluation
