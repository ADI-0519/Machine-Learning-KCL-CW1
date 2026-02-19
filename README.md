# Machine-Learning-KCL-CW1

Module: Machine Learning (Coursework)
Task: Train a supervised regression model to predict the "outcome" column from tabular features and minimise out of sample R² error on a hidden test set. 

This repository contains:

1. Full exploratory analysis (EDA notebook)
2. Modular preprocessing + model pipelines
3. Cross-validated model comparison
4. Hyperparameter tuning (GridSearchCV)
5. Model stacking (ensemble)
6. Permutation feature importance analysis
7. Final prediction script for submission generation

## Objective
To minimise the generalisation error measured by out of sample R² on the hidden test set.
All models are evaluated using:

- 5 fold cross validation
- R² scoring
- Fixed random_state for reproducibility

## Preprocessing Pipeline
Two preprocessing strategies are implemented:

1. Scaled Pipeline (Linear Models)
- StandardScaler applied to numeric features
- OneHotEncoder with drop="first" for categorical features
Used for:
- Linear Regression
- Ridge Regression

2. Unscaled Pipeline (Tree/Boosting Models)
- Numeric features passed through unchanged
- OneHotEncoder (dense output)
Used for:
- Gradient Boosting
- HistGradientBoosting
- Random Forest
- ExtraTrees
- AdaBoost
- XGBoost
- Stacking ensemble

All preprocessing is implemented using ColumnTransformer inside sklearn Pipeline objects to prevent data leakage.

## Models Implemented
The following models were implemented and compared using cross-validation in src/model.py:
- Linear Regression
- Ridge Regression
- GradientBoostingRegressor (GBR)
- HistGradientBoostingRegressor (HGB)
- RandomForestRegressor
- ExtraTreesRegressor
- AdaBoostRegressor
- XGBRegressor (XGBoost)
- StackingRegressor (GBR + HGB + ridge meta-learner)

Hyperparameter tuning was performed using GridSearchCV via setting run_grids=True.

## Final Model

The final model is a stacked ensemble combining:
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- Ridge meta-learner

This was selected based on cross validated R² performance.

## Feature Importance
Permutation importance is computed on a hold out validation split using:
```bash
python src/importance.py
```

Which produces:

```bash
reports/feature_importance_top10.png
```

## Steps

### 1. Clone Repository

```bash
git clone https://github.com/ADI-0519/Machine-Learning-KCL-CW1.git

```

### 2. Install dependencies.

```bash
pip install -r requirements.txt

```

### 3. Run the final_model.py file

To generate predictions for the hidden test set:

```bash
python src/final_model.py
```
This creates:

```bash
submission/CW1_submission_23149795.csv
```

Which has a single column yhat, no index and one prediction per test observation.

### 4. Reproducibility

- All randomness controlled via random_state = 123
- Pipelines used to prevent data leakage
- Cross-validation used for robust evaluation
- Requirements pinned via requirements.txt

## Author
Aditya Ranjan - K23149795
King's College London
Coursework for Machine Learning 5CCSAMLF
