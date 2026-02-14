import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from config import categorical_cols, random_state


def preprocessing(X):
    numeric_cols = []
    for column in X.columns:
        if column not in categorical_cols:
            numeric_cols.append(column)

    cat_tf = OneHotEncoder(drop="first",handle_unknown="ignore")
    numerical_tf_scaled = StandardScaler()

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num",numerical_tf_scaled,numeric_cols),
            ("cat",cat_tf,categorical_cols)
        ],
        remainder="drop"
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("num","passthrough",numeric_cols),
            ("cat",cat_tf,categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor_scaled,preprocessor_unscaled


def build_linear_pipeline(preprocessor):
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", LinearRegression())
        ]
    )

    return pipe


def build_ridge_pipeline(preprocessor,alpha):
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", Ridge(alpha=alpha))
        ]
    )

    return pipe 

def build_gradientboost_pipeline(preprocessor, **kwargs):
    model = GradientBoostingRegressor(random_state=random_state, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", model)
        ]
    )

    return pipe