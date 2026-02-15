import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from config import categorical_cols, random_state



def preprocessing(X):
    numeric_cols = []
    for column in X.columns:
        if column not in categorical_cols:
            numeric_cols.append(column)

    cat_tf = OneHotEncoder(drop=None,handle_unknown="ignore")
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

def build_histgradientboost_pipeline(preprocessor, **kwargs):
    model = HistGradientBoostingRegressor(random_state=random_state, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )

    return pipe

def build_extratrees_pipeline(preprocessor, **kwargs):
    model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe


def build_adaboost_pipeline(preprocessor, base_max_depth=2, **kwargs):
    base = DecisionTreeRegressor(max_depth=base_max_depth,random_state=random_state)
    model = AdaBoostRegressor(estimator=base,random_state=random_state,**kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe

def build_spline_ridge_pipeline(preprocessor, spline_degree=3, n_knots=5, alpha=1.0):
    model = Ridge(alpha=alpha)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("spline", SplineTransformer(degree=spline_degree, n_knots=n_knots)),
            ("model",model)
        ]
    )
    return pipe

def build_randomforest_pipeline(preprocessor, **kwargs):
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe

def build_xgboost_pipeline(preprocessor, **kwargs):
    model = XGBRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe
