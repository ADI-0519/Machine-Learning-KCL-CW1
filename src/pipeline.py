from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from config import categorical_cols, random_state



def preprocessing(X):
    """Builds two preprocessors, scaled for linear models (scales numeric, one hot encodes categoricals with drop='first') and unscaled for tree models (no scaling, one hot encoding is dense to avoid sparse issues)."""
    numeric_cols = []
    for column in X.columns:
        if column not in categorical_cols:
            numeric_cols.append(column)

    onehot_encoding_linear = OneHotEncoder(drop="first",handle_unknown="ignore")
    onehot_encoding_tree = OneHotEncoder(drop=None,handle_unknown="ignore", sparse_output=False)

    numerical_tf_scaled = StandardScaler()

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num",numerical_tf_scaled,numeric_cols),
            ("cat",onehot_encoding_linear,categorical_cols)
        ],
        remainder="drop"
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("num","passthrough",numeric_cols),
            ("cat",onehot_encoding_tree,categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor_scaled,preprocessor_unscaled


def build_linear_pipeline(preprocessor):
    """Builds and returns the linear regression model pipeline."""
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", LinearRegression())
        ]
    )

    return pipe


def build_ridge_pipeline(preprocessor,alpha):
    """Builds and returns the ridge regression model pipeline."""
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", Ridge(alpha=alpha))
        ]
    )

    return pipe 

def build_gradientboost_pipeline(preprocessor, **kwargs):
    """Builds and returns the gradient boost pipeline."""
    model = GradientBoostingRegressor(random_state=random_state, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model", model)
        ]
    )

    return pipe

def build_histgradientboost_pipeline(preprocessor, **kwargs):
    """Builds and returns the HistGradientBoost pipeline."""
    model = HistGradientBoostingRegressor(random_state=random_state, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )

    return pipe

def build_extratrees_pipeline(preprocessor, **kwargs):
    """Builds and returns the extra trees model pipeline."""

    model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe


def build_adaboost_pipeline(preprocessor, base_max_depth=2, **kwargs):
    """Builds and returns the AdaBoost pipeline."""
    base = DecisionTreeRegressor(max_depth=base_max_depth,random_state=random_state)
    model = AdaBoostRegressor(estimator=base,random_state=random_state,**kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe

def build_randomforest_pipeline(preprocessor, **kwargs):
    """Builds and returns the RandomForest pipeline."""
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe

def build_xgboost_pipeline(preprocessor, **kwargs):
    """Builds and returns the XGBoost model pipeline."""
    model = XGBRegressor(random_state=random_state, n_jobs=-1, objective="reg:squarederror", **kwargs)
    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe

def build_stacking_pipeline(preprocessor, estimators):
    """Builds and returns the stacking pipeline."""
    meta = Ridge(alpha=1.0)
    model = StackingRegressor(estimators=estimators,final_estimator=meta,passthrough=False,n_jobs=-1)

    pipe = Pipeline(
        steps=[
            ("preprocess",preprocessor),
            ("model",model)
        ]
    )
    return pipe