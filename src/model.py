import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from config import train_path, cv_folds, random_state
from pipeline import preprocessing, build_linear_pipeline, build_ridge_pipeline, build_gradientboost_pipeline, build_histgradientboost_pipeline, build_adaboost_pipeline, build_extratrees_pipeline, build_randomforest_pipeline, build_xgboost_pipeline, build_stacking_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

run_grids = False

def load_data():
    """Loads training CSV file"""
    train_data = pd.read_csv(train_path)
    X = train_data.drop(columns=["outcome"])
    Y = train_data["outcome"]
    return X,Y

def evaluate(pipe,X_train, y_train, name, cv):
    """Evaluates a pipeline using cross validated R^2 and prints mean ± standard deviation."""
    scores = cross_val_score(pipe,X_train, y_train, cv=cv, scoring="r2")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

def main():
    """Runs model comparisons and optional grid searches to select the best performing approach."""
    X_train, Y_train = load_data()

    cv = KFold(n_splits=cv_folds,shuffle=True,random_state=random_state)
    preprocessor_scaled, preprocessor_unscaled = preprocessing(X_train)

    # Linear Regression
    linear_pipe = build_linear_pipeline(preprocessor_scaled)
    evaluate(linear_pipe,X_train,Y_train, "Linear Regression", cv)


    # Ridge regression
    alphas = [0.01, 0.1, 1, 10, 100]

    for alpha in alphas:
        ridge_pipe = build_ridge_pipeline(preprocessor_scaled, alpha)
        evaluate(ridge_pipe, X_train, Y_train, f"Ridge alpha={alpha}", cv)


    # Gradient boosting
    gbr_pipe = build_gradientboost_pipeline(preprocessor_unscaled)
    evaluate(gbr_pipe, X_train, Y_train, "Grad boosting (default)", cv)

    # HGB
    hgb_pipe = build_histgradientboost_pipeline(preprocessor_unscaled)
    evaluate(hgb_pipe, X_train, Y_train, "HistGB (default)", cv)

    # Grid searches
    if run_grids:
        gbr_param_grid = {
            "model__n_estimators": [800,1000],
            "model__learning_rate": [0.02, 0.03],
            "model__max_depth": [2],
            "model__subsample": [0.7, 0.8]
        }

    
        gbr_pipe = build_gradientboost_pipeline(preprocessor_unscaled)
        grid = GridSearchCV(estimator=gbr_pipe, param_grid=gbr_param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=1)
        grid.fit(X_train, Y_train)

        print("Best R2:", grid.best_score_)
        print("Best params:", grid.best_params_)

        best_pipe = grid.best_estimator_
        evaluate(best_pipe, X_train, Y_train, "GBR tuned (5-fold)", cv)



        param_grid = {
            "model__max_iter": [200, 400, 800],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }

        grid_hgb = GridSearchCV(estimator=hgb_pipe, param_grid=param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=2)
        grid_hgb.fit(X_train, Y_train)

        print("Best R2:", grid_hgb.best_score_)
        print("Best params:", grid_hgb.best_params_)

        hgb_best_pipe = grid_hgb.best_estimator_
        evaluate(hgb_best_pipe, X_train, Y_train, "HistGB tuned (5-fold)", cv)

    # ExtraTrees

    et_pipe = build_extratrees_pipeline(
        preprocessor_unscaled,
        n_estimators=800,
        min_samples_leaf=1
    )
    evaluate(et_pipe, X_train, Y_train, "ExtraTrees (800 trees)", cv)

    # AdaBoost

    ada_pipe = build_adaboost_pipeline(
        preprocessor_unscaled,
        base_max_depth=2,
        n_estimators=500,
        learning_rate=0.05
    )
    evaluate(ada_pipe, X_train, Y_train, "AdaBoost (depth=2, 500)", cv)

    # Random forest

    rf_pipe = build_randomforest_pipeline(
        preprocessor_unscaled,
        n_estimators=600,
        min_samples_leaf=1
    )
    evaluate(rf_pipe, X_train, Y_train, "RandomForest (600 trees)", cv)

    # XGBoost

    xgb_pipe = build_xgboost_pipeline(preprocessor_unscaled, n_estimators=600, learning_rate=0.05, max_depth = 4, subsample=0.8, colsample_bytree=0.8)
    evaluate(xgb_pipe, X_train, Y_train, "XGBoost (baseline)", cv)

    # StackingRegressor

    gbr_base = build_gradientboost_pipeline(
        preprocessor_unscaled,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=1,
        n_estimators=600,
        subsample=0.7
    ).named_steps["model"]

    estimators = [
        ("gbr", clone(gbr_base)),
        ("hgb", HistGradientBoostingRegressor(
            random_state=random_state,
            learning_rate=0.03,
            max_depth=3,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=0.1
        ))
    ]

    stack_pipe = build_stacking_pipeline(preprocessor_unscaled, estimators)
    evaluate(stack_pipe, X_train, Y_train, "Stacking (GBR+HGB)", cv)


if __name__ == "__main__":
    main()

