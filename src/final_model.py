import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from config import train_path, test_path, submission_path, random_state
from sklearn.base import clone

from pipeline import build_gradientboost_pipeline, build_stacking_pipeline, preprocessing, build_linear_pipeline


def load_data():
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X = train_data.drop(columns=["outcome"])
    Y = train_data["outcome"]
    return X,Y, test_data

def main():
    X_train, y_train, test = load_data()
    preprocessor_scaled, preprocessor_unscaled = preprocessing(X_train)

    gbr_pipe = build_gradientboost_pipeline(
        preprocessor_unscaled,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=1,
        n_estimators=600,
        subsample=0.7,
    )
    gbr_base = clone(gbr_pipe.named_steps["model"])  # stacking expects estimators, not pipelines

    hgb_base = HistGradientBoostingRegressor(
        random_state=random_state,
        learning_rate=0.03,
        max_depth=3,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.1,
    )   

    estimators = [
        ("gbr", gbr_base),
        ("hgb", hgb_base),
    ]

    final_pipe = build_stacking_pipeline(preprocessor_unscaled, estimators)

    # Fit on all training data
    final_pipe.fit(X_train, y_train)

    yhat = final_pipe.predict(test)

    # Format submission:
    # This is a single-column CSV with predictions
    out = pd.DataFrame({'yhat': yhat})
    out.to_csv(submission_path, index=False) # k-no

if __name__ == "__main__":
    main()