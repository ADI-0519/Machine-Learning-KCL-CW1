import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from config import train_path, test_path, submission_path, random_state
from sklearn.base import clone
from pathlib import Path

from pipeline import build_gradientboost_pipeline, build_stacking_pipeline, preprocessing


def load_data():
    """Loads training and test CSV file."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X = train_data.drop(columns=["outcome"])
    Y = train_data["outcome"]
    return X,Y, test_data

def main():
    """Loads and runs the best model, producing a .csv file in submission folder."""
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
    gbr_base = clone(gbr_pipe.named_steps["model"])

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

    Path(submission_path).parent.mkdir(parents=True,exist_ok=True)
    out = pd.DataFrame({'yhat': yhat})
    out.to_csv(submission_path, index=False) # k-no
    print(f"Saved submission: {submission_path} (rows={len(yhat)})")

if __name__ == "__main__":
    main()