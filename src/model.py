import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from config import train_path, cv_folds, random_state
from pipeline import preprocessing, build_linear_pipeline, build_ridge_pipeline, build_gradientboost_pipeline

def load_data():
    train_data = pd.read_csv(train_path)
    X = train_data.drop(columns=["outcome"])
    Y = train_data["outcome"]
    return X,Y

def evaluate(pipe,X_train, y_train, name):
    cv = KFold(n_splits=cv_folds,shuffle=True,random_state=random_state)
    scores = cross_val_score(pipe,X_train, y_train, cv=cv, scoring="r2")
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def main():

    X_train, Y_train = load_data()
    print("train shape:", X_train.shape)
    preprocessor_scaled, preprocessor_unscaled = preprocessing(X_train)

    linear_pipe = build_linear_pipeline(preprocessor_scaled)
    evaluate(linear_pipe,X_train,Y_train, "Linear Regression")



    alphas = [0.01, 0.1, 1, 10, 100]

    for alpha in alphas:
        ridge_pipe = build_ridge_pipeline(preprocessor_scaled, alpha)
        evaluate(ridge_pipe, X_train, Y_train, f"Ridge alpha={alpha}")


    gbr_pipe = build_gradientboost_pipeline(preprocessor_unscaled)
    evaluate(gbr_pipe, X_train, Y_train, "Grad boosting (default)")

if __name__ == "__main__":
    main()

