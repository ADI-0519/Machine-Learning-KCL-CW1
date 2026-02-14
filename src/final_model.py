import pandas as pd
from config import train_path, test_path, submission_path

from pipeline import preprocessing, build_linear_pipeline


def load_data():
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X = train_data.drop(columns=["outcome"])
    Y = train_data["outcome"]
    return X,Y, test_data

def main():
    X_train, Y_train, test = load_data()
    preprocessor_scaled, preprocessor_unscaled = preprocessing(X_train)

    pipe = build_linear_pipeline(preprocessor_scaled)

    pipe.fit(X_train,Y_train)
    yhat = pipe.predict(test)

    # Format submission:
    # This is a single-column CSV with predictions
    out = pd.DataFrame({'yhat': yhat})
    out.to_csv(submission_path, index=False) # k-no

if __name__ == "__main__":
    main()