import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from config import train_path, random_state
from pipeline import preprocessing, build_gradientboost_pipeline

def load_data():
    df = pd.read_csv(train_path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"]
    return X,y

def get_feature_names(preprocessor):
    return preprocessor.get_feature_names_out()

def main():
    X,y = load_data()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X,y, test_size=0.2, random_state=random_state
    )

    pre_scaled, pre_unscaled = preprocessing(X_tr)

    pipe = build_gradientboost_pipeline(pre_unscaled, learning_rate=0.03, max_depth=2, min_samples_leaf=1, n_estimators=600,subsample=0.7)
    pipe.fit(X_tr,y_tr)

    result = permutation_importance(
        pipe,
        X_va,
        y_va,
        n_repeats=10,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1,
    )

    pre = pipe.named_steps["preprocess"]

    # Get names from fitted preprocessor
    feature_names = X_va.columns.to_numpy()

    imp_mean = result.importances_mean
    imp_std = result.importances_std

    print("n_raw_features:", len(feature_names))
    print("n_importances:", len(imp_mean))

    # Safety: if mismatch, fall back to generic names so the analysis still runs
    if len(feature_names) != len(imp_mean):
        feature_names = np.array([f"feature_{i}" for i in range(len(imp_mean))])

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance_mean": imp_mean, "importance_std": imp_std}
    ).sort_values("importance_mean", ascending=False)

    print("\nTop 15 permutation importances:")
    print(imp_df.head(15).to_string(index=False))

    top = imp_df.head(10).iloc[::-1]
    plt.figure()
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Permutation importance (drop in R²)")
    plt.title("Top 10 Feature Importances (Permutation)")
    plt.tight_layout()
    plt.savefig("reports/feature_importance_top10.png", dpi=200)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

if __name__ == "__main__":
    main()