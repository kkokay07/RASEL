# example_classify.py
import argparse
import pandas as pd
import joblib


def main(input_file, output_file):
    # Load model and features
    model = joblib.load("model/rasel_ensemble_model.joblib")
    feature_names = pd.read_csv("model/feature_names.txt", header=None).squeeze().tolist()

    # Read input data
    data = pd.read_csv(input_file)
    sample_ids = data["Individuals"]
    X = data[feature_names]

    # Predict classes and probabilities
    predictions = model.predict(X)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        df_out = pd.DataFrame({
            "Individuals": sample_ids,
            "Predicted_Class": predictions,
            "Dairy_Prob": probabilities[:, 1],
            "Draft_Prob": probabilities[:, 0]
        })
    else:
        df_out = pd.DataFrame({
            "Individuals": sample_ids,
            "Predicted_Class": predictions
        })

    # Save results
    df_out.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with genotype data")
    parser.add_argument("--output", required=True, help="Output CSV with predictions")
    args = parser.parse_args()
    main(args.input, args.output)


# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Individuals", "Class"])
    y = df["Class"]
    return X, y, df.columns[2:]  # exclude 'Individuals' and 'Class'


def train_model(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    lr = LogisticRegression(max_iter=1000)

    ensemble = VotingClassifier(estimators=[
        ("rf", rf),
        ("gb", gb),
        ("xgb", xgb),
        ("lr", lr)
    ], voting="soft")

    ensemble.fit(X, y)
    return ensemble


def main():
    X, y, feature_names = load_data("model/training_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model/rasel_ensemble_model.joblib")
    pd.Series(feature_names).to_csv("model/feature_names.txt", index=False, header=False)


if __name__ == "__main__":
    main()

