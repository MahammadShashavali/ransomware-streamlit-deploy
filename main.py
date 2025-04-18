from src.data_preprocessing import load_data, clean_data, encode_labels, split_data
from src.feature_engineering import scale_features
from src.model import (
    train_random_forest,
    train_lightgbm,
    evaluate_model,
    save_model,
)
from src.utils import save_metrics

def main():
    df = load_data("data/raw/data_file.csv")
    df = clean_data(df)
    df = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    rf_model = train_random_forest(X_train_scaled, y_train)
    lgbm_model = train_lightgbm(X_train_scaled, y_train)

    rf_accuracy, rf_report = evaluate_model(rf_model, X_test_scaled, y_test)
    lgbm_accuracy, lgbm_report = evaluate_model(lgbm_model, X_test_scaled, y_test)

    print(f"\nRandom Forest Accuracy: {rf_accuracy}")
    print(f"\nLightGBM Accuracy: {lgbm_accuracy}")

    save_model(rf_model, "models/random_forest.pkl")
    save_model(lgbm_model, "models/lightgbm.pkl")

    save_metrics(
        {"RandomForest": {"accuracy": rf_accuracy, "report": rf_report},
         "LightGBM": {"accuracy": lgbm_accuracy, "report": lgbm_report}},
        "results/evaluation_metrics.json"
    )

if __name__ == "__main__":
    main()
