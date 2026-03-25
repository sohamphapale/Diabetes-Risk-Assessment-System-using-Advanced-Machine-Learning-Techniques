import joblib
from pathlib import Path

from data_preprocessing import (
    load_diabetes_health_data,
    remove_redundant_columns,
    replace_na_values,
    transform_numerical_to_categorical,
    encode_categorical_features,
    prepare_feature,
    split_data,
)
from model_evaluation import train_random_forest, evaluate_classifier


def main():

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    FEATURES = ['invoice_quantity',
                'invoice_dollars',
                "Freight",
                'total_item_quantity',
                'total_item_dollars',
                ]

    TARGET = "flag_invoice"
    df = load_diabetes_health_data("./data/diabetes-data.csv")
    df = remove_redundant_columns(df)
    df = replace_na_values(df)
    df = transform_numerical_to_categorical(df)
    df = encode_categorical_features(df)
    X, y = prepare_feature(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    grid_search = train_random_forest(X_train, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test,
        y_test,
        'RandomForestClassifier'
    )

    model_path = model_dir / "predict_flag_invoice.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)
    

if __name__ == "__main__":
    main()
