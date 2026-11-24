import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline"
    )

    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Nom du fichier dataset"
    )

    parser.add_argument(
        "--model", default="classifier.joblib", help="Nom du fichier modèle"
    )

    args = parser.parse_args()

    # 1. Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(args.data)

    # 2. Entraînement
    model = train_model(X_train, y_train)

    # 3. Évaluation
    evaluate_model(model, X_test, y_test)

    # 4. Sauvegarde
    save_model(model, args.model)

    # 5. Chargement
    load_model(args.model)

    print("Pipeline execute avec succes")


if __name__ == "__main__":
    main()
