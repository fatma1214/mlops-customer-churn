import pandas as pd
from model_pipeline import train_model, evaluate_model


def test_train_model():
    X_train = pd.DataFrame({
        "CreditScore": [600, 700, 650, 720],
        "Age": [30, 40, 35, 45]
    })
    y_train = pd.Series([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_evaluate_model():
    X_test = pd.DataFrame({
        "CreditScore": [600, 700],
        "Age": [30, 40]
    })
    y_test = pd.Series([0, 1])

    model = train_model(X_test, y_test)

    accuracy, matrix = evaluate_model(model, X_test, y_test)

    assert 0 <= accuracy <= 1
    assert matrix.shape == (2, 2)
