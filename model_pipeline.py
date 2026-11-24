import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


def prepare_data(data_filename):
    """
    Charger et préparer les données du projet Customer Churn.
    """

    # Charger le dataset
    df = pd.read_csv(data_filename)

    # Encoder la colonne Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Supprimer les colonnes inutiles
    df = df.drop(["Surname", "Geography"], axis=1)

    # Séparer les variables indépendantes et dépendantes
    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # Supprimer les identifiants
    X = X.drop(["RowNumber", "CustomerId"], axis=1)

    # Diviser les données en train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraîner un modèle Random Forest.
    """

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluer les performances du modèle.
    """

    # Prédire les résultats
    y_pred = model.predict(X_test)

    # Calculer accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculer la matrice de confusion
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy score:", accuracy * 100)
    print("Confusion Matrix:")
    print(matrix)

    return accuracy, matrix


def save_model(model, filename):
    """
    Sauvegarder le modèle entraîné.
    """
    joblib.dump(model, filename)
    print("Modele sauvegarde avec succes")


def load_model(filename):
    """
    Charger un modèle sauvegardé.
    """
    model = joblib.load(filename)
    print("Modele charge avec succes")
    return model
