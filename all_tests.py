
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)

print("\n TEST 1 : PREPARATION DES DONNEES")
X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)
print("TEST 1 REUSSI")

print("\nTEST 2 : ENTRAINEMENT ")
model = train_model(X_train, y_train)
print("Modele :", model)
print("TEST 2 REUSSI")

print("\nTEST 3 : EVALUATION ")
accuracy, matrix = evaluate_model(model, X_test, y_test)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("TEST 3 REUSSI")

print("\n  TEST 4 : SAUVEGARDE ")
save_model(model, "test_classifier.joblib")
print("TEST 4 REUSSI")

print("\n TEST 5 : CHARGEMENT ")
loaded_model = load_model("test_classifier.joblib")
print("Modele charge :", loaded_model)
print("TEST 5 REUSSI")

print("\n TOUS LES TESTS SONT REUSSIS ")
