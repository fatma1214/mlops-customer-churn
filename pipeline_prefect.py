import argparse
import subprocess
from pathlib import Path

from prefect import task, flow

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
)

# ==================================
# DOSSIER POUR LES ERREURS
# ==================================

ERROR_DIR = Path("errors")


def save_result(filename, result):
    ERROR_DIR.mkdir(exist_ok=True)

    output = result.stdout or ""
    error = result.stderr or ""

    content = output + "\n" + error

    if result.returncode != 0:
        content += (
            f"\nCommand failed with exit code: " f"{result.returncode}\n"
        )
    else:
        content += "\nExecution successful.\n"

    (ERROR_DIR / filename).write_text(
        content,
        encoding="utf-8",
    )


# ==================================
# TASKS PRINCIPALES
# ==================================


@task(name="Install Dependencies")
def install_dependencies():
    result = subprocess.run(
        ["python", "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True,
    )
    save_result("install.txt", result)


@task(name="Prepare Data")
def prepare_data_task():
    return prepare_data("Churn_Modelling.csv")


@task(name="Train Model")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Save Model")
def save_model_task(model):
    save_model(model, "classifier.joblib")


@task(name="Load Model")
def load_model_task():
    return load_model("classifier.joblib")


@task(name="Evaluate Model")
def evaluate_model_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ==================================
# TASKS CODE QUALITY
# ==================================


@task(name="Format Code")
def format_code():
    result = subprocess.run(
        [
            "black",
            "--line-length",
            "79",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        capture_output=True,
        text=True,
    )
    save_result("black.txt", result)


@task(name="Check Code Quality")
def check_code_quality():
    result = subprocess.run(
        [
            "flake8",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        capture_output=True,
        text=True,
    )
    save_result("flake8.txt", result)


@task(name="Check Code Security")
def check_code_security():
    result = subprocess.run(
        [
            "bandit",
            "-r",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        capture_output=True,
        text=True,
    )
    save_result("bandit.txt", result)


@task(name="Run Unit Tests")
def run_tests():
    result = subprocess.run(
        ["pytest", "-v"],
        capture_output=True,
        text=True,
    )
    save_result("pytest.txt", result)


# ==================================
# FLOW CODE QUALITY
# ==================================


@flow(name="code")
def code():
    format_code()
    check_code_quality()
    check_code_security()
    run_tests()


# ==================================
# FLOW TRAIN
# ==================================


@flow(name="train")
def train():
    X_train, X_test, y_train, y_test = prepare_data_task()

    model = train_model_task(X_train, y_train)

    save_model_task(model)


# ==================================
# FLOW EVALUATE
# ==================================


@flow(name="evaluate")
def evaluate():
    model = load_model_task()

    X_train, X_test, y_train, y_test = prepare_data_task()

    evaluate_model_task(model, X_test, y_test)


# ==================================
# FLOW ALL
# ==================================


@flow(name="all")
def all():
    install_dependencies()

    code()

    X_train, X_test, y_train, y_test = prepare_data_task()

    model = train_model_task(X_train, y_train)

    save_model_task(model)

    evaluate_model_task(model, X_test, y_test)


# ==================================
# EXECUTION
# ==================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flow",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.flow == "code":
        code()

    elif args.flow == "train":
        train()

    elif args.flow == "evaluate":
        evaluate()

    elif args.flow == "all":
        all()

    else:
        print("Flow inconnu")
