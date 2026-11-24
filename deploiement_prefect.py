
from prefect import serve

from pipeline_prefect import all, train


if __name__ == "__main__":

    all_deployment = all.to_deployment(
        name="ml-pipeline-all",
        cron="0 0 * * *",
    )

    train_deployment = train.to_deployment(
        name="ml-pipeline-train",
        cron="0 1 * * *",
    )

    serve(
        all_deployment,
        train_deployment,
    )
