import typer
import pandas as pd
from packages.kongzii.customs.ofv_market_resolver.ofv_market_resolver import (
    run as ofv_run,
)
from packages.napthaai.customs.resolve_market_reasoning.resolve_market_reasoning import (
    Results,
    run as original_run,
)
from pydantic import SecretStr, ValidationError
from joblib import Memory

# File cache to not re-run the same questions.
MEMORY = Memory(".benchmark-cache", verbose=0)
APP = typer.Typer()


@MEMORY.cache
def ofv_run_cached(
    question: str,
    openai_api_key: SecretStr,
    serper_api_key: SecretStr,
) -> bool | None:
    return ofv_run(
        prompt=question,
        api_keys={
            "openai": openai_api_key.get_secret_value(),
            "serperapi": serper_api_key.get_secret_value(),
        },
    )


@MEMORY.cache
def run_original_resolver_cached(
    question: str,
    openai_api_key: SecretStr,
    google_api_key: SecretStr,
    google_engine_id: SecretStr,
) -> bool | None:
    try:
        dump = original_run(
            api_keys={
                "openai": openai_api_key.get_secret_value(),
                "google_api_key": google_api_key.get_secret_value(),
                "google_engine_id": google_engine_id.get_secret_value(),
            },
            tool="resolve-market-reasoning-gpt-4",
            prompt=question,
        )[0]
        return Results.model_validate_json(dump).has_occurred
    except ValueError:
        return None


@APP.command()
def full(
    data_path: str,
    openai_api_key: str,
    serper_api_key: str,
    google_api_key: str,
    google_engine_id: str,
) -> None:
    """
    Will run the prediction market resolver on all provided data and compare the results.

    Expects a tsv file with columns:
        - question
        - resolution (YES/NO, as currently resolved on Omen)
        - my_resolution (YES/NO, as resolved manually by you, used as ground truth)

    Example command:

    ```
    python packages/kongzii/customs/ofv_market_resolver/benchmark.py full markets.tsv {openai api key} {serper api key} {google api key} {google engine id}
    ```
    """
    df = pd.read_csv(data_path, sep="\t")

    # Run the resolution on all the data.
    df["ofv_resolution"] = df["question"].apply(
        lambda q: ofv_run_cached(
            q,
            openai_api_key=SecretStr(openai_api_key),
            serper_api_key=SecretStr(serper_api_key),
        )
    )
    df["new_original_resolution"] = df["question"].apply(
        lambda q: run_original_resolver_cached(
            q,
            openai_api_key=SecretStr(openai_api_key),
            google_api_key=SecretStr(google_api_key),
            google_engine_id=SecretStr(google_engine_id),
        )
    )
    # Normalise boolean to YES/NO/None.
    df["ofv_resolution"] = df["ofv_resolution"].apply(
        lambda r: "None" if r is None else "YES" if r else "NO"
    )
    df["new_original_resolution"] = df["new_original_resolution"].apply(
        lambda r: "None" if r is None else "YES" if r else "NO"
    )
    # Save all the predictions and separatelly these that are incorrect.
    df.to_csv("markets_resolved.tsv", sep="\t", index=False)
    df[df["ofv_resolution"] != df["my_resolution"]].to_csv(
        "markets_resolved_incorretly_by_ofv.tsv", sep="\t", index=False
    )

    # Calculate the accuracy.
    accuracy_current = sum(df["resolution"] == df["my_resolution"]) / len(df)
    accuracy_new_original = sum(
        df["new_original_resolution"] == df["my_resolution"]
    ) / len(df)
    accuracy_ofv = sum(df["ofv_resolution"] == df["my_resolution"]) / len(df)
    print(
        f"""
Current accuracy: {accuracy_current*100:.2f}%
Original's new run accuracy: {accuracy_new_original * 100:.2f}
OFV's accuracy: {accuracy_ofv*100:.2f}%
"""
    )


@APP.command()
def single(
    question: str,
    openai_api_key: str,
    serper_api_key: str,
) -> None:
    """
    Will run the prediction market resolver and print the result on a single question.

    Example command:

    ```
    python packages/kongzii/customs/ofv_market_resolver/benchmark.py single "Will McDonald's successfully buy back all its Israeli restaurants by 12 April 2024?" {openai api key} {serper api key}
    ```
    """
    ofv_run(
        question,
        openai_api_key=SecretStr(openai_api_key),
        serper_api_key=SecretStr(serper_api_key),
    )


if __name__ == "__main__":
    APP()
