import json
import os
import typing as t

from dotenv import load_dotenv
from prediction_market_agent_tooling.benchmark.agents import (
    AbstractBenchmarkedAgent,
    RandomAgent,
)
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import (
    EvalautedQuestion,
    MarketSource,
    OutcomePrediction,
    Prediction,
    get_markets,
)

from tools.prediction_request_embedding import prediction_sentence_embedding
from tools.prediction_request_sme import prediction_request_sme


class SMEAgent(AbstractBenchmarkedAgent):
    def evaluate(self, market_question: str) -> EvalautedQuestion:
        return EvalautedQuestion(question=market_question, is_predictable=True)

    def research(self, market_question: str) -> str:
        return ""  # Research included in `predict`

    def predict(
        self, market_question: str, researched: str, evaluated: EvalautedQuestion
    ) -> Prediction:
        load_dotenv()
        response = prediction_request_sme.run(
            tool="prediction-online-sme",
            prompt=market_question,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "google_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            },
        )
        p = json.loads(response[0])
        return Prediction(
            evaluation=evaluated,
            outcome_prediction=OutcomePrediction(
                p_yes=p["p_yes"],
                confidence=p["confidence"],
                info_utility=p["info_utility"],
            ),
        )


class EmbeddingAgent(AbstractBenchmarkedAgent):
    def evaluate(self, market_question: str) -> EvalautedQuestion:
        return EvalautedQuestion(question=market_question, is_predictable=True)

    def research(self, market_question: str) -> str:
        return ""  # Research included in `predict`

    def predict(
        self, market_question: str, researched: str, evaluated: EvalautedQuestion
    ) -> Prediction:
        load_dotenv()
        response = prediction_sentence_embedding.run(
            tool="prediction-sentence-embedding-conservative",
            prompt=market_question,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "google_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            },
        )
        p = json.loads(response[0])
        return Prediction(
            evaluation=evaluated,
            outcome_prediction=OutcomePrediction(
                p_yes=p["p_yes"],
                confidence=p["confidence"],
                info_utility=p["info_utility"],
            ),
        )


def main(
    num_markets: int,
    output: str = "./benchmark_report.md",
    reference: MarketSource = MarketSource.MANIFOLD,
    max_workers: int = 1,
    cache_path: t.Optional[str] = "predictions_cache.json",
    only_cached: bool = False,
) -> None:
    markets = get_markets(number=num_markets, source=reference)
    markets_deduplicated = list(({m.question: m for m in markets}.values()))
    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=[
            RandomAgent(agent_name="random", max_workers=max_workers),
            SMEAgent(agent_name="sme", max_workers=max_workers),
            EmbeddingAgent(agent_name="embedding", max_workers=max_workers),
        ],
        cache_path=cache_path,
        only_cached=only_cached,
    )

    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)


if __name__ == "__main__":
    main(num_markets=20)
