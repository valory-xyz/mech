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

        # TODO write reports and inspect

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
    excluded_questions: list[str] = [],
) -> None:
    markets = get_markets(
        number=num_markets, source=reference, excluded_questions=excluded_questions
    )
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
    main(
        num_markets=25,
        excluded_questions=[
            "By the end of 2026, will we have transparency into any useful internal pattern within a Large Language Model whose semantics would have been unfamiliar to AI and cognitive science in 2006?",
            "does manifold hate or love @america",
            "Will @firstuserhere coauthor a publication in AIstats, AAAI, ICLR or JMLR before end of 2024? ($11,000M subsidy)",
            "Will @firstuserhere coauthor a NeurIPS or ICML conference publication before end of 2024? (10,000 Mana subsidy)",
            "Will there be a >0 value liquidity event for me, a former Consensys Software Inc. employee, on my shares of the company?",
            "Will there be a >0 value liquidity event for me, a former Consensys employee, on my shares of the company by 2025?",
            "Will we find something showing equal or greater architectural advancement to Gobekli Tepe, from before 11,000 BC?",
            "Instant deepfakes of anyone by the end of 2027?",
            "Will Eliezer Yudkowsky win his $150,000 - $1,000 bet about UFOs not having a worldview-shattering origin?",
            "Will @firstuserhere author a bestselling book by the end of 2027? (10000 Mana subsidy)",
        ],
        reference=MarketSource.MANIFOLD,
        output="benchmark_report.manifold.md",
    )
    main(
        num_markets=30,
        reference=MarketSource.POLYMARKET,
        output="benchmark_report.polymarket.md",
    )
