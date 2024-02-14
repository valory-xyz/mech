import json
import os
import random

import functions_framework
from dotenv import load_dotenv
from flask.wrappers import Request
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.deploy.gcp.deploy import (
    deploy_to_gcp,
    run_deployed_gcp_function,
    schedule_deployed_gcp_function,
)
from prediction_market_agent_tooling.deploy.gcp.utils import gcp_function_is_active
from prediction_market_agent_tooling.markets.data_models import AgentMarket
from prediction_market_agent_tooling.markets.markets import MarketType

from tools.prediction_request_sme import prediction_request_sme


class SMEAgent(DeployableAgent):
    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        return random.sample(markets, 1)

    def answer_binary_market(self, market: AgentMarket) -> bool:
        load_dotenv()
        response = prediction_request_sme.run(
            tool="prediction-online-sme",
            prompt=market.question,
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "google_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            },
        )
        p = json.loads(response[0])
        return p["p_yes"] > 0.5


@functions_framework.http
def main(request: Request) -> str:
    """
    Entrypoint for the deployed function.
    """
    SMEAgent().run(market_type=MarketType.MANIFOLD)
    return "Success"


if __name__ == "__main__":
    """
    Script to execute locally to deploy the agent to GCP.
    """
    deployable_agent_name = "sme_agent"
    load_dotenv()
    fname = deploy_to_gcp(
        requirements_file=None,
        extra_deps=["git+https://github.com/polywrap/evo.researcher.git@peter/pmat"],
        function_file=os.path.abspath(__file__),
        market_type=MarketType.MANIFOLD,
        api_keys={
            "MANIFOLD_API_KEY": os.environ["MANIFOLD_API_KEY"],
            "GOOGLE_SEARCH_API_KEY": os.environ["GOOGLE_SEARCH_API_KEY"],
            "GOOGLE_SEARCH_ENGINE_ID": os.environ["GOOGLE_SEARCH_ENGINE_ID"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "DEPLOYABLE_AGENT_NAME": deployable_agent_name,
        },
        memory=512,
    )

    # Check that the function is deployed
    assert gcp_function_is_active(fname), "Failed to deploy the function"

    # Run the function
    response = run_deployed_gcp_function(fname)
    assert response.ok, "Failed to run the deployed function"

    # Schedule the function
    schedule_deployed_gcp_function(fname, cron_schedule="0 */2 * * *")
