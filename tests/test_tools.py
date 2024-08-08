# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""This module contains tool tests."""
from typing import List, Any

from packages.gnosis.customs.omen_tools import omen_buy_sell
from packages.victorpolisetty.customs.dalle_request import dalle_request
from packages.napthaai.customs.prediction_request_rag import prediction_request_rag
from packages.jhehemann.customs.prediction_sentence_embeddings import prediction_sentence_embeddings
from packages.gnosis.customs.ofv_market_resolver import ofv_market_resolver
from packages.napthaai.customs.prediction_request_rag_cohere import (
    prediction_request_rag_cohere,
)
from packages.napthaai.customs.prediction_request_reasoning import (
    prediction_request_reasoning,
)
from packages.napthaai.customs.prediction_url_cot import prediction_url_cot
from packages.valory.customs.prediction_request import prediction_request
from packages.valory.skills.task_execution.utils.apis import KeyChain
from packages.valory.skills.task_execution.utils.benchmarks import TokenCounterCallback
from tests.constants import (
    OPENAI_SECRET_KEY,
    STABILITY_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_ENGINE_ID,
    CLAUDE_API_KEY,
    REPLICATE_API_KEY,
    NEWS_API_KEY,
    OPENROUTER_API_KEY,
    GNOSIS_RPC_URL,
    GEMINI_API_KEY, SERPER_API_KEY
)


class BaseToolTest:
    """Base tool test class."""

    keys = KeyChain(
        {
            "openai": [OPENAI_SECRET_KEY],
            "stabilityai": [STABILITY_API_KEY],
            "google_api_key": [GOOGLE_API_KEY],
            "google_engine_id": [GOOGLE_ENGINE_ID],
            "anthropic": [CLAUDE_API_KEY],
            "replicate": [REPLICATE_API_KEY],
            "newsapi": [NEWS_API_KEY],
            "openrouter": [OPENROUTER_API_KEY],
            "gnosis_rpc_url": [GNOSIS_RPC_URL],
            "gemini": [GEMINI_API_KEY],
            "serperapi": [SERPER_API_KEY],
        }
    )
    models: List = [None]
    tools: List[str]
    prompts: List[str]
    tool_module: Any = None
    tool_callable: str = "run"

    def _validate_response(self, response: Any) -> None:
        """Validate response."""
        assert type(response) == tuple, "Response of the tool must be a tuple."
        assert len(response) == 5, "Response must have 5 elements."
        assert type(response[0]) == str, "Response[0] must be a string."
        assert type(response[1]) == str, "Response[1] must be a string."
        assert (
            type(response[2]) == dict or response[2] is None
        ), "Response[2] must be a dictionary or None."
        assert (
            type(response[3]) == TokenCounterCallback or response[3] is None
        ), "Response[3] must be a TokenCounterCallback or None."
        assert type(response[4]) == KeyChain, "Response[4] must be a KeyChain object."

    def test_run(self) -> None:
        """Test run method."""
        assert self.tools, "Tools must be provided."
        assert self.prompts, "Prompts must be provided."
        assert self.tool_module, "Callable function must be provided."

        for model in self.models:
            for tool in self.tools:
                for prompt in self.prompts:
                    kwargs = dict(
                        prompt=prompt,
                        tool=tool,
                        api_keys=self.keys,
                        counter_callback=TokenCounterCallback(),
                        model=model,
                    )
                    func = getattr(self.tool_module, self.tool_callable)
                    response = func(**kwargs)
                    self._validate_response(response)


class TestPredictionOnline(BaseToolTest):
    """Test Prediction Online."""

    tools = prediction_request.ALLOWED_TOOLS
    models = prediction_request.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_request


class TestPredictionRAG(BaseToolTest):
    """Test Prediction RAG."""

    tools = prediction_request_rag.ALLOWED_TOOLS
    models = prediction_request_rag.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_request_rag


class TestPredictionRAGCohere(BaseToolTest):
    """Test Prediction RAG using cohere model."""

    tools = prediction_request_rag_cohere.ALLOWED_TOOLS
    models = prediction_request_rag_cohere.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_request_rag_cohere


class TestPredictionReasoning(BaseToolTest):
    """Test Prediction Reasoning."""

    tools = prediction_request_reasoning.ALLOWED_TOOLS
    models = prediction_request_reasoning.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_request_reasoning


class TestPredictionCOT(BaseToolTest):
    """Test Prediction COT."""

    tools = prediction_url_cot.ALLOWED_TOOLS
    models = prediction_url_cot.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_url_cot


class TestOmenTransactionBuilder(BaseToolTest):
    """Test Prediction COT."""

    tools = omen_buy_sell.ALLOWED_TOOLS
    models = omen_buy_sell.ALLOWED_MODELS
    market_id = (
        "0x7323440218011988f0e431e19298d1921e41197f"  # to be resolved in 8 years
    )
    prompts = [
        f"The sender 0x669F3CD2015eB9298b3feA01FCBb034068FE2D3f wants to buy 0 yes Tokens from market {market_id}."
    ]
    tool_module = omen_buy_sell

    def _validate_response(self, response: Any) -> None:
        super()._validate_response(response)
        expected_num_tx_params = 2
        assert len(response[2].keys()) == expected_num_tx_params


class TestDALLEGeneration(BaseToolTest):
    """Test DALL-E Generation."""

    tools = dalle_request.ALLOWED_TOOLS
    models = dalle_request.ALLOWED_MODELS
    prompts = [
        "Generate an image of a futuristic cityscape."
    ]
    tool_module = dalle_request


class TestPredictionSentenceEmbeddings(BaseToolTest):
    """Test Prediction Sum URL Content."""

    tools = prediction_sentence_embeddings.ALLOWED_TOOLS
    models = prediction_sentence_embeddings.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = prediction_sentence_embeddings


class TestOfvMarketResolverTool(BaseToolTest):
    """Test OFV Market Resolver Tool."""

    tools = ofv_market_resolver.ALLOWED_TOOLS
    models = ofv_market_resolver.ALLOWED_MODELS
    prompts = [
        'Please take over the role of a Data Scientist to evaluate the given question. With the given question "Will Apple release iPhone 17 by March 2025?" and the `yes` option represented by `Yes` and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?'
    ]
    tool_module = ofv_market_resolver
