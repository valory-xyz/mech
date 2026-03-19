# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
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

"""Shared constants used by both the test suite (conftest) and the isolated runner.

This module must have NO external dependencies — it is imported inside
isolated venvs that only contain the tool's component.yaml dependencies.
"""

# KeyChain service name -> environment variable mapping.
# Used to build the KeyChain from env vars and to determine
# which env vars a tool requires based on its dependencies.
SERVICE_TO_ENV_VAR = {
    "openai": "OPENAI_SECRET_KEY",
    "stabilityai": "STABILITY_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "google_engine_id": "GOOGLE_ENGINE_ID",
    "anthropic": "CLAUDE_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "newsapi": "NEWS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gnosis_rpc_url": "GNOSIS_RPC_URL",
    "gemini": "GEMINI_API_KEY",
    "serperapi": "SERPER_API_KEY",
}

# Result JSON keys — shared between conftest, isolated_runner, and test_tools
RESULT_KEY_SUCCESS = "success"
RESULT_KEY_MODEL = "model"
RESULT_KEY_TOOL = "tool"
RESULT_KEY_DELIVER_MSG = "deliver_msg"
RESULT_KEY_RESULTS = "results"
RESULT_KEY_ERRORS = "errors"
RESULT_KEY_PROMPT = "prompt"

# Default callable name for tool run() functions
DEFAULT_CALLABLE = "run"

# Truncation limit for deliver_msg previews in logs and failure output
DELIVER_MSG_PREVIEW_LENGTH = 200

# Platform-specific binary paths
PYTHON_BIN_DIR = "bin"
PYTHON_EXECUTABLE = "python"
PIP_EXECUTABLE = "pip"
