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

"""Standalone runner for executing tool tests inside isolated virtual environments.

This script is invoked as a subprocess by the test suite. It runs inside a venv
whose dependencies match the tool's component.yaml, ensuring tests exercise
the same dependency versions as production.

Usage:
    <venv>/bin/python tests/isolated_runner.py '<json_config>' <results_file>
"""

import importlib
import importlib.util
import json
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Suppress known harmless warnings from pinned dependency versions:
# - requests: version check against urllib3/charset_normalizer is overly strict
# - anthropic: pydantic v1 compat layer triggers on Python 3.14+
warnings.filterwarnings("ignore", message="urllib3.*or chardet.*doesn't match a supported version")
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

ENV_FILE_NAME = ".env"
ENV_COMMENT_PREFIX = "#"

# Relative path from project root to the utility modules.
# Loaded directly by file path to bypass aea imports in __init__.py.
UTILS_REL_PATH = Path("packages/valory/skills/task_execution/utils")
APIS_MODULE_FILE = "apis.py"
BENCHMARKS_MODULE_FILE = "benchmarks.py"
APIS_MODULE_NAME = "task_execution_apis"
BENCHMARKS_MODULE_NAME = "task_execution_benchmarks"

# Expected response tuple lengths from tool `run()` calls
EXPECTED_RESPONSE_LENGTHS = (5, 6)

# Prediction response fields that must appear in deliver_msg
PREDICTION_FIELDS = ("p_yes", "p_no", "confidence", "info_utility")

# Known error patterns in deliver_msg that indicate missing credentials
GOOGLE_API_KEY_MISSING_MSG = "Google API key not found"
OPENAI_API_KEY_MISSING_PATTERN = "The api_key client option must be set"
UNEXPECTED_ERROR_PATTERN = "Unexpected error:"

# Class names checked by string to avoid cross-venv import issues
TOKEN_COUNTER_CLASS_NAME = "TokenCounterCallback"
KEYCHAIN_CLASS_NAME = "KeyChain"

# Module attribute names for discovering tools and models
ALLOWED_TOOLS_ATTR = "ALLOWED_TOOLS"
ALLOWED_MODELS_ATTR = "ALLOWED_MODELS"

# Config keys
CONFIG_PROJECT_ROOT = "project_root"
CONFIG_MODULE_PATH = "module_path"
CONFIG_CALLABLE = "callable"
CONFIG_PROMPTS = "prompts"
CONFIG_VALIDATE_PREDICTION = "validate_prediction"
CONFIG_REQUIRED_ENV_VARS = "required_env_vars"
CONFIG_SERVICE_TO_ENV_VAR = "service_to_env_var"

# Truncation limits
PROMPT_TRUNCATE_LENGTH = 100
DELIVER_MSG_TRUNCATE_LENGTH = 1000
ERROR_MSG_TRUNCATE_LENGTH = 200


def _load_env_file(project_root: str) -> None:
    """Load a .env file into os.environ without requiring python-dotenv.

    Only sets variables not already present, so CI env vars take precedence.
    Silently skips if the file doesn't exist.
    """
    env_path = Path(project_root) / ENV_FILE_NAME
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        _apply_env_line(line.strip())


def _apply_env_line(line: str) -> None:
    """Parse and apply a single .env line to os.environ."""
    if not line or line.startswith(ENV_COMMENT_PREFIX) or "=" not in line:
        return
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip().strip("\"'")
    if not key or key in os.environ:
        return
    os.environ[key] = value


def _load_module_from_file(name: str, file_path: str) -> Any:
    """Load a Python module directly from a file path, bypassing __init__.py chains."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_utils(project_root: str) -> Tuple[Any, Any]:
    """Load KeyChain and TokenCounterCallback directly from their .py files."""
    utils_dir = Path(project_root) / UTILS_REL_PATH
    apis_mod = _load_module_from_file(
        APIS_MODULE_NAME, str(utils_dir / APIS_MODULE_FILE)
    )
    benchmarks_mod = _load_module_from_file(
        BENCHMARKS_MODULE_NAME, str(utils_dir / BENCHMARKS_MODULE_FILE)
    )
    return apis_mod.KeyChain, benchmarks_mod.TokenCounterCallback


def _collect_keys(env_var: str) -> List[str]:
    """Collect API keys. Plural env var (JSON array) takes precedence over singular."""
    plural = os.environ.get(f"{env_var}S", "")
    if plural:
        try:
            keys = json.loads(plural)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{env_var}S must be a JSON array (e.g. '[\"key1\",\"key2\"]'), got: {plural[:50]}"
            ) from e
        return [k for k in keys if k]
    single = os.environ.get(env_var, "")
    return [single] if single else [""]


def build_keychain(keychain_cls: Any, service_to_env_var: Dict[str, str]) -> Any:
    """Build a KeyChain from environment variables."""
    return keychain_cls(
        {
            service: _collect_keys(env_var)
            for service, env_var in service_to_env_var.items()
        }
    )


def _validate_deliver_msg(deliver_msg: str, validate_prediction: bool) -> List[str]:
    """Validate the deliver_msg (response[0]) content."""
    errors: List[str] = []
    if deliver_msg == GOOGLE_API_KEY_MISSING_MSG:
        errors.append("Google API key is required to run the test.")
    if OPENAI_API_KEY_MISSING_PATTERN in deliver_msg:
        errors.append("OpenAI API key is required to run the test.")
    if UNEXPECTED_ERROR_PATTERN in deliver_msg:
        errors.append(f"Unexpected error in delivered message: {deliver_msg[:ERROR_MSG_TRUNCATE_LENGTH]}")
    if not validate_prediction:
        return errors
    for field in PREDICTION_FIELDS:
        if field not in deliver_msg:
            errors.append(f"Missing '{field}' in delivered message.")
    return errors


def _validate_response_types(response: tuple) -> List[str]:
    """Validate the types of response elements.

    5-tuple (no key rotation): (result, prompt, tx_data, callback, used_params)
    6-tuple (key rotation):    (result, prompt, tx_data, callback, used_params, api_keys)
    """
    errors: List[str] = []
    if not isinstance(response[1], str):
        errors.append("Response[1] must be a string.")
    if not (isinstance(response[2], dict) or response[2] is None):
        errors.append("Response[2] must be a dictionary or None.")
    if response[3] is not None and type(response[3]).__name__ != TOKEN_COUNTER_CLASS_NAME:
        errors.append(f"Response[3] must be a {TOKEN_COUNTER_CLASS_NAME} or None.")
    if not (isinstance(response[4], dict) or response[4] is None):
        errors.append("Response[4] must be a dict (used_params) or None.")
    if len(response) == 6 and type(response[5]).__name__ != KEYCHAIN_CLASS_NAME:
        errors.append(f"Response[5] must be a {KEYCHAIN_CLASS_NAME} object.")
    return errors


def validate_response(response: Any, validate_prediction: bool = True) -> List[str]:
    """Validate a tool response. Returns a list of error strings (empty = pass)."""
    if not isinstance(response, tuple):
        return ["Response of the tool must be a tuple."]
    if len(response) not in EXPECTED_RESPONSE_LENGTHS:
        return [f"Response must have {EXPECTED_RESPONSE_LENGTHS} elements, got {len(response)}."]

    errors: List[str] = []
    deliver_msg = response[0]
    if not isinstance(deliver_msg, str):
        errors.append("Response[0] must be a string.")
    else:
        errors.extend(_validate_deliver_msg(deliver_msg, validate_prediction))
    errors.extend(_validate_response_types(response))
    return errors


def _check_required_env_vars(required_env_vars: List[str]) -> List[str]:
    """Return list of missing env var names (empty if all present)."""
    return [
        var for var in required_env_vars
        if not os.environ.get(var) and not os.environ.get(f"{var}S")
    ]


def _make_error_result(errors: List[str]) -> Dict[str, Any]:
    """Build a single failure result dict."""
    return {
        "model": None,
        "tool": None,
        "prompt": None,
        "success": False,
        "errors": errors,
    }


def _run_single_invocation(
    func: Any,
    prompt: str,
    tool: str,
    keys: Any,
    token_counter_cls: Any,
    model: Any,
    validate_prediction: bool,
) -> Dict[str, Any]:
    """Run a single tool invocation and return the result dict."""
    result: Dict[str, Any] = {
        "model": model,
        "tool": tool,
        "prompt": prompt[:PROMPT_TRUNCATE_LENGTH],
        "success": False,
        "errors": [],
    }
    try:
        response = func(
            prompt=prompt,
            tool=tool,
            api_keys=keys,
            counter_callback=token_counter_cls(),
            model=model,
        )
        errs = validate_response(response, validate_prediction)
        result["success"] = len(errs) == 0
        result["errors"] = errs
        if isinstance(response, tuple) and len(response) > 0:
            result["deliver_msg"] = str(response[0])[:DELIVER_MSG_TRUNCATE_LENGTH]
    except Exception as e:
        result["errors"] = [f"{type(e).__name__}: {e}"]
    return result


def _missing_env_vars_result(missing_vars: List[str]) -> Dict[str, Any]:
    """Build the output dict for missing env vars."""
    return {
        "results": [
            _make_error_result([
                f"Missing required environment variables: {', '.join(sorted(missing_vars))}. "
                "Either export them in your shell or set them in a .env file "
                "in the project root."
            ])
        ]
    }


def run_tests(config: Dict[str, Any]) -> Dict[str, Any]:
    """Import the tool module, run all tool/model/prompt combinations, validate."""
    project_root = config[CONFIG_PROJECT_ROOT]
    _load_env_file(project_root)

    missing_vars = _check_required_env_vars(config.get(CONFIG_REQUIRED_ENV_VARS, []))
    if missing_vars:
        return _missing_env_vars_result(missing_vars)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        return _execute_all_combinations(config, project_root)
    except Exception as e:
        return {
            "results": [
                _make_error_result([
                    f"Setup error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                ])
            ]
        }


def _execute_all_combinations(config: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    """Run all model x tool x prompt combinations and return results."""
    module = importlib.import_module(config[CONFIG_MODULE_PATH])
    KeyChain, TokenCounterCallback = _load_utils(project_root)

    keys = build_keychain(KeyChain, config[CONFIG_SERVICE_TO_ENV_VAR])
    tools = getattr(module, ALLOWED_TOOLS_ATTR, [])
    models = getattr(module, ALLOWED_MODELS_ATTR, [None])
    func = getattr(module, config[CONFIG_CALLABLE])
    prompts = config[CONFIG_PROMPTS]
    validate_prediction = config.get(CONFIG_VALIDATE_PREDICTION, True)

    results = [
        _run_single_invocation(func, prompt, tool, keys, TokenCounterCallback, model, validate_prediction)
        for model in models
        for tool in tools
        for prompt in prompts
    ]
    return {"results": results}


def main() -> None:
    """Entry point — read config from argv, write JSON results to file."""
    config = json.loads(sys.argv[1])
    results_file = sys.argv[2]
    output = run_tests(config)
    Path(results_file).write_text(json.dumps(output))
    all_passed = all(r["success"] for r in output["results"])
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
