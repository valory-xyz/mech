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

"""Test configuration and fixtures for isolated-venv tool testing."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.shared_constants import (
    DEFAULT_CALLABLE,
    DELIVER_MSG_PREVIEW_LENGTH,
    PYTHON_BIN_DIR,
    PYTHON_EXECUTABLE,
    RESULT_KEY_DELIVER_MSG,
    RESULT_KEY_ERRORS,
    RESULT_KEY_MODEL,
    RESULT_KEY_PROMPT,
    RESULT_KEY_RESULTS,
    RESULT_KEY_SUCCESS,
    RESULT_KEY_TOOL,
    SERVICE_TO_ENV_VAR,
)
from tests.venv_manager import cleanup_all_venvs, get_or_create_venv, parse_component_deps

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RUNNER_SCRIPT = PROJECT_ROOT / "tests" / "isolated_runner.py"
RUNNER_TIMEOUT_SECONDS = 600
OUTPUT_TRUNCATE_LENGTH = 1000
RESULTS_FILE_SUFFIX = ".json"

# Status labels for per-result logging
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

# Maps pip package names (from component.yaml) to the KeyChain service names they require at runtime.
DEPENDENCY_TO_SERVICES = {
    "openai": ["openai", "openrouter"],
    "anthropic": ["anthropic"],
    "google-api-python-client": ["google_api_key", "google_engine_id"],
    "googlesearch-python": ["google_api_key", "google_engine_id"],
    "google-generativeai": ["gemini"],
    "replicate": ["replicate"],
}


def _required_env_vars_for_component(component_yaml: str) -> List[str]:
    """Determine which env vars a tool needs based on its component.yaml deps."""
    deps = parse_component_deps(component_yaml)
    services: List[str] = []
    for pkg in deps:
        services.extend(DEPENDENCY_TO_SERVICES.get(pkg, []))
    return list(dict.fromkeys(
        SERVICE_TO_ENV_VAR[s]
        for s in services
        if s in SERVICE_TO_ENV_VAR
    ))


def _check_env_vars(required_env_vars: List[str]) -> None:
    """Fail fast if required API key env vars are not set."""
    missing = [var for var in required_env_vars if not os.environ.get(var)]
    if not missing:
        return
    pytest.fail(
        f"Missing required environment variables: {', '.join(sorted(missing))}. "
        "Either export them in your shell or set them in a .env file in the project root."
    )


@pytest.fixture(autouse=True, scope="session")
def _cleanup_tool_venvs():
    """Session fixture that cleans up all isolated tool venvs after tests finish."""
    yield
    cleanup_all_venvs()


def _build_runner_config(
    module_path: str,
    callable_name: str,
    prompts: List[str],
    validate_prediction: bool,
    required_env_vars: List[str],
) -> Dict[str, Any]:
    """Build the JSON config dict passed to the isolated runner subprocess."""
    return {
        "project_root": str(PROJECT_ROOT),
        "module_path": module_path,
        "callable": callable_name,
        "prompts": prompts,
        "validate_prediction": validate_prediction,
        "required_env_vars": required_env_vars,
        "service_to_env_var": SERVICE_TO_ENV_VAR,
    }


def _execute_runner(
    python_exe: Path,
    config: Dict[str, Any],
    results_file: str,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Execute the isolated runner subprocess."""
    return subprocess.run(
        [
            str(python_exe),
            str(RUNNER_SCRIPT),
            json.dumps(config),
            results_file,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


def _log_subprocess_output(
    component_name: str, result: subprocess.CompletedProcess
) -> None:
    """Log stdout/stderr from the runner subprocess at DEBUG level."""
    if result.stdout.strip():
        logger.debug("[%s] Tool stdout:\n%s", component_name, result.stdout.strip())
    if result.stderr.strip():
        logger.debug("[%s] Tool stderr:\n%s", component_name, result.stderr.strip())


def _log_results(component_name: str, all_results: List[Dict[str, Any]]) -> None:
    """Log each result's status and deliver_msg preview."""
    for r in all_results:
        status = STATUS_PASS if r[RESULT_KEY_SUCCESS] else STATUS_FAIL
        preview = r.get(RESULT_KEY_DELIVER_MSG, "")[:DELIVER_MSG_PREVIEW_LENGTH]
        logger.info(
            "[%s] %s model=%s tool=%s\n    -> %s",
            component_name,
            status,
            r[RESULT_KEY_MODEL],
            r[RESULT_KEY_TOOL],
            preview,
        )
    passed = sum(r[RESULT_KEY_SUCCESS] for r in all_results)
    logger.info("[%s] Completed: %d/%d combinations passed", component_name, passed, len(all_results))


def _make_error_result(returncode: int, stdout: str, stderr: str) -> Dict[str, Any]:
    """Build a failure result dict when the runner subprocess produces no output file."""
    return {
        RESULT_KEY_RESULTS: [
            {
                RESULT_KEY_MODEL: None,
                RESULT_KEY_TOOL: None,
                RESULT_KEY_PROMPT: None,
                RESULT_KEY_SUCCESS: False,
                RESULT_KEY_ERRORS: [
                    f"Runner failed with exit code {returncode}",
                    f"stdout: {stdout[:OUTPUT_TRUNCATE_LENGTH]}",
                    f"stderr: {stderr[:OUTPUT_TRUNCATE_LENGTH]}",
                ],
            }
        ]
    }


def _parse_results_file(results_file: str) -> Dict[str, Any]:
    """Parse the JSON results file written by the runner, or return None if unavailable."""
    results_path = Path(results_file)
    if not results_path.exists() or results_path.stat().st_size == 0:
        return {}
    return json.loads(results_path.read_text())


def run_tool_in_isolated_venv(
    component_yaml: str,
    module_path: str,
    prompts: List[str],
    callable_name: str = DEFAULT_CALLABLE,
    validate_prediction: bool = True,
    timeout: int = RUNNER_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run a tool test inside an isolated venv matching its component.yaml deps.

    Returns the parsed JSON results from the runner subprocess.
    """
    component_name = Path(component_yaml).parent.name

    required_env_vars = _required_env_vars_for_component(component_yaml)
    logger.info("[%s] Required API keys: %s", component_name, ", ".join(required_env_vars) or "(none)")
    _check_env_vars(required_env_vars)

    logger.info("[%s] Preparing isolated environment...", component_name)
    venv_dir = get_or_create_venv(component_yaml)
    python_exe = venv_dir / PYTHON_BIN_DIR / PYTHON_EXECUTABLE

    config = _build_runner_config(module_path, callable_name, prompts, validate_prediction, required_env_vars)

    with tempfile.NamedTemporaryFile(mode="w", suffix=RESULTS_FILE_SUFFIX, delete=False) as tmp:
        results_file = tmp.name

    try:
        logger.info("[%s] Running tool in isolated subprocess...", component_name)
        result = _execute_runner(python_exe, config, results_file, timeout)
        _log_subprocess_output(component_name, result)

        output = _parse_results_file(results_file)
        if not output:
            return _make_error_result(result.returncode, result.stdout, result.stderr)

        _log_results(component_name, output[RESULT_KEY_RESULTS])
        return output
    finally:
        Path(results_file).unlink(missing_ok=True)
