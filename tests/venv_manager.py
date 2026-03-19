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

"""Manage isolated virtual environments based on component.yaml dependencies.

Each tool's component.yaml declares the exact dependencies used in production.
This module creates and caches per-tool venvs with those dependencies so that
tests run against the same package versions as production.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Dict, List, Set

import yaml

from tests.shared_constants import PIP_EXECUTABLE, PYTHON_BIN_DIR, PYTHON_EXECUTABLE

logger = logging.getLogger(__name__)

VENVS_DIR_NAME = ".tool_venvs"
VENVS_DIR = Path(__file__).parent.parent / VENVS_DIR_NAME
DEPS_MARKER_FILE = ".deps_installed"
DEPS_YAML_KEY = "dependencies"
VERSION_YAML_KEY = "version"
VERSION_OPERATOR_PREFIXES = ("=", ">", "<", "!", "~")
DEPS_HASH_LENGTH = 12
PIP_INSTALL_TIMEOUT_SECONDS = 300
PIP_BIN_DIR = PYTHON_BIN_DIR  # pip lives in the same bin dir as python

# Tracks all venvs created during this session for teardown cleanup.
_created_venvs: Set[Path] = set()


def parse_component_deps(component_yaml_path: str) -> Dict:
    """Parse dependencies from a component.yaml file."""
    with open(component_yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get(DEPS_YAML_KEY, {})


def deps_to_pip_specs(deps: Dict) -> List[str]:
    """Convert component.yaml dependencies to pip install specifiers.

    Examples:
        {"openai": {"version": "==1.30.2"}} -> ["openai==1.30.2"]
        {"requests": {}}                     -> ["requests"]
        {"pydantic": {"version": ">=1.9.0,<3"}} -> ["pydantic>=1.9.0,<3"]
    """
    specs = []
    for pkg, spec in deps.items():
        if isinstance(spec, dict) and VERSION_YAML_KEY in spec:
            version = spec[VERSION_YAML_KEY]
            if version.startswith(VERSION_OPERATOR_PREFIXES):
                specs.append(f"{pkg}{version}")
            else:
                specs.append(f"{pkg}=={version}")
        else:
            specs.append(pkg)
    return sorted(specs)


def _deps_hash(pip_specs: List[str]) -> str:
    """Compute a short hash of the dependency list for cache keying."""
    content = json.dumps(pip_specs, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:DEPS_HASH_LENGTH]


# Python env vars that can cause IDEs/debuggers to inject into child processes.
# Removed from subprocess env to prevent interference during venv creation.
PYTHON_INJECTION_VARS = ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME")


def _clean_subprocess_env() -> Dict[str, str]:
    """Return a copy of os.environ safe for spawning clean Python subprocesses.

    Strips Python path-injection variables that IDEs, debuggers, and other
    tools use to hook into child processes — these break venv creation and
    pip installs when running under a debugger.
    """
    env = os.environ.copy()
    for var in PYTHON_INJECTION_VARS:
        env.pop(var, None)
    return env


def _create_venv_with_pip(venv_dir: Path, component_name: str) -> None:
    """Create a venv with pip available.

    Creates the venv without pip first (pure directory setup, no subprocess),
    then bootstraps pip using the parent Python's pip with a clean environment
    that won't be affected by IDE/debugger injection.
    """
    venv.create(str(venv_dir), with_pip=False, clear=True)
    python_exe = venv_dir / PIP_BIN_DIR / PYTHON_EXECUTABLE
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python_exe),
            "install",
            "--quiet",
            "pip",
        ],
        env=_clean_subprocess_env(),
        timeout=PIP_INSTALL_TIMEOUT_SECONDS,
    )


def get_or_create_venv(component_yaml_path: str) -> Path:
    """Get or create a venv matching the component.yaml dependencies.

    Venvs are stored under .tool_venvs/<component_name>_<deps_hash>/.
    All created venvs are tracked and cleaned up at session teardown
    via ``cleanup_all_venvs``.

    Returns the path to the venv directory.
    """
    deps = parse_component_deps(component_yaml_path)
    pip_specs = deps_to_pip_specs(deps)

    component_name = Path(component_yaml_path).parent.name
    cache_key = _deps_hash(pip_specs)
    venv_dir = VENVS_DIR / f"{component_name}_{cache_key}"
    marker = venv_dir / DEPS_MARKER_FILE

    if marker.exists():
        logger.info("[%s] Reusing cached venv", component_name)
        _created_venvs.add(venv_dir)
        return venv_dir

    logger.info(
        "[%s] Creating isolated venv with %d dependencies...",
        component_name,
        len(pip_specs),
    )

    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    _create_venv_with_pip(venv_dir, component_name)

    pip_exe = venv_dir / PIP_BIN_DIR / PIP_EXECUTABLE
    if pip_specs:
        logger.info(
            "[%s] Installing: %s", component_name, ", ".join(pip_specs)
        )
        subprocess.check_call(
            [str(pip_exe), "install", "--quiet"] + pip_specs,
            env=_clean_subprocess_env(),
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )

    logger.info("[%s] Venv ready", component_name)
    marker.write_text(json.dumps(pip_specs))
    _created_venvs.add(venv_dir)
    return venv_dir


def cleanup_all_venvs() -> None:
    """Remove all venvs created during this session and the parent directory if empty."""
    for venv_dir in _created_venvs:
        if venv_dir.exists():
            component_name = venv_dir.name.rsplit("_", 1)[0]
            logger.info("[%s] Cleaning up venv", component_name)
            shutil.rmtree(venv_dir)
    _created_venvs.clear()

    if VENVS_DIR.exists() and not any(VENVS_DIR.iterdir()):
        VENVS_DIR.rmdir()
