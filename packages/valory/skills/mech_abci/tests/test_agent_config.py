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
"""Tests that the mech agent's skill overrides match what the skills declare."""

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest
import yaml

PACKAGES_DIR = Path(__file__).parents[4]
AGENT_CONFIG = PACKAGES_DIR / "valory" / "agents" / "mech" / "aea-config.yaml"


def _dev_skill_names() -> Set[str]:
    """Return the names of the skills this repository develops."""
    dev = json.loads((PACKAGES_DIR / "packages.json").read_text())["dev"]
    return {key.split("/")[2] for key in dev if key.startswith("skill/valory/")}


def _skill_overrides() -> List[Tuple[str, Dict[str, Any]]]:
    """Return ``(skill name, override)`` for each dev skill the agent overrides."""
    documents = list(yaml.safe_load_all(AGENT_CONFIG.read_text()))
    dev_skills = _dev_skill_names()
    return [
        (doc["public_id"].split(":")[0].split("/")[1], doc)
        for doc in documents[1:]
        if doc.get("type") == "skill"
        and doc["public_id"].split(":")[0].split("/")[1] in dev_skills
    ]


@pytest.mark.parametrize(
    "skill_name,override", _skill_overrides(), ids=lambda value: str(value)[:40]
)
def test_overridden_model_args_are_declared_by_the_skill(
    skill_name: str, override: Dict[str, Any]
) -> None:
    """Every model arg the agent overrides exists in that skill's ``skill.yaml``.

    The AEA refuses to start when an override names an arg the skill does not
    declare, so an undeclared override stops the agent at load time.

    :param skill_name: the overridden skill's name.
    :param override: the agent's override document for that skill.
    """
    skill = yaml.safe_load(
        (PACKAGES_DIR / "valory" / "skills" / skill_name / "skill.yaml").read_text()
    )
    undeclared = [
        f"{model}.{arg}"
        for model, model_override in (override.get("models") or {}).items()
        for arg in (model_override.get("args") or {})
        if arg not in (skill["models"].get(model) or {}).get("args", {})
    ]
    assert undeclared == []


def test_the_agent_overrides_dev_skills() -> None:
    """The override check above has dev skills to look at."""
    assert {name for name, _ in _skill_overrides()} >= {
        "mech_abci",
        "task_execution",
    }
