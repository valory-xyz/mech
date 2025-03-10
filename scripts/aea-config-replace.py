#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2021-2025 Valory AG
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


"""Updates fetched agent with correct config"""
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


AGENT_NAME = "agent"

PATH_TO_VAR = {
    # ledger
    "config/ledger_apis/ethereum/address": "ETHEREUM_LEDGER_RPC_0",
    "config/ledger_apis/gnosis/address": "GNOSIS_RPC_0",
    # Params
    # mech abci
    "models/params/args/on_chain_service_id": "ON_CHAIN_SERVICE_ID",
    "models/params/args/mech_marketplace_address": "MECH_MARKETPLACE_ADDRESS",
    "models/params/args/agent_registry_address": "AGENT_REGISTRY_ADDRESS",
    "models/params/args/hash_checkpoint_address": "CHECKPOINT_ADDRESS",
    "models/params/args/setup/all_participants": "ALL_PARTICIPANTS",
    "models/params/args/setup/safe_contract_address": "SAFE_CONTRACT_ADDRESS",
    "models/params/args/mech_to_subscription": "MECH_TO_SUBSCRIPTION",
    # task_execution
    "models/params/args/tools_to_package_hash": "TOOLS_TO_PACKAGE_HASH",
    "models/params/args/api_keys": "API_KEYS",
    "models/params/args/num_agents": "NUM_AGENTS",
    "models/params/args/mech_to_config": "MECH_TO_CONFIG",
}

CONFIG_REGEX = r"\${.*?:(.*)}"


def find_and_replace(config, path, new_value):
    """Find and replace a variable"""

    # Find the correct section where this variable fits
    section_indexes = []
    for i, section in enumerate(config):
        value = section
        try:
            for part in path:
                value = value[part]
            section_indexes.append(i)
        except KeyError:
            continue

    if not section_indexes:
        raise ValueError(f"Could not update {path}")

    # To persist the changes in the config variable,
    # access iterating the path parts but the last part
    for section_index in section_indexes:
        sub_dic = config[section_index]
        for part in path[:-1]:
            sub_dic = sub_dic[part]

        # Now, get the whole string value
        old_str_value = sub_dic[path[-1]]

        # Extract the old variable value
        match = re.match(CONFIG_REGEX, old_str_value)
        old_var_value = match.groups()[0]

        # Replace the old variable with the secret value in the complete string
        new_str_value = old_str_value.replace(old_var_value, new_value)
        sub_dic[path[-1]] = new_str_value

    return config


def main() -> None:
    """Main"""
    load_dotenv(dotenv_path=".1env")

    # Load the aea config
    with open(Path(AGENT_NAME, "aea-config.yaml"), "r", encoding="utf-8") as file:
        config = list(yaml.safe_load_all(file))

    # Search and replace all the secrets
    for path, var in PATH_TO_VAR.items():
        try:
            new_value = os.getenv(var)
            if new_value is None:
                print(f"Env var {var} is not set")
                continue
            config = find_and_replace(config, path.split("/"), new_value)
        except Exception as e:
            raise ValueError(f"Could not update {path}") from e

    # Dump the updated config
    with open(Path(AGENT_NAME, "aea-config.yaml"), "w", encoding="utf-8") as file:
        yaml.dump_all(config, file, sort_keys=False)


if __name__ == "__main__":
    main()
