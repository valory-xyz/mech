# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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
"""The script allows the user to generate the metadata of the tools"""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import yaml


ROOT_DIR = "./packages"
CUSTOMS = "customs"
METADATA_FILE_PATH = "metadata.json"
INIT_PY = "__init__.py"
COMPONENT_YAML = "component.yaml"
ALLOWED_TOOLS = "ALLOWED_TOOLS"
AVAILABLE_TOOLS = "AVAILABLE_TOOLS"
TOOLS_IDENTIFIERS = frozenset([ALLOWED_TOOLS, AVAILABLE_TOOLS])
METADATA_TEMPLATE = {
    "name": "Autonolas Mech III",
    "description": "The mech executes AI tasks requested on-chain and delivers the results to the requester.",
    "inputFormat": "ipfs-v0.1",
    "outputFormat": "ipfs-v0.1",
    "image": "tbd",
    "tools": [],
    "toolMetadata": {},
}
INPUT_SCHEMA = {
    "type": "text",
    "description": "The text to make a prediction on",
}
OUTPUT_SCHEMA = {
    "type": "object",
    "description": "A JSON object containing the prediction and confidence",
    "schema": {
        "type": "object",
        "properties": {
            "requestId": {
                "type": "integer",
                "description": "Unique identifier for the request",
            },
            "result": {
                "type": "string",
                "description": "Result information in JSON format as a string",
                "example": '{\n  "p_yes": 0.6,\n  "p_no": 0.4,\n  "confidence": 0.8,\n  "info_utility": 0.6\n}',
            },
            "prompt": {
                "type": "string",
                "description": "The prompt used to make the prediction.",
            },
        },
        "required": ["requestId", "result", "prompt"],
    },
}


def find_customs_folders() -> List[Path]:
    """Finds all the customs folders inside the packages dir"""
    return [p for p in Path(ROOT_DIR).rglob("*") if p.is_dir() and CUSTOMS in p.name]


def get_immediate_subfolders(folder_path: Path) -> List[Path]:
    """Finds all the subfolders inside the dir"""
    folder = Path(folder_path)
    return [item for item in folder.iterdir() if item.is_dir()]


def read_files_in_folder(folder_path: Path) -> Dict[str, str]:
    """Reads contents of all the files inside the dir"""
    contents = {}
    folder = Path(folder_path)
    try:
        for file_path in folder.iterdir():
            if file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    contents[file_path.name] = f.read()
    except Exception as e:
        print(f"Error reading files in {folder_path}: {e}")
    return contents


def import_module_from_path(module_name: str, file_path: str) -> ModuleType:
    """Imports the py file as a module"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        print(f"Cannot load module '{module_name!r}' from '{file_path!r}'")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_tools_data() -> List[Dict[str, Any]]:
    """Generates the tools data needed for the metadata.json"""
    tools_data: List[Dict[str, Any]] = []
    matches = find_customs_folders()
    for folder in matches:
        print(f"\n Matched folder: {folder}")
        subfolders = get_immediate_subfolders(folder)
        for sub in subfolders:
            print(f"  └── Subfolder: {sub}")
            files = read_files_in_folder(sub)
            tool_entry: Dict[str, Any] = {}
            for fname, content in files.items():
                if fname == INIT_PY:
                    continue
                if fname == COMPONENT_YAML:
                    try:
                        data = yaml.safe_load(content)
                        tool_entry["author"] = data.get("author")
                        tool_entry["tool_name"] = data.get("name")
                        tool_entry["description"] = data.get("description")
                    except Exception as e:
                        print(f"Failed to parse YAML in {sub}: {e}")
                        continue
                else:
                    file = str(Path(sub) / fname)
                    try:
                        mod = import_module_from_path(fname, file)
                        for k in TOOLS_IDENTIFIERS:
                            tools = getattr(mod, k, None)
                            if isinstance(tools, list):
                                tool_entry["allowed_tools"] = tools
                                break
                    except Exception as e:
                        print(f"Failed to parse PY from {file}: {e}")
                        continue

            if tool_entry:
                tools_data.append(tool_entry)

    return tools_data


def build_tools_metadata(tools_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds the metadata.json from the tools data"""
    result: dict[str, Any] = METADATA_TEMPLATE

    for entry in tools_data:
        author = entry.get("author", "")
        tool_name = entry.get("tool_name", "")
        allowed_tools = entry.get("allowed_tools", [])
        if not allowed_tools:
            print(
                f"Warning: '{tool_name!r}' by '{author!r}' has no allowed tools/invalid format!"
            )

        for tool in entry.get("allowed_tools", []):
            if tool not in result["tools"]:
                result["tools"].append(tool)

            result["toolMetadata"][tool] = {
                "name": entry.get("tool_name", ""),
                "description": entry.get("description", ""),
                "input": INPUT_SCHEMA,
                "output": OUTPUT_SCHEMA,
            }

    return result


def main() -> None:
    """Run the generate_metadata script."""
    tools_data = generate_tools_data()
    metadata = build_tools_metadata(tools_data)

    # Dump the result to the JSON file
    with open(METADATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata has been stored to {METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
