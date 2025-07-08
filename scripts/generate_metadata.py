import os
import yaml
import json
import importlib.util
from types import ModuleType
from pathlib import Path
from typing import List, Dict, Any


ROOT_DIR = "./packages"
CUSTOMS = "customs"
METADATA_FILE_PATH = "metadata.json"
INIT_PY = "__init__.py"
COMPONENT_YAML = "component.yaml"
ALLOWED_TOOLS = "ALLOWED_TOOLS"
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


def find_folders_with_name() -> List[str]:
    matched_folders = []
    for dirpath, dirnames, _ in os.walk(ROOT_DIR):
        for dirname in dirnames:
            if CUSTOMS in dirname:
                matched_folders.append(os.path.join(dirpath, dirname))
    return matched_folders


def get_immediate_subfolders(folder_path) -> List[str]:
    subfolders = []
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isdir(full_path):
            subfolders.append(full_path)
    return subfolders


def read_files_in_folder(folder_path) -> Dict[str, str]:
    contents = {}
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    contents[filename] = f.read()
    except Exception as e:
        print(f"Error reading files in {folder_path}: {e}")
    return contents


def import_module_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_tools_data() -> List[Dict[str, Any]]:
    tools_data: List[Dict[str, Any]] = []
    matches = find_folders_with_name()
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
                    file = Path(sub) / fname
                    try:
                        mod = import_module_from_path(fname, file)
                        keys = ["ALLOWED_TOOLS", "AVAILABLE_TOOLS"]
                        for k in keys:
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


def build_tools_metadata(tools_data: List[Dict[str, Any]]):
    result = METADATA_TEMPLATE.copy()

    for entry in tools_data:
        author = entry.get("author", "")
        tool_name = entry.get("tool_name", "")
        allowed_tools = entry.get("allowed_tools", [])
        if not allowed_tools:
            print(
                f"Warning: '{tool_name}' by '{author}' has no allowed tools/invalid format!"
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
    tools_data = generate_tools_data()
    metadata = build_tools_metadata(tools_data)

    # Dump the result to the JSON file
    with open(METADATA_FILE_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata has been stored to {METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
