#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2022-2023 Valory AG
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
"""
This script checks that the pipfile of the repository meets the requirements.

In particular:
- Avoid the usage of "*"

It is assumed the script is run from the repository root.
"""
import re
import sys
from pathlib import Path


UNPINNED_PACKAGE_REGEX = r'(?P<package_name>.*)\s?=\s?"\*"'


def check_pipfile(pipfile_path: Path) -> bool:
    """Check a Pipfile"""

    print(f"Checking {pipfile_path.joinpath()}... ", end="")
    with open(pipfile_path, "r", encoding="utf-8") as pipfile:
        lines = pipfile.readlines()
        unpinned = []
        for line in lines:
            m = re.match(UNPINNED_PACKAGE_REGEX, line)
            if m:
                unpinned.append(m.groupdict()["package_name"])
        if unpinned:
            print(
                f"\nThe packages {unpinned} have not been pinned in {pipfile_path.joinpath()}"
            )
            return False
    print("OK")
    return True


import os
from typing import List
try:
    from aea.configurations.data_types import Dependency, PackageType
    from aea.package_manager.base import load_configuration
    from aea.package_manager.v1 import PackageManagerV1
except ImportError as e:
    raise ImportError("open-aea installation not found") from e


import toml

def load_pyproject_toml() -> dict:
    # Path to the pyproject.toml file
    pyproject_toml_path = "./pyproject.toml"

    # Load the pyproject.toml file
    with open(pyproject_toml_path, "r") as toml_file:
        toml_data = toml.load(toml_file)

    # Get the [tool.poetry.dependencies] section
    dependencies = toml_data.get("tool", {}).get("poetry", {}).get("dependencies", {})

    return dependencies


def get_package_dependencies() -> List[str]:
    """Returns a list of package dependencies."""
    package_manager = PackageManagerV1.from_dir(
        Path(os.environ.get("PACKAGES_DIR", str(Path.cwd() / "packages")))
    )
    dependencies: Dict[str, Dependency] = {}
    for package in package_manager.iter_dependency_tree():
        if package.package_type == PackageType.SERVICE:
            continue
        _dependencies = load_configuration(
            package_type=package.package_type,
            package_path=package_manager.package_path_from_package_id(
                package_id=package
            ),
        ).dependencies
        for key, value in _dependencies.items():
            if key not in dependencies:
                dependencies[key] = value
            else:
                if value.version == "":
                    continue
                if dependencies[key].version == "":
                    dependencies[key] = value
                elif value == dependencies[key]:
                    continue
                else:
                    print(f"Non-matching dependency versions for {key}: {value} vs {dependencies[key]}")
                    #raise ValueError(f"Non-matching dependency versions for {key}: {value} vs {dependencies[key]}")

    # return [
    #     " ".join(package.get_pip_install_args()) for package in dependencies.values()
    # ]
    return {package.name: package.version for package in dependencies.values()}


def update_toml(new_package_dependencies: dict) -> None:
    pyproject_toml_path = "./pyproject.toml"

    # Load the pyproject.toml file
    with open(pyproject_toml_path, "r") as toml_file:
        toml_data = toml.load(toml_file)

    toml_data["tool"]["poetry"]["dependencies"] = {key: value if value != "" else "*" for key, value in new_package_dependencies.items()}

    # Write the updated TOML content back to the file
    with open(pyproject_toml_path, "w") as toml_file:
        toml.dump(toml_data, toml_file)

if __name__ == "__main__":
    package_dependencies = get_package_dependencies()
    listed_package_dependencies = load_pyproject_toml()
    listed_package_dependencies.update(package_dependencies)
    update_toml(listed_package_dependencies)
    root_path = Path(".")
    for file_path in root_path.rglob("*Pipfile*"):
        if not check_pipfile(pipfile_path=file_path):
            sys.exit(1)
