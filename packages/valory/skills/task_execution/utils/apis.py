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
"""Utils for API integrations."""
from typing import Dict, List


class KeyChain:
    """Class for managing API keys."""

    def __init__(self, services: Dict[str, List[str]]) -> None:
        """Initialize the KeyChain with a dictionary of service names and corresponding lists of API keys."""
        if not isinstance(services, dict):
            raise ValueError(
                "Services must be a dictionary with service names as keys and lists of API keys as values.")

        self.services = services
        self.current_index = {service: 0 for service in services}  # Start with the first key for each service

    def rotate(self, service_name: str) -> None:
        """Rotate the current API key for a given service to the next one."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name}' not found in KeyChain.")

        # Increment the current index, looping back if at the end of the list
        self.current_index[service_name] += 1
        if self.current_index[service_name] >= len(self.services[service_name]):
            self.current_index[service_name] = 0  # Reset to the start

    def __getitem__(self, service_name: str) -> str:
        """Get the current API key for a given service."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name}' not found in KeyChain.")

        index = self.current_index[service_name]
        return self.services[service_name][index]
