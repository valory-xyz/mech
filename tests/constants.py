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
"""This module contains constants."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_SECRET_KEY = os.environ("OPENAI_SECRET_KEY")
STABILITY_API_KEY = os.environ("STABILITY_API_KEY")
GOOGLE_API_KEY = os.environ("GOOGLE_API_KEY")
GOOGLE_ENGINE_ID = os.environ("GOOGLE_ENGINE_ID")
CLAUDE_API_KEY = os.environ("CLAUDE_API_KEY")
REPLICATE_API_KEY = os.environ("REPLICATE_API_KEY")
NEWS_API_KEY = os.environ("NEWS_API_KEY")
