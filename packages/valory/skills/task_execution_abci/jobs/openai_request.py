# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
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
import openai

DEFAULT_OPENAI_SETTINGS = dict(
    engine="text-davinci-003",
    prompt = None,
    max_tokens = 500,
    n = 1,
    stop = None,
    temperature=0.7,
)


def run(*args, **kwargs) -> str:

    openai.api_key = kwargs["openai_api_key"]

    request_args = {
        key: kwargs.get(key, DEFAULT_OPENAI_SETTINGS[key])
        for key in DEFAULT_OPENAI_SETTINGS.keys()
    }

    # Call the OpenAI API
    response = openai.Completion.create(**request_args)

    # Extract the result from the API response
    result = response.choices[0].text

    return result