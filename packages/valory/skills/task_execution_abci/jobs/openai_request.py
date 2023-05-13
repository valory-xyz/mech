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
    max_tokens = 500,
    temperature=0.7,
)


def run(*args, **kwargs) -> str:

    openai.api_key = kwargs["openai_api_key"]

    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature =  kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])

    if kwargs["use_gpt4"]:
        messages = [{"role": "user", "content": kwargs["prompt"]}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=kwargs["prompt"],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Extract the result from the API response
    result = response.choices[0].text.strip()

    return result


