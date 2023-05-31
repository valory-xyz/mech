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

"""Contains the job definitions"""

from enum import Enum
from typing import Any, List, Dict, Union

import requests

DEFAULT_STABILITYAI_SETTINGS = {
    "cfg_scale": 7,
    "weight": 0.5,
    "clip_guidance_preset": "FAST_BLUE",
    "height": 512,
    "width": 512,
    "samples": 1,
    "steps": 30,
}
PREFIX = "stabilityai-"
ENGINES = {
    "picture": [
        "stable-diffusion-v1-5",
        "stable-diffusion-xl-beta-v2-2-2",
        "stable-diffusion-512-v2-1",
        "stable-diffusion-768-v2-1",
    ]
}

ALLOWED_TOOLS = [PREFIX + value for value in ENGINES["picture"]]


class FinishReason(Enum):
    """The finish reasons of the API."""

    SUCCESS = 0
    CONTENT_FILTERED = 1
    ERROR = 2


def run(**kwargs: Any) -> Union[Dict, List]:
    """Run the task"""

    api_key = kwargs["api_keys"]["stabilityai"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")
    engine = tool.replace(PREFIX, "")

    # Place content moderation request here if needed

    # default values or kwargs
    cfg_scale = kwargs.get("cfg_scale", DEFAULT_STABILITYAI_SETTINGS["cfg_scale"])
    weight = kwargs.get("weight", DEFAULT_STABILITYAI_SETTINGS["weight"])
    clip_guidance_preset = kwargs.get(
        "clip_guidance_preset", DEFAULT_STABILITYAI_SETTINGS["clip_guidance_preset"]
    )
    height = kwargs.get("height", DEFAULT_STABILITYAI_SETTINGS["height"])
    width = kwargs.get("width", DEFAULT_STABILITYAI_SETTINGS["width"])
    samples = kwargs.get("samples", DEFAULT_STABILITYAI_SETTINGS["samples"])
    steps = kwargs.get("steps", DEFAULT_STABILITYAI_SETTINGS["steps"])

    optional_params = {}
    for optional in ("sampler", "seed", "style_preset", "extras"):
        value = kwargs.get(optional, None)
        if value is not None:
            optional_params[optional] = value

    json_params = {
        "text_prompts": [{"text": prompt, "weight": weight}],
        "cfg_scale": cfg_scale,
        "clip_guidance_preset": clip_guidance_preset,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
    }
    json_params.update(optional_params)

    # request stabilityai api
    response = requests.post(
        f"https://api.stability.ai/v1/generation/{engine}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Stability-Client-ID": "mechs-tool",
        },
        json=json_params,
    )

    if response.status_code != 200:
        raise ValueError(f"Non-200 response ({response.status_code}): {response.text}")

    data = response.json()

    finish_reason = data.get("finishReason", FinishReason.SUCCESS.name)
    if finish_reason == FinishReason.SUCCESS.name:
        # return image data
        return data
    if finish_reason == FinishReason.CONTENT_FILTERED.name:
        raise ValueError(
            f"The result was affected by the content filter and may be blurred: {data}"
        )
    if finish_reason == FinishReason.ERROR.name:
        raise ValueError(f"There was an error while generating the image: {data}")
