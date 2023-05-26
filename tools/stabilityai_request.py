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
import base64
import os
import requests

api_host = os.getenv('API_HOST', 'https://api.stability.ai')

# for testing (remove later)
STABILITYAI_API_KEY = os.getenv("STABILITYAI_API_KEY")
if STABILITYAI_API_KEY is None:
    raise Exception("Missing Stability API key.")


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
    "picture": ["stable-diffusion-v1-5", "stable-diffusion-xl-beta-v2-2-2", "stable-diffusion-512-v2-1", "stable-diffusion-768-v2-1"]
}

ALLOWED_TOOLS = [PREFIX + value for value in values for values in ENGINES.values()]


def run(**kwargs) -> str:
    """Run the task"""

    api_key = kwargs["api_keys"]["stabilityai"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")
    engine = tool.removeprefix(PREFIX)

    # Moderation?

    # default values or kwargs
    cfg_scale = kwargs.get("cfg_scale", DEFAULT_STABILITYAI_SETTINGS["cfg_scale"])
    weight = kwargs.get("weight", DEFAULT_STABILITYAI_SETTINGS["weight"])
    clip_guidance_preset = kwargs.get("clip_guidance_preset", DEFAULT_STABILITYAI_SETTINGS["clip_guidance_preset"])
    height = kwargs.get("height", DEFAULT_STABILITYAI_SETTINGS["height"])
    width = kwargs.get("width", DEFAULT_STABILITYAI_SETTINGS["width"])
    samples = kwargs.get("samples", DEFAULT_STABILITYAI_SETTINGS["samples"])
    steps = kwargs.get("steps", DEFAULT_STABILITYAI_SETTINGS["steps"])

    response = requests.post(
    f"{api_host}/v1/generation/{engine}/text-to-image",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "text_prompts": [
            {
                "text": prompt,
                "weight": weight
            }
        ],
        "cfg_scale": cfg_scale,
        "clip_guidance_preset": clip_guidance_preset,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
    },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()

    # decode into png file
    #for i, image in enumerate(data["artifacts"]):
    #   with open(f"./out/v1_txt2img_{i}.png", "wb") as f:
    #        f.write(base64.b64decode(image["base64"]))

    return data




def main(task: str):
    """Run the task"""

    # build dictionary of kwargs
    kwargs = {
        "prompt": task,
        "tool": "stabilityai-stable-diffusion-v1-5",
        "max_tokens": 500,
        "temperature": .7,
        "top_p": 1,
        "api_keys": {"openai": STABILITYAI_API_KEY},
    }

    txs_list = native_transfer(prompt=kwargs["prompt"]), 
    print("RETURNED DICT LIST : " + str(txs_list))


if __name__ == "__main__":
    task = sys.argv
    try:
        main(task)
    except KeyboardInterrupt:
        pass