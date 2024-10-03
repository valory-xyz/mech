import functools
import requests
import json
from typing import Optional, Dict, Any, Tuple, Callable

# Custom exception class
class CorcelAPIException(Exception):
    pass

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]

def error_response(msg: str) -> MechResponse:
    """Return an error mech response."""
    return msg, None, None, None

def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except CorcelAPIException:
                # try with a new key again
                service = "corcel"
                if retries_left[service] <= 0:
                    raise Exception("Error: API retries exhausted")
                retries_left[service] -= 1
                api_keys.rotate(service)
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper

CORCEL_URL = "https://api.corcel.io/v1/image/vision/text-to-image"

AVAILABLE_CFG_SCALE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
AVAILABLE_HEIGHT = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344]
AVAILABLE_WIDTH = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344]
ALLOWED_TOOLS = ["stable-diffusion-xl-turbo", "proteus", "dreamshaper", "playground", "flux-schnell"]
ALLOWED_MODELS = ["text-to-image"]
AVAILABLE_WEIGHT = [-1, 0, 1]

DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS = {
    "cfg_scale": 2,
    "height": 1024,
    "width": 1024,
    "steps": 8,
    "weight": 0,
}

@with_key_rotation
def run(**kwargs) -> MechResponse:
    """Run the task"""

    api_key = kwargs["api_keys"]["corcel"]
    tool = kwargs["tool"]
    prompt = kwargs["prompt"]

    if api_key is None:
        return error_response("Corcel API key is not found.")

    if tool is None:
        return error_response("No tool has been specified.")

    if prompt is None:
        return error_response("No prompt has been specified.")

    if tool not in ALLOWED_TOOLS:
        return error_response(f"Engine {tool} is not in the list of supported engines.")

    cfg_scale = kwargs.get("cfg_scale", DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS["cfg_scale"])
    height = kwargs.get("height", DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS["height"])
    width = kwargs.get("width", DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS["width"])
    steps = kwargs.get("steps", DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS["steps"])
    weight = kwargs.get("weight", DEFAULT_CORCEL_TEXT_TO_IMAGE_SETTINGS["weight"])

    if cfg_scale not in AVAILABLE_CFG_SCALE:
        return error_response(f"Cfg scale {cfg_scale} is not in the list of supported cfg scales.")
    if height not in AVAILABLE_HEIGHT:
        return error_response(f"Height {height} is not in the list of supported heights.")
    if width not in AVAILABLE_WIDTH:
        return error_response(f"Width {width} is not in the list of supported widths.")
    if weight not in AVAILABLE_WEIGHT:
        return error_response(f"Weight {weight} is not in the list of supported weights.")
    if tool == "playground" and (steps < 25 or steps > 51):
        return error_response("Playground engine only accepts steps between 25-51.")
    if tool in ["proteus", "stable-diffusion-xl-turbo", "dreamshaper", "flux-schnell"] and (steps < 6 or steps > 12):
        return error_response("Proteus/Dreamshaper/Stable Diffusion/Flux Schnell engines only accept steps between 6-12.")

    text_prompts = [{"text": prompt, "weight": weight}]
    payload = {
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "steps": steps,
        "engine": tool,
        "text_prompts": text_prompts
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": api_key
    }

    try:
        # Make the request to the Corcel API
        response = requests.post(CORCEL_URL, json=payload, headers=headers)

        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Try parsing the response content as JSON
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            return error_response("Failed to parse response as JSON.")

        # Extract signed URLs from the response
        signed_urls = response_data.get("signed_urls", [])
        if not signed_urls:
            return error_response("No signed URLs found in the response.")

        # Return the first signed URL
        return signed_urls[0], prompt, None, None

    except requests.exceptions.HTTPError as http_err:
        return error_response(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        return error_response(f"Request error occurred: {req_err}")
    except Exception as e:
        return error_response(f"An unexpected error occurred: {str(e)}")

# ---- Main Function ----

def main():
    """Main function to run the Corcel image generation"""

    # Define parameters for the run function
    kwargs = {
        "api_keys": "03001143-7422-4c8f-9b31-0dae5e6bfcd8",  # Replace with your actual API key
        "tool": "proteus",
        "prompt": "An autonolas robot named Pearl",
        "cfg_scale": 7,
        "height": 832,
        "width": 1024,
        "steps": 10,
        "weight": 0,
    }

    # Run the function and print the result
    result = run(**kwargs)

    # Print a readable message based on the result
    if result[0]:
        print("Successfully generated image. URL:", result[0])
    else:
        print("Failed to generate image. Error:", result[0])

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
