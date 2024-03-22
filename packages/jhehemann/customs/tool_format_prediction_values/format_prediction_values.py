# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023-2024 Valory AG
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

"""This tool takes prediction values as input and formats them in a JSON formatted string."""

import json

# Function to check if the input is a string representing a number between 0 and 1
def is_valid_number(s):
    try:
        value = float(s)
        return 0 <= value <= 1
    except ValueError:
        return False


def format_prediction_values(p_yes: str, p_no: str, confidence: str, info_utility: str) -> str:
    """
    Function to be called by assistant:
    Format the prediction values p_yes, p_no, confidence and info_utility as JSON and return the string
    """
    
    # Check if the input strings values are valid numbers between 0 and 1
    inputs = [p_yes, p_no, confidence, info_utility]
    for input_value in inputs:
        if not is_valid_number(input_value):
            return "Invalid input. Please provide valid numbers between 0 and 1."

    # Construct a dictionary with the prediction values
    prediction_values = {
        "p_yes": float(p_yes),
        "p_no": float(p_no),
        "confidence": float(confidence),
        "info_utility": float(info_utility)
    }

    # Convert the dictionary to a JSON-formatted string
    json_output = json.dumps(prediction_values, indent=4)

    return json_output