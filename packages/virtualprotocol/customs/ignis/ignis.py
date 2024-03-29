"""This module implements a Mech tool for chatting with Ignis model on VirtualProtocol."""

import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
import jwt

client = None

class VirtualProtocolClientManager:
    def __init__(self, api_key: str, api_secret: str):
        # Initialize the client here
        global client
        if client is None or client.api_key != api_key or client.api_secret != api_secret:
            client = VirtualProtocolClient(api_key, api_secret)

    def __enter__(self):
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


class VirtualProtocolClient:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = self.get_access_token()
        self.decoded_data = self.decode_access_token(self.access_token)
    
    def prompt (self, prompt: str) -> str:
        access_token = self.get_access_token()
        decoded_data = self.decode_access_token(access_token)
        runner_url = self.get_runner_url(decoded_data)
        
        if (access_token is None) or (runner_url is None):
                return f"Error: Access token or runner URL is None", None, None, None
        
        data = {
            "text": prompt,
            "skipTTS": True,  
            "userName": "Olas",
            "botName": "Ignis",
        }
        
        try:
            response = requests.post(
                f"{runner_url}/prompts",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}",
                },
                json=data,
            )
            
            response_data = response.json()
            if response.status_code == 200:
                return response_data['text']
            return f"Error: Non-200 response ({response.status_code}): {response.text}",
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to make request to runner: {e}") from e

    def get_access_token(self) -> str:
        if hasattr(self, "access_token") and self.access_token is not None:
            decoded_data = self.decoded_data
            if decoded_data["exp"] > time.time():
                return self.access_token
        
        access_token = self.request_access_token()
        return access_token
    
    def get_runner_url(self,decoded_data) -> str:
        return decoded_data["runner"]
    
    def decode_access_token(self,access_token) -> str:
        try:
            decoded_data = jwt.decode(jwt=access_token,
                              options={"verify_signature": False})
            
            self.decoded_data = decoded_data
            
            return decoded_data
        except (KeyError, json.JSONDecodeError) as e:
            raise ConnectionError(f"Failed to get runner url: {e}") from e
            
    def request_access_token(self) -> str:
        try:
            response = requests.post(
                f"https://api.virtuals.io/api/access/token",
                headers={
                    "Content-Type": "application/json",
                },
                json={"apiKey": self.api_key, "apiSecret": self.api_secret},
            )
            
            self.access_token = response.json()["accessToken"]
            
            
            return response.json()["accessToken"]
        except (KeyError, json.JSONDecodeError) as e:
            raise ConnectionError(f"Failed to generate access token: {e}") from e
        

def run(**kwargs) -> Tuple[Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """Run the task"""
    api_keys = kwargs["api_keys"]['ignis'].split(":")
    prompt = kwargs["prompt"]
    
    if (len(api_keys) != 2) or (api_keys[0] is None) or (api_keys[1] is None):
        return f"Error: API key or API secret is missing.", None, None, None

    api_key = api_keys[0]
    api_secret = api_keys[1]
      
    with VirtualProtocolClientManager(api_key, api_secret):
        try:
            return client.prompt(prompt=prompt), None, None, None
        except Exception as e:
            return f"Failed to run task: {e}", None, None, None
