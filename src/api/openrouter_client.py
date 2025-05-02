import os
import json
import requests
from typing import Dict, List, Any, Optional

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set it as OPENROUTER_API_KEY environment variable or pass it to the constructor.")
        
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://multi-llm-chatbot.streamlit.app",  # Replace with your actual domain when deployed
            "X-Title": "Multi-LLM Chat with OpenRouter"
        }

    def generate_response(self, messages: List[Dict[str, str]], model: str, 
                          temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response from a specified model via OpenRouter
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model identifier to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Response from the API as a dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
            
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            return [{"error": str(e)}] 