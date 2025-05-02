import os
import json
import requests
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", f"api_calls_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

class ModelClient:
    """Base class for LLM API clients"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    def generate_response(self, messages: List[Dict[str, str]], 
                         model: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the model
        
        Args:
            messages: List of message dictionaries
            model: The model to use
            **kwargs: Additional parameters for the model
            
        Returns:
            Model response
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def _log_request(self, model: str, messages: List[Dict[str, str]], response: Dict[str, Any]) -> None:
        """
        Log API request and response information
        
        Args:
            model: The model used
            messages: The messages sent
            response: The response received
        """
        try:
            # Create log entry
            prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "success": "error" not in response
            }
            
            # Log to file
            logging.info(json.dumps(log_entry))
            
        except Exception as e:
            logging.error(f"Error logging API call: {str(e)}")


class OpenRouterClient(ModelClient):
    """Client for the OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
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
                          temperature: float = 0.7, max_tokens: int = 1000,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate a response from a specified model via OpenRouter
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model identifier to use
            temperature: Controls randomness (0-1)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Response from the API as a dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Log the request
            self._log_request(model, messages, response_data)
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            error_response = {"error": str(e)}
            self._log_request(model, messages, error_response)
            return error_response
            
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            return [{"error": str(e)}] 