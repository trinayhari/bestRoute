import os
import json
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", f"api_calls_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_environment():
    """Load environment variables from .env file if available"""
    try:
        from dotenv import load_dotenv
        # Try to load from a .env file if it exists
        env_loaded = load_dotenv()
        if env_loaded:
            logging.info("Environment variables loaded from .env file")
        return env_loaded
    except ImportError:
        # python-dotenv is not installed
        if os.path.exists(".env"):
            warnings.warn(
                "Found .env file but python-dotenv is not installed. "
                "Install it with: pip install python-dotenv"
            )
        return False

# Try to load environment variables
load_environment()

def check_api_key(api_key: Optional[str] = None) -> str:
    """
    Check if the API key is set and valid
    
    Args:
        api_key: API key to check, or None to check environment variable
        
    Returns:
        A valid API key
        
    Raises:
        ValueError: If no valid API key is found
    """
    # First check the provided API key
    if api_key:
        # Remove any quotes that might be accidentally included
        api_key = api_key.strip('\'"')
        if api_key:
            return api_key
    
    # Then check environment variable
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        # Remove any quotes that might be accidentally included
        env_key = env_key.strip('\'"')
        if env_key:
            return env_key
    
    # No valid key found
    raise ValueError(
        "OpenRouter API key is required. Set it as OPENROUTER_API_KEY environment variable "
        "or pass it to the function. Make sure you've created a .env file or exported "
        "the variable in your shell."
    )

def send_prompt_to_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> Tuple[str, Dict[str, Any], float]:
    """
    Send a prompt to the OpenRouter API and return the response with detailed metrics.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model ID to use (e.g., 'anthropic/claude-3-opus')
        api_key: OpenRouter API key (will use env var if not provided)
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Tuple containing:
        - response_text: The text response from the model
        - usage_stats: Dictionary with token usage statistics
        - latency: The round-trip time in seconds
    """
    # Make sure messages is valid
    if not messages:
        raise ValueError("Messages list cannot be empty")
    
    # Validate all messages have required fields
    for i, msg in enumerate(messages):
        if "role" not in msg:
            raise ValueError(f"Message at index {i} is missing the 'role' field")
        if "content" not in msg:
            raise ValueError(f"Message at index {i} is missing the 'content' field")
    
    # Try to load environment variables if not already done
    if api_key is None and "OPENROUTER_API_KEY" not in os.environ:
        load_environment()
    
    # Get API key from environment if not provided
    try:
        api_key = check_api_key(api_key)
    except ValueError as e:
        logging.error(f"API Key Error: {str(e)}")
        raise
    
    # Set up request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://multi-llm-chatbot.streamlit.app",  # Replace with your actual domain
        "X-Title": "Multi-LLM Chat with OpenRouter"
    }
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    # Record start time for latency calculation
    start_time = time.time()
    
    try:
        # Make the API request
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=kwargs.get("timeout", 60)
        )
        
        # Calculate round-trip latency
        latency = time.time() - start_time
        
        # Process the response
        if response.status_code != 200:
            error_msg = f"OpenRouter API Error (Status {response.status_code}): "
            try:
                error_detail = response.json()
                error_msg += json.dumps(error_detail)
            except Exception:
                error_msg += response.text
            
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
        response_data = response.json()
        
        # Extract the response text
        if not response_data.get("choices"):
            error_msg = f"Unexpected response format: {json.dumps(response_data)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
        response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract usage statistics
        usage_stats = {
            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": response_data.get("usage", {}).get("total_tokens", 0),
            "model": response_data.get("model", model),
            "latency": latency,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the request and response metrics
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "latency": latency,
            "status": "success",
            **usage_stats
        }
        logging.info(json.dumps(log_entry))
        
        return response_text, usage_stats, latency
        
    except requests.exceptions.RequestException as e:
        # Calculate latency even for errors
        latency = time.time() - start_time
        
        # Log the error
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "latency": latency,
            "status": "error",
            "error": str(e)
        }
        logging.error(json.dumps(log_entry))
        
        # Re-raise the exception with more context
        raise RuntimeError(f"Error calling OpenRouter API: {str(e)}. Latency: {latency:.2f}s")


# Example usage:
if __name__ == "__main__":
    print("OpenRouter API Client - Example Usage")
    
    # Check for API key
    try:
        api_key = check_api_key()
        print(f"✅ Found API key: {api_key[:4]}...{api_key[-4:]}")
    except ValueError as e:
        print(f"❌ {str(e)}")
        api_key = input("Enter your OpenRouter API key to continue: ").strip()
    
    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke."}
    ]
    
    try:
        # Call the function
        print("\nSending request to OpenRouter API...")
        response_text, usage, latency = send_prompt_to_openrouter(
            messages=messages,
            model="anthropic/claude-3-haiku",
            temperature=0.7,
            max_tokens=100,
            api_key=api_key
        )
        
        # Print the results
        print("\n✅ Success!")
        print(f"\nResponse: {response_text}")
        print(f"\nToken usage: {usage}")
        print(f"\nLatency: {latency:.2f} seconds")
        
    except Exception as e:
        print(f"\n❌ Error: {e}") 