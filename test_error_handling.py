#!/usr/bin/env python3
"""
OpenRouter Error Handling Tests

This script tests various error scenarios when calling the OpenRouter API
and demonstrates how to handle them gracefully.
"""

import os
import time
import json
from typing import Dict, List, Any
from src.api.openrouter_client_enhanced import send_prompt_to_openrouter

def test_error_scenario(scenario_name: str, model: str, 
                        messages: List[Dict[str, str]], **kwargs):
    """Test an error scenario and print the results"""
    print(f"\n===== Testing: {scenario_name} =====")
    print(f"Model: {model}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"Additional params: {kwargs}")
    
    try:
        start_time = time.time()
        response, usage, latency = send_prompt_to_openrouter(
            messages=messages,
            model=model,
            **kwargs
        )
        end_time = time.time()
        
        print("\n✅ SUCCESS (expected error didn't occur)")
        print(f"Response: {response[:100]}...")
        print(f"Usage: {usage}")
        print(f"Time: {end_time - start_time:.2f}s")
        
    except Exception as e:
        print("\n⚠️ ERROR CAUGHT")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        # Try to extract HTTP status code if it's a request exception
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            print(f"HTTP status: {e.response.status_code}")
        
        print("\nHandling strategy:")
        print(f"✓ Log the error: error_logger.log('{type(e).__name__}', '{str(e)}')")
        print("✓ Retry with backoff: retry_with_exponential_backoff(function, max_retries=3)")
        print("✓ Fallback to alternate model: try_alternate_model('openai/gpt-3.5-turbo')")

def main():
    print("OpenRouter API Error Handling Tests")
    print("-----------------------------------")
    
    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # 1. Test with invalid API key
    try:
        invalid_key_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        test_error_scenario(
            "Invalid API Key",
            "anthropic/claude-3-haiku",
            invalid_key_messages,
            api_key="invalid_api_key_test_12345"
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 2. Test with invalid model name
    try:
        invalid_model_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        test_error_scenario(
            "Invalid Model Name",
            "invalid/model-does-not-exist",
            invalid_model_messages
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 3. Test with empty messages array
    try:
        test_error_scenario(
            "Empty Messages Array",
            "anthropic/claude-3-haiku",
            []
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 4. Test with malformed messages
    try:
        malformed_messages = [
            {"wrong_key": "system", "text": "You are a helpful assistant."},
            {"role": "user", "wrong_content_key": "Hello, how are you?"}
        ]
        
        test_error_scenario(
            "Malformed Messages",
            "anthropic/claude-3-haiku",
            malformed_messages
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 5. Test with token limit exceeded
    try:
        # Generate a very long prompt
        long_text = "Translate this text to French. " + ("Hello world. " * 5000)
        
        token_limit_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": long_text}
        ]
        
        test_error_scenario(
            "Token Limit Exceeded",
            "anthropic/claude-3-haiku",
            token_limit_messages,
            max_tokens=100
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 6. Test with invalid parameters
    try:
        invalid_params_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        test_error_scenario(
            "Invalid Parameters",
            "anthropic/claude-3-haiku",
            invalid_params_messages,
            temperature=5.0,  # Temperature should be between 0 and 1
            invalid_parameter="test"
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 7. Test with timeout
    try:
        timeout_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a detailed 5000 word essay on the history of artificial intelligence."}
        ]
        
        # Note: This may not always trigger a timeout, but reduces the timeout value to increase chances
        test_error_scenario(
            "Request Timeout",
            "anthropic/claude-3-opus",  # Using a larger model for more complex task
            timeout_messages,
            max_tokens=1000,
            timeout=1  # Very short timeout of 1 second
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    # 8. Test error handling when missing required message fields
    try:
        missing_field_messages = [
            {"role": "system"},  # Missing content
            {"content": "Hello, how are you?"}  # Missing role
        ]
        
        test_error_scenario(
            "Missing Required Fields",
            "anthropic/claude-3-haiku",
            missing_field_messages
        )
    except Exception as e:
        print(f"Unexpected exception in test runner: {e}")
    
    print("\n===== DEMONSTRATION OF ERROR RECOVERY STRATEGIES =====")
    print("""
Example implementation of error recovery:

```python
def send_prompt_with_recovery(messages, model, max_retries=3):
    # Try primary model
    try:
        response, usage, latency = send_prompt_to_openrouter(
            messages=messages,
            model=model
        )
        return response, model
        
    except Exception as primary_error:
        # Log the error
        logger.error(f"Error with primary model {model}: {str(primary_error)}")
        
        # List of fallback models in order of preference
        fallbacks = ["openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"]
        
        # Try fallback models
        for fallback_model in fallbacks:
            try:
                print(f"Trying fallback model: {fallback_model}")
                response, usage, latency = send_prompt_to_openrouter(
                    messages=messages,
                    model=fallback_model
                )
                return response, fallback_model
                
            except Exception as fallback_error:
                logger.error(f"Error with fallback model {fallback_model}: {str(fallback_error)}")
                continue
        
        # If all models fail, raise the original error
        raise primary_error
```

The above function provides:
1. Automatic fallback to different models
2. Error logging
3. Ultimate error propagation if all attempts fail
    """)

if __name__ == "__main__":
    main() 