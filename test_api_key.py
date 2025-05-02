#!/usr/bin/env python3
"""
Direct OpenRouter API Test

This script directly tests the OpenRouter API without any additional layers.
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

def test_openrouter_api():
    """Directly test the OpenRouter API"""
    # Try to load from .env file
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables")
        api_key = input("Please enter your OpenRouter API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            sys.exit(1)
    
    # Test the models endpoint first (doesn't cost tokens)
    print(f"Testing OpenRouter API with key: {api_key[:4]}...{api_key[-4:]}")
    print("Fetching available models...")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://test-script.local"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            model_count = len(data.get("data", []))
            print(f"✅ SUCCESS: Retrieved {model_count} available models")
            
            # Print first few models
            print("\nAvailable models:")
            for i, model in enumerate(data.get("data", [])[:5]):
                print(f"{i+1}. {model.get('id')}")
            
            if model_count > 5:
                print(f"...and {model_count - 5} more")
                
        else:
            print(f"❌ ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
    
    # Now test a basic completion
    print("\nTesting a basic model completion...")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://test-script.local"
        }
        
        payload = {
            "model": "anthropic/claude-3-haiku",  # Using a relatively inexpensive model
            "messages": [
                {"role": "user", "content": "Say hello in one short sentence."}
            ],
            "max_tokens": 20
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Get usage stats
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            print(f"✅ SUCCESS: Model responded with: \"{content}\"")
            print(f"Token usage: {prompt_tokens} (prompt) + {completion_tokens} (completion)")
            print("\nYour API key is working correctly with OpenRouter!")
        else:
            print(f"❌ ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        
    print("\nTest completed!")

if __name__ == "__main__":
    test_openrouter_api() 