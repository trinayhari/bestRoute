#!/usr/bin/env python3
"""
OpenRouter Test Examples

This script provides example prompts for testing different models via OpenRouter,
including error handling scenarios.
"""

import os
import time
import json
from typing import Dict, List, Any
from src.api.openrouter_client_enhanced import send_prompt_to_openrouter

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

def run_test(model: str, prompt: str, system_message: str = "You are a helpful assistant."):
    """Run a test with a specific model and prompt"""
    print(f"\n--- Testing {model} ---")
    print(f"Prompt: {prompt}")
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    try:
        start_time = time.time()
        response_text, usage_stats, latency = send_prompt_to_openrouter(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=1000
        )
        total_time = time.time() - start_time
        
        print(f"Response: {response_text[:150]}...")
        print(f"Stats: {json.dumps(usage_stats, indent=2)}")
        print(f"API Latency: {latency:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print("Success!")
        return True, response_text, usage_stats
        
    except Exception as e:
        print(f"Error: {e}")
        return False, str(e), {}


def main():
    # Basic prompts for different model strengths
    test_cases = [
        # Basic cognitive tasks
        {
            "name": "Basic Question",
            "models": ["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"],
            "prompt": "What is the capital of France and what's a famous landmark there?",
            "system": "You are a helpful assistant that provides brief, accurate answers."
        },
        
        # Creative writing (Claude Sonnet should excel)
        {
            "name": "Creative Writing",
            "models": ["anthropic/claude-3-sonnet", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"],
            "prompt": "Write a short, poetic description of a sunset over the ocean.",
            "system": "You are a creative writing assistant with a poetic style."
        },
        
        # Complex reasoning (GPT-4o and Claude Opus should excel)
        {
            "name": "Complex Reasoning",
            "models": ["anthropic/claude-3-opus", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"],
            "prompt": "Explain the prisoner's dilemma in game theory and provide a real-world example.",
            "system": "You are an expert assistant who provides clear explanations of complex topics."
        },
        
        # Coding task (GPT-4o often excels)
        {
            "name": "Coding Task",
            "models": ["openai/gpt-4o", "anthropic/claude-3-opus", "mistralai/mixtral-8x7b-instruct"],
            "prompt": "Write a Python function that takes a list of integers and returns the median value.",
            "system": "You are a programming assistant. Provide clean, well-documented code."
        },
        
        # Concise response test (Haiku and smaller models should be efficient)
        {
            "name": "Conciseness Test",
            "models": ["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo", "mistralai/mistral-7b-instruct"],
            "prompt": "Explain how vaccines work in 2-3 sentences.",
            "system": "You are a concise assistant. Provide brief but complete answers."
        }
    ]
    
    # Error handling tests
    error_test_cases = [
        # Invalid model name
        {
            "name": "Invalid Model",
            "models": ["invalid/model-name"],
            "prompt": "This should trigger an error because the model doesn't exist.",
            "system": "You are a helpful assistant."
        },
        
        # Token limit exceeded
        {
            "name": "Token Limit Test",
            "models": ["openai/gpt-3.5-turbo"],
            "prompt": "Translate the following to French: " + ("Hello world. " * 5000),  # Very long prompt
            "system": "You are a translation assistant."
        },
        
        # Empty prompt test
        {
            "name": "Empty Prompt",
            "models": ["anthropic/claude-3-haiku"],
            "prompt": "",
            "system": "You are a helpful assistant."
        }
    ]
    
    # Run standard tests
    print("\n===== STANDARD MODEL TESTS =====")
    results = []
    
    for test in test_cases:
        print(f"\n\n==== {test['name']} ====")
        
        for model in test["models"]:
            success, response, stats = run_test(model, test["prompt"], test["system"])
            results.append({
                "test_name": test["name"],
                "model": model,
                "success": success,
                "tokens": stats.get("total_tokens", 0) if success else 0,
                "latency": stats.get("latency", 0) if success else 0
            })
            time.sleep(1)  # Brief pause between tests
    
    # Run error tests
    print("\n\n===== ERROR HANDLING TESTS =====")
    
    for test in error_test_cases:
        print(f"\n\n==== {test['name']} ====")
        
        for model in test["models"]:
            success, response, stats = run_test(model, test["prompt"], test["system"])
            # Should fail, so success=False is expected
            results.append({
                "test_name": test["name"],
                "model": model,
                "success": success,
                "error": response if not success else "No error"
            })
            time.sleep(1)  # Brief pause between tests
    
    # Print summary
    print("\n\n===== TEST SUMMARY =====")
    success_count = sum(1 for r in results if r["success"])
    print(f"Total tests: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")


if __name__ == "__main__":
    main() 