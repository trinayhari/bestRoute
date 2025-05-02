#!/usr/bin/env python3
"""
Test Prompts for Advanced Router

This script demonstrates how to test different prompt types using
the AdvancedRouter to automatically select the most appropriate model.
"""

import os
import time
import json
from typing import Dict, List, Any
from src.utils.advanced_router import AdvancedRouter
from src.config.config_loader import load_config

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

def test_prompt_routing(router: AdvancedRouter, prompt: str, name: str):
    """Test how a prompt is routed by the advanced router"""
    print(f"\n--- Testing prompt: {name} ---")
    print(f"Prompt: {prompt}")
    
    # First, see which model would be selected
    selected_model = router.select_model(prompt)
    model_name = router.models.get(selected_model, {}).get("name", selected_model)
    
    prompt_type = router.identify_prompt_type(prompt)
    
    print(f"Detected prompt type: {prompt_type}")
    print(f"Selected model: {model_name} ({selected_model})")
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    # Send prompt and get response
    try:
        start_time = time.time()
        response_text, usage_stats = router.send_prompt(messages)
        total_time = time.time() - start_time
        
        print(f"Response: {response_text[:150]}...")
        print(f"Token usage: {usage_stats.get('total_tokens', 0)} tokens")
        print(f"Total time: {total_time:.2f}s")
        return True, response_text, usage_stats
        
    except Exception as e:
        print(f"Error: {e}")
        return False, str(e), {}

def main():
    # Load configuration
    config = load_config()
    
    # Add optimization target if not present
    if "optimization_target" not in config:
        config["optimization_target"] = "balanced"
    
    # Initialize router
    router = AdvancedRouter(config)
    print(f"Router initialized with optimization target: {router.optimization_target}")
    
    # Define test prompts for each prompt type
    test_prompts = [
        # Coding prompts
        {
            "name": "Python Function",
            "category": "coding",
            "prompt": "Write a Python function to find the longest palindrome in a string."
        },
        {
            "name": "JavaScript Class",
            "category": "coding",
            "prompt": "Create a JavaScript class for a shopping cart with methods to add, remove, and calculate total."
        },
        
        # Creative prompts
        {
            "name": "Short Story",
            "category": "creative",
            "prompt": "Write a short story about a robot discovering emotions for the first time."
        },
        {
            "name": "Poem",
            "category": "creative",
            "prompt": "Create a poem about the changing of seasons from summer to autumn."
        },
        
        # Analysis prompts
        {
            "name": "Economic Analysis",
            "category": "analysis",
            "prompt": "Analyze the potential economic impacts of a four-day work week."
        },
        {
            "name": "Compare Technologies",
            "category": "analysis",
            "prompt": "Compare and contrast relational databases and NoSQL databases. What are the pros and cons of each?"
        },
        
        # Quick questions
        {
            "name": "Simple Question",
            "category": "quick_questions",
            "prompt": "What is photosynthesis and how does it work?"
        },
        {
            "name": "How-to Question",
            "category": "quick_questions",
            "prompt": "How to make sourdough bread at home?"
        },
        
        # Mixed (should be interesting to see routing decisions)
        {
            "name": "Mixed - Code + Analysis",
            "category": "mixed",
            "prompt": "Explain the concept of recursion and provide a Python example of a recursive function to calculate factorial."
        },
        
        # Edge case - Very short
        {
            "name": "Edge Case - Very Short",
            "category": "edge_case",
            "prompt": "Hello?"
        }
    ]
    
    # Test each optimization target
    optimization_targets = ["balanced", "speed", "cost", "quality"]
    
    for target in optimization_targets:
        print(f"\n\n===== TESTING WITH OPTIMIZATION TARGET: {target} =====")
        router.optimization_target = target
        router._init_router_params()
        
        # Test with selected prompts
        for prompt_info in test_prompts:
            test_prompt_routing(router, prompt_info["prompt"], prompt_info["name"])
            time.sleep(1)  # Brief pause between tests
    
    # Demonstrate error handling
    print("\n\n===== ERROR HANDLING DEMONSTRATION =====")
    
    # Invalid model forced
    print("\n--- Testing with invalid model override ---")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "This should fail because the model doesn't exist."}
        ]
        response_text, usage_stats = router.send_prompt(messages, model_id="invalid/model-name")
        print("This should not succeed!")
    except Exception as e:
        print(f"Expected error caught: {e}")
    
    # Very long prompt
    print("\n--- Testing with extremely long prompt ---")
    long_prompt = "Translate this to Spanish: " + ("Hello world. " * 2000)
    try:
        test_prompt_routing(router, long_prompt, "Token Limit Test")
    except Exception as e:
        print(f"Error with long prompt: {e}")
    
    print("\n===== TEST COMPLETED =====")


if __name__ == "__main__":
    main() 