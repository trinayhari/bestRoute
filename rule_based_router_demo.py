#!/usr/bin/env python3
"""
Rule-Based Router Demo

This script demonstrates how to use the RuleBasedRouter to route prompts
to appropriate models based on the prompt type (code, summary, question)
and prompt length.
"""

import os
import sys
import logging
import yaml
from typing import Dict, List, Any
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "router_demo.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the rule-based router
try:
    from src.utils.rule_based_router import RuleBasedRouter
except ImportError as e:
    logging.error(f"Error importing RuleBasedRouter: {e}")
    print("Make sure all dependencies are installed and the module is in the correct location.")
    sys.exit(1)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

def test_model_selection(router, test_prompts: List[Dict[str, str]]) -> None:
    """
    Test model selection for different prompts
    
    Args:
        router: Initialized RuleBasedRouter
        test_prompts: List of test prompts with labels
    """
    print("\n=== Testing Model Selection Logic ===\n")
    print(f"{'Prompt Type':<15} {'Length':<10} {'Selected Model':<30} {'Prompt':<40}")
    print("-" * 100)
    
    for test in test_prompts:
        prompt = test["prompt"]
        label = test.get("label", "N/A")
        
        # Classify the prompt
        prompt_type = router.classify_prompt(prompt)
        
        # Estimate token count
        token_count = router.estimate_token_count(prompt)
        
        # Determine length category
        length_category = router.determine_length_category(token_count)
        
        # Select model
        selected_model = router.select_model(prompt)
        
        # Print results
        truncated_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
        print(f"{prompt_type:<15} {length_category:<10} {selected_model:<30} {truncated_prompt:<40}")
        
def test_full_prompt_cycle(router, prompt: str) -> None:
    """
    Test the full prompt cycle, including API calls
    
    Args:
        router: Initialized RuleBasedRouter
        prompt: Test prompt
    """
    print("\n=== Testing Full Prompt Cycle ===\n")
    print(f"Prompt: {prompt}")
    
    # Create messages array
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Send prompt through router
        print(f"\nSending prompt to router...")
        start_time = datetime.now()
        
        response_text, metrics = router.send_prompt(messages)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        print(f"\n--- Results ---")
        print(f"Selected model: {metrics['model']}")
        print(f"Classified as: {metrics['prompt_type']}")
        print(f"Response time: {duration:.2f} seconds")
        print(f"API latency: {metrics['latency']:.2f} seconds")
        print(f"Token count: {metrics['token_count']}")
        print(f"Cost: ${router.calculate_cost(metrics['token_count'], metrics['model']):.6f}")
        print(f"\nResponse: {response_text[:500]}...")
        
    except Exception as e:
        logging.error(f"Error in test_full_prompt_cycle: {e}")
        print(f"Error: {e}")

def main():
    """Main function"""
    print("=== Rule-Based Router Demo ===")
    
    # Check for API key
    if "OPENROUTER_API_KEY" not in os.environ:
        print("\nWarning: OPENROUTER_API_KEY environment variable not set.")
        print("API calls will fail without a valid API key.")
        print("Please set it and restart or use the .env file.")
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize router
        router = RuleBasedRouter(config)
        
        # Test prompts for different categories
        test_prompts = [
            {
                "label": "Short code prompt",
                "prompt": "Write a function to calculate the fibonacci sequence in Python."
            },
            {
                "label": "Medium code prompt",
                "prompt": """Create a React component that fetches data from an API and displays it in a table with pagination.
                Include error handling and loading states. The component should allow sorting by different columns."""
            },
            {
                "label": "Short summary prompt",
                "prompt": "Summarize the key features of Streamlit for data applications."
            },
            {
                "label": "Medium summary prompt",
                "prompt": "Provide a summary of the main events in World War II, focusing on the Pacific theater."
            },
            {
                "label": "Short question prompt",
                "prompt": "What is the capital of France?"
            },
            {
                "label": "Medium question prompt",
                "prompt": "Explain how transformer models work in natural language processing. What makes them effective for tasks like translation and text generation?"
            }
        ]
        
        # Test model selection
        test_model_selection(router, test_prompts)
        
        # Ask if user wants to test API calls
        choice = input("\nDo you want to test actual API calls? (y/n): ").strip().lower()
        
        if choice == 'y':
            # Ask user which prompt to use
            print("\nChoose a prompt to test:")
            for i, test in enumerate(test_prompts):
                print(f"{i+1}. {test['label']}: {test['prompt'][:50]}...")
            
            prompt_idx = int(input("\nEnter prompt number (1-6) or 0 for custom: "))
            
            if prompt_idx == 0:
                custom_prompt = input("\nEnter your custom prompt: ")
                test_full_prompt_cycle(router, custom_prompt)
            elif 1 <= prompt_idx <= len(test_prompts):
                test_full_prompt_cycle(router, test_prompts[prompt_idx-1]["prompt"])
            else:
                print("Invalid selection.")
                
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main() 