#!/usr/bin/env python3
"""
Test Router Fix

A simple script to test if our fixes for the token limit issue work.
"""

import os
import yaml
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the router
try:
    from src.utils.rule_based_router import RuleBasedRouter
except ImportError as e:
    print(f"Error importing RuleBasedRouter: {e}")
    sys.exit(1)

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def test_router():
    """Test the router with a simple prompt"""
    # Check for API key
    if "OPENROUTER_API_KEY" not in os.environ:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.")
        print("Please set it and try again.")
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    
    # Initialize router
    router = RuleBasedRouter(config)
    
    # Simple test prompt
    prompt = "Write a function to calculate the fibonacci sequence in Python."
    
    # Create messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    print(f"\nTesting prompt: {prompt}")
    print("\nSending to router...")
    
    try:
        start_time = datetime.now()
        response_text, metrics = router.send_prompt(messages)
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\nSUCCESS! ✅")
        print(f"Model used: {metrics['model']}")
        print(f"Prompt type: {metrics['prompt_type']}")
        print(f"Total tokens: {metrics['token_count']}")
        print(f"Response time: {duration:.2f} seconds")
        print(f"\nResponse preview: {response_text[:200]}...\n")
        
        return True
    except Exception as e:
        print(f"\nERROR: ❌ {e}")
        return False

if __name__ == "__main__":
    print("Testing the rule-based router fix...")
    success = test_router()
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. Please check the logs above for details.") 