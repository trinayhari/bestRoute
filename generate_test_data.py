#!/usr/bin/env python3
"""
Generate Test Data for Cost Dashboard

This script generates sample cost data for testing the dashboard.
"""

import os
import yaml
import random
import datetime
from src.utils.cost_tracker import CostTracker

def load_config():
    """Load the configuration file"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"models": {}}

def generate_test_data(days=7, calls_per_day=5):
    """Generate test data for the cost dashboard"""
    config = load_config()
    tracker = CostTracker(config)
    
    # List of models to use for test data
    models = list(config.get("models", {}).keys())
    if not models:
        models = ["anthropic/claude-3-haiku", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"]
    
    print(f"Generating test data for {days} days with {calls_per_day} calls per day")
    print(f"Using models: {', '.join(models)}")
    
    # Generate data for the past N days
    for day in range(days):
        # Create date for this batch
        date = datetime.datetime.now() - datetime.timedelta(days=day)
        session_id = date.strftime("%Y%m%d%H%M%S")
        
        # Generate multiple calls per day
        for call in range(calls_per_day):
            # Select random model
            model = random.choice(models)
            
            # Generate random token counts
            prompt_tokens = random.randint(10, 500)
            completion_tokens = random.randint(50, 1000)
            total_tokens = prompt_tokens + completion_tokens
            
            # Log the API call
            try:
                tracker.log_api_call(
                    model=model,
                    usage_stats={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    session_id=session_id
                )
                print(f"Added data point: {date.strftime('%Y-%m-%d')} - {model} - {total_tokens} tokens")
            except Exception as e:
                print(f"Error logging API call: {e}")
    
    # Get summary after generation
    session_summary = tracker.get_session_summary()
    print("\nTest data generation complete.")
    print(f"Total recorded API calls: {tracker.cost_data.shape[0]}")
    print(f"Date range: {tracker.cost_data['timestamp'].min()} to {tracker.cost_data['timestamp'].max()}")
    print(f"Models used: {tracker.cost_data['model'].nunique()}")
    print(f"Total estimated cost: ${tracker.cost_data['cost'].sum():.6f}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate test data
    generate_test_data(days=30, calls_per_day=10) 