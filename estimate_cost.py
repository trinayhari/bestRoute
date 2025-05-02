#!/usr/bin/env python3
"""
OpenRouter Cost Estimator

A utility for estimating costs of using different models with OpenRouter.
"""

import os
import argparse
import yaml
import tiktoken
import re
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

# Default config location
CONFIG_PATH = "config.yaml"

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"models": {}}

def estimate_tokens(text: str, model_family: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a text string
    
    Args:
        text: The input text
        model_family: The model family to use for encoding (affects tokenization)
        
    Returns:
        Estimated token count
    """
    try:
        # Use tiktoken for OpenAI-compatible models
        if "gpt" in model_family.lower():
            encoding = tiktoken.encoding_for_model(model_family)
            return len(encoding.encode(text))
        
        # For Claude models (rule of thumb)
        if "claude" in model_family.lower():
            # Claude tokenization is roughly 1 token per 4 characters (English text)
            return len(text) // 4
            
        # For Mistral and other models (fallback)
        # A generalized approximation for English text
        return len(text) // 4
    except Exception as e:
        print(f"Error estimating tokens, using fallback method: {e}")
        # Fallback to character-based estimation (conservative)
        return len(text) // 4

def format_messages(system: str, user: str) -> List[Dict[str, str]]:
    """Format system and user messages into the OpenRouter format"""
    messages = []
    
    if system:
        messages.append({"role": "system", "content": system})
    
    messages.append({"role": "user", "content": user})
    
    return messages

def calculate_cost(token_count: int, model_id: str, config: Dict[str, Any]) -> float:
    """
    Calculate the estimated cost for a given token count and model
    
    Args:
        token_count: Number of tokens
        model_id: Model identifier (e.g., "anthropic/claude-3-haiku")
        config: Configuration dictionary
        
    Returns:
        Estimated cost in USD
    """
    models = config.get("models", {})
    model_info = models.get(model_id, {})
    
    cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
    
    # Calculate cost
    estimated_cost = (token_count * cost_per_1k) / 1000
    
    return estimated_cost

def estimate_completion_tokens(prompt_tokens: int, model_id: str) -> int:
    """
    Estimate the number of completion tokens based on the prompt
    
    Args:
        prompt_tokens: Number of tokens in the prompt
        model_id: Model identifier
        
    Returns:
        Estimated completion tokens
    """
    # Simple heuristic: Completion is typically 1.5x the prompt for general questions
    # But we can refine this based on model characteristics
    if "opus" in model_id or "gpt-4" in model_id:
        # More verbose models
        return int(prompt_tokens * 2)
    elif "turbo" in model_id or "haiku" in model_id or "mistral-7b" in model_id:
        # More concise models
        return int(prompt_tokens * 1.2)
    else:
        # Default case
        return int(prompt_tokens * 1.5)

def get_model_family(model_id: str) -> str:
    """Determine the model family for tokenization purposes"""
    if "gpt" in model_id:
        if "gpt-4" in model_id:
            return "gpt-4"
        else:
            return "gpt-3.5-turbo"
    elif "claude" in model_id:
        return "claude"
    elif "mistral" in model_id:
        return "mistral"
    else:
        return "gpt-3.5-turbo"  # default fallback

def list_available_models(config: Dict[str, Any]):
    """List all available models and their pricing"""
    models = config.get("models", {})
    
    if not models:
        print("No models found in configuration.")
        return
    
    print("\n=== Available Models ===")
    print(f"{'Model ID':<30} {'Name':<20} {'Cost per 1K tokens':<20} {'Max Tokens':<12}")
    print("-" * 80)
    
    for model_id, model_info in models.items():
        name = model_info.get("name", "Unknown")
        cost = model_info.get("cost_per_1k_tokens", "Unknown")
        max_tokens = model_info.get("max_tokens", "Unknown")
        
        print(f"{model_id:<30} {name:<20} ${cost:<19.5f} {max_tokens:<12}")

def main():
    parser = argparse.ArgumentParser(description="Estimate costs for OpenRouter API usage")
    
    # Main arguments
    parser.add_argument("--prompt", type=str, help="Text prompt to estimate tokens for")
    parser.add_argument("--file", type=str, help="File containing prompt text")
    parser.add_argument("--system", type=str, help="System message to include")
    parser.add_argument("--model", type=str, help="Model ID to use for estimation")
    parser.add_argument("--list-models", action="store_true", help="List available models and their pricing")
    
    # Optional arguments
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # List models if requested
    if args.list_models:
        list_available_models(config)
        return
    
    # Get the prompt text
    prompt_text = ""
    if args.prompt:
        prompt_text = args.prompt
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                prompt_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    
    # Get system message
    system_text = args.system or ""
    
    # If no prompt provided, enter interactive mode
    if not prompt_text and not args.list_models:
        print("\n=== OpenRouter Cost Estimator (Interactive Mode) ===")
        system_text = input("Enter system message (optional): ")
        prompt_text = input("Enter your prompt: ")
    
    # If we have a prompt, estimate costs
    if prompt_text:
        # Format messages
        messages = format_messages(system_text, prompt_text)
        
        # Convert messages to text for token estimation
        full_text = system_text + "\n" + prompt_text
        
        # Get available models
        models = config.get("models", {})
        
        if args.model:
            # Estimate for a specific model
            if args.model not in models:
                print(f"Model '{args.model}' not found in configuration")
                return
                
            model_family = get_model_family(args.model)
            prompt_tokens = estimate_tokens(full_text, model_family)
            completion_tokens = estimate_completion_tokens(prompt_tokens, args.model)
            total_tokens = prompt_tokens + completion_tokens
            cost = calculate_cost(total_tokens, args.model, config)
            
            print(f"\nEstimate for model: {args.model}")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Estimated completion tokens: {completion_tokens}")
            print(f"Total tokens: {total_tokens}")
            print(f"Estimated cost: ${cost:.5f}")
            
        else:
            # Compare all models
            print("\n=== Cost Comparison ===")
            print(f"{'Model':<30} {'Prompt Tokens':<15} {'Completion':<15} {'Total':<10} {'Cost':<10}")
            print("-" * 80)
            
            for model_id, model_info in models.items():
                model_family = get_model_family(model_id)
                prompt_tokens = estimate_tokens(full_text, model_family)
                completion_tokens = estimate_completion_tokens(prompt_tokens, model_id)
                total_tokens = prompt_tokens + completion_tokens
                cost = calculate_cost(total_tokens, model_id, config)
                
                print(f"{model_id:<30} {prompt_tokens:<15} {completion_tokens:<15} {total_tokens:<10} ${cost:.5f}")

if __name__ == "__main__":
    main() 