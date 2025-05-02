#!/usr/bin/env python3
"""
Setup Environment File Helper

This script helps set up the .env file from .env.example and prompts for the API key.
"""

import os
import sys
import shutil
from pathlib import Path

# Try to import dotenv, install if missing
try:
    from dotenv import load_dotenv, set_key
    DOTENV_AVAILABLE = True
except ImportError:
    print("python-dotenv not installed. This script requires python-dotenv.")
    print("Please install it with: pip install python-dotenv")
    sys.exit(1)

def main():
    print("\n==== OpenRouter Chatbot Environment Setup ====")
    
    env_path = Path('.env')
    example_path = Path('.env.example')
    
    # Check if .env already exists
    if env_path.exists():
        print("\n.env file already exists. Do you want to overwrite it? (y/n)")
        choice = input("> ").strip().lower()
        if choice != 'y':
            print("Setup cancelled. Your .env file remains unchanged.")
            return
    
    # Check if .env.example exists
    if not example_path.exists():
        print("\nError: .env.example file not found.")
        print("Please make sure you're running this script from the project root directory.")
        return
    
    # Copy .env.example to .env
    shutil.copy(example_path, env_path)
    print("\nCreated .env file from .env.example template.")
    
    # Ask for API key
    print("\nPlease enter your OpenRouter API key.")
    print("You can get one at: https://openrouter.ai/keys")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("\nNo API key provided. You'll need to manually add it to the .env file later.")
    else:
        # Add API key to .env file
        set_key(str(env_path), "OPENROUTER_API_KEY", api_key)
        print("\nAPI key added to .env file.")
    
    print("\n==== Setup Complete ====")
    print("You can now run the chatbot application.")
    print("If you need to change settings later, edit the .env file directly.")

if __name__ == "__main__":
    main() 