#!/usr/bin/env python3
"""
Install Dependencies

This script installs the required dependencies for the OpenRouter chatbot.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install the required dependencies."""
    print("Installing dependencies...")
    
    # List of required packages
    dependencies = [
        "requests",
        "pyyaml",
        "python-dotenv",
        "pandas",
        "streamlit",
        "tiktoken",  # For token estimation
    ]
    
    # Install each dependency
    for package in dependencies:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
    
    print("\nDependencies installation completed.")

def check_api_key():
    """Check if OpenRouter API key is set."""
    if "OPENROUTER_API_KEY" in os.environ:
        print(f"✅ OPENROUTER_API_KEY environment variable is set.")
    else:
        print("⚠️ OPENROUTER_API_KEY environment variable is not set.")
        print("You can set it in your .env file or as an environment variable.")
        print("Example: export OPENROUTER_API_KEY=your_api_key_here")

if __name__ == "__main__":
    install_dependencies()
    check_api_key()
    
    print("\nSetup complete! You can now run the chatbot application.")
    print("To test the rule-based router, run:")
    print("  python rule_based_router_demo.py") 