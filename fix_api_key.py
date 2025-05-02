#!/usr/bin/env python3
"""
OpenRouter API Key Fixer

This script helps diagnose and fix common issues with OpenRouter API keys.
It will:
1. Check if the API key is set in the environment
2. Check if a .env file exists and if it contains the API key
3. Create or update the .env file if needed
4. Test the API key against the OpenRouter API
5. Fix any issues that are found
"""

import os
import sys
import requests
import subprocess
import json
import re
from pathlib import Path

# Try to import dotenv, install if missing
try:
    from dotenv import load_dotenv, set_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv not installed. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv, set_key
        DOTENV_AVAILABLE = True
        print("Successfully installed python-dotenv")
    except Exception as e:
        print(f"Failed to install python-dotenv: {e}")

# Colors for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"

def print_header(message):
    print(f"\n{Colors.PURPLE}==== {message} ===={Colors.RESET}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.RESET}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.RESET}")

def check_api_key_format(api_key):
    """Check if the API key format looks valid"""
    # Most OpenRouter keys start with 'sk-'
    if not api_key.startswith('sk-'):
        return False, "API key should start with 'sk-'"
    
    # Check for quotes or spaces
    if '"' in api_key or "'" in api_key:
        return False, "API key contains quotes which should be removed"
    
    if ' ' in api_key:
        return False, "API key contains spaces which should be removed"
    
    # Most API keys are quite long
    if len(api_key) < 20:
        return False, "API key seems too short"
        
    return True, "API key format looks valid"

def check_env_file():
    """Check if .env file exists and contains the API key"""
    env_path = Path('.env')
    example_path = Path('.env.example')
    
    if not env_path.exists():
        if example_path.exists():
            print_warning(".env file not found, but .env.example exists")
            print_info("You should copy .env.example to .env and fill in your API key")
        else:
            print_warning(".env file not found. We'll create one for you.")
        return False, None
    
    # Read the .env file
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    # Check if it contains OPENROUTER_API_KEY
    match = re.search(r'OPENROUTER_API_KEY\s*=\s*([^\n#]+)', env_content)
    if not match:
        print_warning("OPENROUTER_API_KEY not found in .env file")
        return True, None
    
    api_key = match.group(1).strip()
    is_valid, message = check_api_key_format(api_key)
    
    if not is_valid:
        print_warning(f"Found API key in .env file but {message.lower()}")
        return True, api_key
    
    print_success("Found valid API key in .env file")
    return True, api_key

def test_api_key(api_key):
    """Test the API key against the OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://api-key-fixer.local"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            model_count = len(data.get("data", []))
            return True, f"API key is valid! Retrieved {model_count} available models."
        
        # Handle common error responses
        if response.status_code == 401:
            return False, "API key is invalid. Please check it at https://openrouter.ai/keys"
        if response.status_code == 429:
            return False, "Rate limit exceeded. Please try again later."
        
        return False, f"API request failed with status code {response.status_code}: {response.text}"
    
    except Exception as e:
        return False, f"Error testing API key: {str(e)}"

def create_or_update_env_file(api_key):
    """Create or update the .env file with the API key"""
    env_path = Path('.env')
    
    if env_path.exists():
        # Update existing file
        if DOTENV_AVAILABLE:
            set_key(str(env_path), "OPENROUTER_API_KEY", api_key)
            print_success("Updated API key in .env file")
        else:
            # Manually update the file
            with open(env_path, 'r') as f:
                content = f.read()
            
            if "OPENROUTER_API_KEY" in content:
                content = re.sub(
                    r'OPENROUTER_API_KEY\s*=\s*[^\n#]*', 
                    f'OPENROUTER_API_KEY={api_key}', 
                    content
                )
            else:
                content += f"\nOPENROUTER_API_KEY={api_key}\n"
            
            with open(env_path, 'w') as f:
                f.write(content)
            print_success("Updated API key in .env file")
    else:
        # Create new file
        with open(env_path, 'w') as f:
            f.write(f"# OpenRouter API Key - Get yours at https://openrouter.ai/keys\n")
            f.write(f"OPENROUTER_API_KEY={api_key}\n")
        print_success("Created .env file with API key")

def main():
    print_header("OpenRouter API Key Fixer")
    print("This script will check your OpenRouter API key configuration and help fix any issues.")
    
    # Load environment variables from .env file if it exists
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    # First check if API key is in environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if api_key:
        print_info(f"Found API key in environment: {api_key[:4]}...{api_key[-4:]}")
        is_valid, message = check_api_key_format(api_key)
        if not is_valid:
            print_warning(message)
    else:
        print_warning("OPENROUTER_API_KEY not found in environment variables")
    
    # Check .env file
    env_exists, env_api_key = check_env_file()
    
    # Determine which API key to use
    if not api_key and not env_api_key:
        print_header("API Key Setup")
        print_info("No API key found. Please enter your OpenRouter API key.")
        print_info("You can get one at: https://openrouter.ai/keys")
        api_key = input("Enter your OpenRouter API key: ").strip()
        
        is_valid, message = check_api_key_format(api_key)
        if not is_valid:
            print_warning(message)
            print_info("Continuing anyway, but this might cause issues.")
    elif not api_key and env_api_key:
        # Found key in .env but it's not in environment
        api_key = env_api_key
        print_warning("API key found in .env but not in environment.")
        print_info("This suggests the .env file is not being loaded properly.")
    
    # Test the API key if we have one
    if api_key:
        print_header("Testing API Key")
        success, message = test_api_key(api_key)
        
        if success:
            print_success(message)
            
            # Make sure the API key is in the .env file
            if not env_exists or not env_api_key or env_api_key != api_key:
                create_or_update_env_file(api_key)
                
            print_header("Final Steps")
            print_success("Your OpenRouter API key is properly configured!")
            print_info("Make sure your application is loading the .env file:")
            print("```python")
            print("from dotenv import load_dotenv")
            print("load_dotenv()  # This loads the variables from .env")
            print("```")
        else:
            print_error(message)
            print_header("Troubleshooting")
            print_info("1. Check that you've copied the API key correctly from https://openrouter.ai/keys")
            print_info("2. Ensure you're using the correct environment variable name: OPENROUTER_API_KEY")
            print_info("3. If you just created the API key, it might take a few minutes to activate")
            print_info("4. Try again with a new API key if the issue persists")
    else:
        print_error("No API key provided. Cannot continue.")
        sys.exit(1)
    
    print_header("Done")

if __name__ == "__main__":
    main() 