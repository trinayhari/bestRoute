# OpenRouter API Key Troubleshooting

This document provides guidance on how to set up and troubleshoot issues with your OpenRouter API key for the chatbot application.

## Quick Start

1. Get an API key from [OpenRouter](https://openrouter.ai/keys)
2. Run our setup helper script:
   ```bash
   python setup_env.py
   ```
3. Verify your key works:
   ```bash
   python test_api_key.py
   ```

## Common Issues and Solutions

### "Error: OpenRouter API key is required"

This error indicates the application cannot find your API key. Here are possible solutions:

1. **Check if your API key is set correctly**:

   - Run `python fix_api_key.py` to diagnose and fix common issues
   - Make sure your `.env` file contains `OPENROUTER_API_KEY=your-key-here`
   - Verify that there are no extra quotes or spaces in your API key

2. **Ensure your code is loading the environment variables**:

   - Make sure you have `python-dotenv` installed: `pip install python-dotenv`
   - Add these lines to the beginning of your main script:
     ```python
     from dotenv import load_dotenv
     load_dotenv()  # This loads variables from .env
     ```

3. **Set the environment variable manually**:
   - In Bash/Zsh: `export OPENROUTER_API_KEY=your-key-here`
   - In Windows Command Prompt: `set OPENROUTER_API_KEY=your-key-here`
   - In Windows PowerShell: `$env:OPENROUTER_API_KEY="your-key-here"`

### Invalid API Key Errors

If you're getting authentication errors:

1. **Check your API key**:

   - Verify you've copied the full key from [OpenRouter](https://openrouter.ai/keys)
   - Make sure the key starts with `sk-`
   - Remove any quotes or extra spaces

2. **Verify your account status**:

   - Check if your OpenRouter account has sufficient credits
   - Newly created API keys may take a few minutes to activate

3. **Test with our diagnostic tool**:
   ```bash
   python test_api_key.py
   ```

### Environment Loading Issues

If your application can't find environment variables:

1. **Check file locations**:

   - Make sure your `.env` file is in the correct directory (project root)
   - Verify the file permissions allow the application to read it

2. **Verify file format**:

   - The `.env` file should use the format `KEY=value` without quotes
   - Each variable should be on a new line

3. **Debug environment loading**:

   ```python
   import os
   from dotenv import load_dotenv

   # Check where dotenv is looking for the file
   print(f"Current working directory: {os.getcwd()}")

   # Load with verbose output
   load_dotenv(verbose=True)

   # Check if the variable was loaded
   api_key = os.getenv("OPENROUTER_API_KEY")
   print(f"API key loaded: {'Yes' if api_key else 'No'}")
   ```

## Setting Up Your Environment

### Option 1: Use Our Setup Script (Recommended)

Run:

```bash
python setup_env.py
```

This script will:

- Create a `.env` file from `.env.example`
- Prompt you for your API key
- Save the key in the correct format

### Option 2: Manual Setup

1. Copy the example configuration:

   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your favorite text editor:

   ```bash
   nano .env  # or vim, VSCode, etc.
   ```

3. Add your API key:
   ```
   OPENROUTER_API_KEY=your-key-here
   ```
4. Save the file

### Option 3: Environmental Variables

For temporary usage or CI/CD environments, set the variable directly:

**Linux/macOS**:

```bash
export OPENROUTER_API_KEY=your-key-here
```

**Windows Command Prompt**:

```
set OPENROUTER_API_KEY=your-key-here
```

**Windows PowerShell**:

```
$env:OPENROUTER_API_KEY="your-key-here"
```

## Advanced Troubleshooting

If you continue to experience issues, try our fix script:

```bash
python fix_api_key.py
```

This script provides detailed diagnostics and automatically fixes common issues with your API key configuration.

## API Key Security

To keep your API key secure:

1. **Never commit your API key to version control**:

   - Always add `.env` to your `.gitignore` file
   - Use `.env.example` with placeholder values for templates

2. **Restrict API key permissions**:

   - Only give your API key the permissions it needs in the OpenRouter dashboard
   - Use different API keys for development and production

3. **Rotate keys regularly**:
   - Generate new API keys periodically
   - Revoke old or compromised keys

## Need More Help?

If you continue to experience issues:

1. Run our advanced diagnostic tool and share the output:

   ```bash
   python fix_api_key.py > api_key_debug.log
   ```

2. Check if the OpenRouter API is operational: [OpenRouter Status](https://status.openrouter.ai)

3. Contact OpenRouter support with your diagnostic information
