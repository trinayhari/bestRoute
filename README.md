# OpenRouter Chatbot

A Streamlit-based chatbot application that uses rule-based routing to select the most appropriate AI model based on prompt type and length.

## Features

- **Rule-Based Routing**: Automatically selects the best model based on whether your prompt is code-related, a summary request, or a general question
- **Multiple Models**: Supports various models from OpenAI, Anthropic, and Mistral AI through OpenRouter
- **Cost Tracking**: Displays cost, token usage, and latency for each response
- **Usage Analytics**: Shows conversation summary with costs, token usage, and model distribution
- **Manual Override**: Option to manually select a specific model instead of using automatic routing

## Detailed Installation Guide

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (get one at [OpenRouter](https://openrouter.ai))
- Git (for cloning the repository)
- pip (Python package installer)

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd openrouter_chatbot
   ```

2. **Create and Activate Virtual Environment** (Recommended)

   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the project root:

   ```bash
   # On Windows
   echo OPENROUTER_API_KEY=your-api-key-here > .env

   # On macOS/Linux
   echo "OPENROUTER_API_KEY=your-api-key-here" > .env
   ```

   Or use the setup script:

   ```bash
   python setup_env.py
   ```

5. **Verify Installation**
   ```bash
   # Check if all dependencies are installed correctly
   pip list
   ```

### Running the Application

1. **Start the Main Application**

   ```bash
   # For the unified app
   streamlit run unified_app.py

   # For the multi-page application
   streamlit run Home.py
   ```

2. **Access the Application**

   - Open your browser and go to [http://localhost:8501](http://localhost:8501)
   - The application should load with the chat interface

3. **Additional Features**
   - Run the cost dashboard:
     ```bash
     streamlit run cost_dashboard.py
     ```
   - Run model comparison:
     ```bash
     streamlit run model_comparison.py
     ```

### Troubleshooting Common Issues

1. **API Key Issues**

   - Verify your API key is correctly set in `.env`
   - Check if the API key is valid at OpenRouter
   - Try running `python check_api_key.py` to verify

2. **Dependency Issues**

   - If you encounter missing module errors:
     ```bash
     pip install -r requirements.txt --upgrade
     ```
   - For specific version conflicts:
     ```bash
     pip install streamlit==1.30.0 pandas==2.1.4
     ```

3. **Port Conflicts**

   - If port 8501 is in use:
     ```bash
     streamlit run Home.py --server.port 8502
     ```

4. **File Permission Issues**
   - Ensure the `logs` directory is writable:
     ```bash
     mkdir -p logs
     chmod 755 logs
     ```

### Directory Structure

```
openrouter_chatbot/
├── src/
│   ├── api/              # API client implementations
│   ├── components/       # UI components
│   ├── config/          # Configuration utilities
│   └── utils/           # Utility functions
├── pages/               # Streamlit pages
├── logs/               # Application logs
├── .env                # Environment variables
├── config.yaml         # Application configuration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Configuration

The application uses `config.yaml` for model configurations and routing rules. You can:

- Add/remove models
- Adjust token limits and costs
- Modify routing preferences for different prompt types

### Cost Management

The application tracks cost per request and conversation totals. You can:

- View detailed metrics for each response
- See aggregate spending by model
- Monitor token usage and response latency

### Dashboard

For more detailed cost analytics, you can run the cost dashboard:

```bash
streamlit run cost_dashboard.py
```

## License

[MIT License](LICENSE)
