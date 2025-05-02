# OpenRouter Chatbot

A Streamlit-based chatbot application that uses rule-based routing to select the most appropriate AI model based on prompt type and length.

## Features

- **Rule-Based Routing**: Automatically selects the best model based on whether your prompt is code-related, a summary request, or a general question
- **Multiple Models**: Supports various models from OpenAI, Anthropic, and Mistral AI through OpenRouter
- **Cost Tracking**: Displays cost, token usage, and latency for each response
- **Usage Analytics**: Shows conversation summary with costs, token usage, and model distribution
- **Manual Override**: Option to manually select a specific model instead of using automatic routing

## Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (get one at [OpenRouter](https://openrouter.ai))

### Installation

1. Clone this repository:

   ```
   git clone <repository-url>
   cd openrouter_chatbot
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your API key:

   Either:

   - Create a `.env` file in the project root with:
     ```
     OPENROUTER_API_KEY=your-api-key-here
     ```

   Or:

   - Export the API key as an environment variable:
     ```
     export OPENROUTER_API_KEY=your-api-key-here
     ```

### Running the Application

Run the Streamlit application:

```
streamlit run unified_app.py
```

Or, for the multi-page application:

```
streamlit run Home.py
```

The application will be available at [http://localhost:8501](http://localhost:8501).

## Configuration

The application uses the `config.yaml` file for model configurations and routing rules. You can:

- Add/remove models
- Adjust token limits and costs
- Modify routing preferences for different prompt types

## Cost Management

The application tracks cost per request and conversation totals. You can:

- View detailed metrics for each response
- See aggregate spending by model
- Monitor token usage and response latency

## Dashboard

For more detailed cost analytics, you can run the cost dashboard:

```
streamlit run cost_dashboard.py
```

## Troubleshooting

If you encounter issues:

1. Check your API key is set correctly
2. Ensure all dependencies are installed
3. Verify your `.env` file is in the correct location
4. Check the logs in the `logs` directory

## License

[MIT License](LICENSE)
