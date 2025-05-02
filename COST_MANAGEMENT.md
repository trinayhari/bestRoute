# Cost Management Tools for OpenRouter

This document describes the tools available for managing and estimating costs when using OpenRouter's API with multiple LLMs.

## Cost Estimation

Understanding and managing API costs is crucial for any application using LLMs. The tools in this repository help you:

- Estimate costs before sending requests
- Track actual usage and costs
- Analyze cost trends over time
- Compare costs across different models

## Available Tools

### 1. Cost Estimator (`estimate_cost.py`)

A command-line utility for estimating costs of using different models.

**Features:**

- Token count estimation for different model families
- Cost calculation based on your config.yaml pricing
- Comparison of costs across all available models
- Support for both CLI arguments and interactive mode

**Usage Examples:**

Basic usage:

```bash
python estimate_cost.py --prompt "Your prompt text here"
```

Estimate for a specific model:

```bash
python estimate_cost.py --prompt "Your prompt text here" --model anthropic/claude-3-haiku
```

List all available models and their pricing:

```bash
python estimate_cost.py --list-models
```

Using a file as input:

```bash
python estimate_cost.py --file your_prompt.txt --system "You are a helpful assistant."
```

### 2. Cost Tracker (`src/utils/cost_tracker.py`)

A Python module for tracking and analyzing API usage costs.

**Features:**

- Log API calls with token usage and cost information
- Generate summaries by session or by day
- Track cost trends over time
- Export comprehensive cost reports

**Integration Example:**

```python
from src.utils.cost_tracker import CostTracker
from src.config.config_loader import load_config

# Initialize
config = load_config()
cost_tracker = CostTracker(config)

# Log API call
cost_tracker.log_api_call(
    model="anthropic/claude-3-haiku",
    usage_stats={
        "prompt_tokens": 100,
        "completion_tokens": 150,
        "total_tokens": 250
    }
)

# Get session summary
session_summary = cost_tracker.get_session_summary()
print(f"Session cost: ${session_summary['total_cost']:.5f}")

# Export report
report_path = cost_tracker.export_cost_report()
print(f"Cost report exported to: {report_path}")
```

### 3. Cost Dashboard (`cost_dashboard.py`)

A Streamlit-based visual dashboard for analyzing costs.

**Features:**

- Real-time overview of costs and usage
- Daily breakdown of API usage
- Model comparison tools
- Interactive cost estimation
- Graphical visualizations
- Raw data export

**Usage:**

```bash
streamlit run cost_dashboard.py
```

## Model Pricing Configuration

All tools use the pricing information defined in `config.yaml`. Each model entry includes a `cost_per_1k_tokens` field:

```yaml
models:
  anthropic/claude-3-haiku:
    name: Claude 3 Haiku
    cost_per_1k_tokens: 0.00025
    max_tokens: 200000

  anthropic/claude-3-opus:
    name: Claude 3 Opus
    cost_per_1k_tokens: 0.015
    max_tokens: 200000

  # ... other models
```

Ensure this pricing information is kept up-to-date with OpenRouter's current pricing.

## Best Practices for Cost Management

1. **Estimate Before Using**: Use the cost estimator to understand potential costs before sending expensive queries.

2. **Monitor Usage**: Regularly check the cost dashboard to track spending.

3. **Choose Wisely**: Select the most appropriate model for each task - don't use Opus for simple tasks when Haiku would suffice.

4. **Optimize Prompts**: More concise prompts mean fewer tokens and lower costs.

5. **Set Budgets**: Define monthly or per-project budgets and track them with the cost dashboard.

6. **Watch for Outliers**: The dashboard helps identify unusually expensive requests that might indicate issues.

7. **Export Reports**: Generate cost reports for accounting and budget planning.

## Additional Resources

- [OpenRouter Pricing](https://openrouter.ai/docs/pricing)
- API key management: See [API_KEY_TROUBLESHOOTING.md](API_KEY_TROUBLESHOOTING.md)
