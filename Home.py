#!/usr/bin/env python3
"""
OpenRouter LLM Suite - Home

A unified interface for working with multiple LLMs through the OpenRouter API.
"""

import os
import sys
import yaml
import streamlit as st
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import common utilities
try:
    from src.utils.rule_based_router import RuleBasedRouter
    from src.utils.cost_tracker import CostTracker
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Page config
st.set_page_config(
    page_title="OpenRouter LLM Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .main-container {
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4169E1;
    }
    .feature-header {
        color: #4169E1;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        color: #555;
        font-size: 0.9rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

def check_api_key():
    """Check if the API key is configured"""
    if "OPENROUTER_API_KEY" in os.environ:
        return True
    else:
        return False

def display_sidebar(config):
    """Display sidebar elements"""
    st.sidebar.title("OpenRouter LLM Suite")
    
    # API Key status
    if check_api_key():
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.error("❌ API Key not found")
        st.sidebar.info("Set OPENROUTER_API_KEY in your environment or .env file")
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.markdown("### Quick Navigation")
    st.sidebar.markdown("Use the menu in the sidebar to navigate between:")
    st.sidebar.markdown("- **Home**: This overview page")
    st.sidebar.markdown("- **Chatbot**: Interact with AI models")
    st.sidebar.markdown("- **Model Comparison**: Compare model responses")
    st.sidebar.markdown("- **Cost Dashboard**: Monitor API usage costs")
    
    st.sidebar.markdown("---")
    
    # Show available models
    st.sidebar.markdown("### Available Models")
    models = config.get("models", {})
    providers = {}
    
    # Group models by provider
    for model_id, model_info in models.items():
        provider = model_info.get("provider", "Unknown")
        if provider not in providers:
            providers[provider] = []
        providers[provider].append({
            "id": model_id,
            "name": model_info.get("name", model_id),
            "cost": model_info.get("cost_per_1k_tokens", 0)
        })
    
    # Show providers and model counts
    for provider, provider_models in providers.items():
        st.sidebar.markdown(f"**{provider}** ({len(provider_models)} models)")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")

def main():
    """Main application function"""
    # Load configuration
    config = load_config()
    
    # Display sidebar
    display_sidebar(config)
    
    # Main area
    st.title("🤖 OpenRouter LLM Suite")
    st.markdown(
        "A unified interface for working with multiple Large Language Models through the OpenRouter API. "
        "This application provides tools for chatting with models, comparing their responses, and monitoring usage costs."
    )
    
    # API Key Warning
    if not check_api_key():
        st.warning(
            "⚠️ OpenRouter API Key not found. Please set the OPENROUTER_API_KEY environment variable or add it to your .env file. "
            "Some features may not work properly without a valid API key."
        )
    
    # Main features
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Chatbot feature card
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-header">💬 Advanced Chatbot</h3>
                <p class="feature-description">
                    Interact with multiple LLMs through a chat interface with automatic model routing
                    based on prompt type and content. Choose between automatic routing or manual model selection.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Cost dashboard feature card
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-header">📊 Cost Dashboard</h3>
                <p class="feature-description">
                    Monitor your API usage costs with detailed analytics and visualizations.
                    Track daily usage, compare model costs, and export usage data for analysis.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        # Model comparison feature card
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-header">🔍 Model Comparison</h3>
                <p class="feature-description">
                    Compare responses from multiple models side by side. Analyze differences
                    in response quality, token usage, latency, and cost per request.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Cost estimation feature card
        st.markdown(
            """
            <div class="feature-card">
                <h3 class="feature-header">💰 Cost Estimation</h3>
                <p class="feature-description">
                    Estimate API usage costs for different models and token counts.
                    Plan your usage and choose the most cost-effective models for your needs.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Getting started
    st.markdown("## Getting Started")
    st.markdown(
        """
        1. **Set up your API key**: Make sure your OpenRouter API key is configured in your environment
        2. **Navigate to the app**: Use the sidebar to switch between different features
        3. **Chat with models**: Start using the chatbot to interact with AI models
        4. **Compare models**: Use the model comparison to evaluate which model works best for your needs
        5. **Monitor costs**: Keep track of your API usage in the cost dashboard
        """
    )
    
    # Available Models
    st.markdown("## Available Models")
    
    # Get model information from config
    models = config.get("models", {})
    if models:
        # Create model data for table
        model_data = []
        
        for model_id, model_info in models.items():
            model_data.append({
                "Model": model_info.get("name", model_id),
                "Provider": model_info.get("provider", "Unknown"),
                "Cost (per 1K tokens)": f"${model_info.get('cost_per_1k_tokens', 0):.6f}",
                "Max Tokens": f"{model_info.get('max_tokens', 0):,}",
                "Context Window": f"{model_info.get('context_length', 'Unknown')}"
            })
        
        # Display as table with filters
        st.dataframe(model_data, use_container_width=True)
    else:
        st.info("No model information available. Please check your config.yaml file.")
    
    # Footer
    st.markdown("---")
    st.caption("OpenRouter LLM Suite | Made with Streamlit")

if __name__ == "__main__":
    main() 