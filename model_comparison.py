#!/usr/bin/env python3
"""
Model Comparison Tool

A Streamlit application that allows users to compare responses from multiple LLMs side by side.
This helps evaluate different models for specific use cases.
"""

import os
import sys
import yaml
import time
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the rule-based router and other utilities
try:
    from src.utils.rule_based_router import RuleBasedRouter
    from src.utils.cost_tracker import CostTracker
    from src.api.openrouter_client_enhanced import send_prompt_to_openrouter
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Page config
st.set_page_config(
    page_title="LLM Model Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .model-header {
        font-size: 1.1em;
        color: #1E88E5;
        padding: 5px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    .model-response {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 3px solid #7E57C2;
        font-size: 0.95em;
        line-height: 1.5;
    }
    .metrics-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
        border-left: 3px solid #4CAF50;
        font-size: 0.85em;
    }
    .model-tag {
        display: inline-block;
        background-color: #e0e0e0;
        color: #333;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .provider-tag {
        display: inline-block;
        background-color: #bbdefb;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .token-usage {
        display: inline-block;
        margin-right: 10px;
    }
    .cost {
        display: inline-block;
        margin-right: 10px;
        color: #4CAF50;
    }
    .latency {
        display: inline-block;
        color: #FF9800;
    }
    .diff-highlight-add {
        background-color: #e6ffed;
        border-left: 2px solid #28a745;
        padding-left: 5px;
    }
    .diff-highlight-remove {
        background-color: #ffeef0;
        border-left: 2px solid #d73a49;
        padding-left: 5px;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
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

@st.cache_resource
def initialize_router(config):
    """Initialize the router with the configuration"""
    try:
        router = RuleBasedRouter(config)
        return router
    except Exception as e:
        st.error(f"Error initializing router: {e}")
        return None

def initialize_session_state():
    """Initialize session state variables"""
    if "comparison_history" not in st.session_state:
        st.session_state.comparison_history = []
    
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    
    if "model_responses" not in st.session_state:
        st.session_state.model_responses = {}
    
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "You are a helpful AI assistant."
    
    if "comparison_view" not in st.session_state:
        st.session_state.comparison_view = "side_by_side"  # or "tabbed"
        
    if "highlight_differences" not in st.session_state:
        st.session_state.highlight_differences = False
    
    if "compare_running" not in st.session_state:
        st.session_state.compare_running = False

def get_model_response(model_id, prompt, system_prompt="You are a helpful AI assistant.", router=None):
    """Get response from a specific model"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    start_time = time.time()
    
    try:
        # Get model-specific parameters from config
        config = load_config()
        model_info = config.get("models", {}).get(model_id, {})
        temperature = model_info.get("temperature", 0.7)
        max_tokens = min(model_info.get("max_tokens", 1000), 4000)  # Cap at 4000 to prevent errors
        
        # Send request directly to OpenRouter
        response_text, usage_stats, latency = send_prompt_to_openrouter(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        total_time = time.time() - start_time
        
        # Calculate cost
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
        cost = (usage_stats.get("total_tokens", 0) * cost_per_1k) / 1000
        
        return {
            "model": model_id,
            "response": response_text,
            "usage_stats": usage_stats,
            "latency": latency,
            "total_time": total_time,
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
    except Exception as e:
        return {
            "model": model_id,
            "response": f"Error: {str(e)}",
            "error": str(e),
            "total_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "success": False
        }

def display_model_header(model_id, config):
    """Display model header with info"""
    model_info = config.get("models", {}).get(model_id, {})
    provider = model_info.get("provider", "Unknown")
    
    header_html = f"""
    <div class="model-header">
        <span class="model-tag">{model_info.get('name', model_id)}</span>
        <span class="provider-tag">{provider}</span>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)

def display_model_response(response_data):
    """Display a model's response with metrics"""
    if not response_data:
        st.markdown("*No response yet*")
        return
    
    # Check if there was an error
    if not response_data.get("success", False):
        st.error(f"Error: {response_data.get('error', 'Unknown error')}")
        return
    
    # Display the response
    st.markdown(f'<div class="model-response">{response_data["response"]}</div>', unsafe_allow_html=True)
    
    # Display metrics if available
    if "usage_stats" in response_data:
        prompt_tokens = response_data.get("usage_stats", {}).get("prompt_tokens", 0)
        completion_tokens = response_data.get("usage_stats", {}).get("completion_tokens", 0)
        total_tokens = response_data.get("usage_stats", {}).get("total_tokens", 0)
        latency = response_data.get("latency", 0)
        cost = response_data.get("cost", 0)
        
        metrics_html = f"""
        <div class="metrics-container">
            <div class="token-usage">Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens}</div>
            <div class="cost">Cost: ${cost:.6f}</div>
            <div class="latency">Latency: {latency:.2f}s</div>
        </div>
        """
        
        st.markdown(metrics_html, unsafe_allow_html=True)

def compare_responses():
    """Compare responses from selected models"""
    if not st.session_state.current_prompt:
        st.warning("Please enter a prompt before comparing models.")
        return
    
    if not st.session_state.selected_models:
        st.warning("Please select at least one model to compare.")
        return
    
    st.session_state.compare_running = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous responses
    st.session_state.model_responses = {}
    
    # Get responses from each model
    for i, model_id in enumerate(st.session_state.selected_models):
        status_text.text(f"Getting response from {model_id}...")
        
        response_data = get_model_response(
            model_id=model_id,
            prompt=st.session_state.current_prompt,
            system_prompt=st.session_state.system_prompt
        )
        
        st.session_state.model_responses[model_id] = response_data
        progress_bar.progress((i + 1) / len(st.session_state.selected_models))
    
    # Save comparison to history
    st.session_state.comparison_history.append({
        "prompt": st.session_state.current_prompt,
        "system_prompt": st.session_state.system_prompt,
        "models": st.session_state.selected_models,
        "responses": st.session_state.model_responses,
        "timestamp": datetime.now().isoformat()
    })
    
    status_text.text("Comparison complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    st.session_state.compare_running = False

def display_side_by_side_comparison():
    """Display models side by side"""
    if not st.session_state.model_responses:
        return
    
    num_models = len(st.session_state.selected_models)
    if num_models == 0:
        return
    
    # Create columns
    cols = st.columns(min(num_models, 3))
    
    # Get config for model info
    config = load_config()
    
    # Display responses in columns
    for i, model_id in enumerate(st.session_state.selected_models):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            display_model_header(model_id, config)
            display_model_response(st.session_state.model_responses.get(model_id, {}))

def display_tabbed_comparison():
    """Display models in tabs"""
    if not st.session_state.model_responses:
        return
    
    num_models = len(st.session_state.selected_models)
    if num_models == 0:
        return
    
    # Get config for model info
    config = load_config()
    
    # Create tabs
    tabs = st.tabs(st.session_state.selected_models)
    
    # Display responses in tabs
    for i, model_id in enumerate(st.session_state.selected_models):
        with tabs[i]:
            display_model_header(model_id, config)
            display_model_response(st.session_state.model_responses.get(model_id, {}))

def calculate_comparison_metrics():
    """Calculate comparison metrics between models"""
    if not st.session_state.model_responses or len(st.session_state.selected_models) < 2:
        return None
    
    metrics = []
    
    for model_id in st.session_state.selected_models:
        response_data = st.session_state.model_responses.get(model_id, {})
        
        if not response_data or not response_data.get("success", False):
            continue
        
        metrics.append({
            "Model": model_id,
            "Response Length (chars)": len(response_data.get("response", "")),
            "Total Tokens": response_data.get("usage_stats", {}).get("total_tokens", 0),
            "Latency (s)": response_data.get("latency", 0),
            "Cost ($)": response_data.get("cost", 0),
        })
    
    return pd.DataFrame(metrics)

def display_comparison_metrics():
    """Display metrics comparing the models"""
    metrics_df = calculate_comparison_metrics()
    
    if metrics_df is None or len(metrics_df) < 2:
        return
    
    st.markdown("### Comparison Metrics")
    
    # Display metrics table
    st.dataframe(metrics_df)
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Token usage
        fig1 = px.bar(
            metrics_df, 
            x="Model", 
            y="Total Tokens", 
            title="Token Usage by Model",
            color="Model"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cost comparison
        fig2 = px.bar(
            metrics_df, 
            x="Model", 
            y="Cost ($)", 
            title="Cost Comparison",
            color="Model"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Latency comparison
    fig3 = px.bar(
        metrics_df, 
        x="Model", 
        y="Latency (s)", 
        title="Response Time Comparison",
        color="Model"
    )
    st.plotly_chart(fig3, use_container_width=True)

def save_comparison_to_file():
    """Save the current comparison to a JSON file"""
    if not st.session_state.comparison_history:
        st.warning("No comparisons to save.")
        return
    
    # Create exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)
    
    # Generate filename based on timestamp
    filename = f"exports/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(st.session_state.comparison_history, f, indent=2)
    
    st.success(f"Comparison saved to {filename}")

def display_sidebar(config, router):
    """Display and handle sidebar elements"""
    st.sidebar.title("Model Comparison")
    
    # API Key status
    if "OPENROUTER_API_KEY" in os.environ:
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.error("❌ API Key not found")
        st.sidebar.info("Set OPENROUTER_API_KEY in your environment or .env file")
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.markdown("### Select Models to Compare")
    
    # Get available models from config
    available_models = sorted(config.get("models", {}).keys())
    
    # Group models by provider
    models_by_provider = {}
    for model_id in available_models:
        model_info = config.get("models", {}).get(model_id, {})
        provider = model_info.get("provider", "Other")
        
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        
        models_by_provider[provider].append(model_id)
    
    # Select models by provider
    for provider, models in models_by_provider.items():
        st.sidebar.markdown(f"**{provider}**")
        for model_id in models:
            model_name = config.get("models", {}).get(model_id, {}).get("name", model_id)
            
            if st.sidebar.checkbox(model_name, model_id in st.session_state.selected_models, key=f"model_{model_id}"):
                if model_id not in st.session_state.selected_models:
                    st.session_state.selected_models.append(model_id)
            else:
                if model_id in st.session_state.selected_models:
                    st.session_state.selected_models.remove(model_id)
    
    # Quick model selection
    st.sidebar.markdown("### Quick Selection")
    
    if st.sidebar.button("Select All"):
        st.session_state.selected_models = available_models.copy()
        st.rerun()
    
    if st.sidebar.button("Clear Selection"):
        st.session_state.selected_models = []
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Comparison settings
    st.sidebar.markdown("### Comparison Settings")
    
    # System prompt
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100
    )
    
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
    
    # View type
    view_options = {
        "side_by_side": "Side by Side",
        "tabbed": "Tabbed View"
    }
    
    selected_view = st.sidebar.radio(
        "Display View",
        options=list(view_options.keys()),
        format_func=lambda x: view_options[x],
        index=0 if st.session_state.comparison_view == "side_by_side" else 1
    )
    
    if selected_view != st.session_state.comparison_view:
        st.session_state.comparison_view = selected_view
    
    # Highlight differences
    st.session_state.highlight_differences = st.sidebar.checkbox(
        "Highlight Differences (Experimental)",
        value=st.session_state.highlight_differences
    )
    
    st.sidebar.markdown("---")
    
    # Export options
    st.sidebar.markdown("### Export Options")
    
    if st.sidebar.button("Save Comparisons"):
        save_comparison_to_file()
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool allows you to compare responses from multiple language models "
        "side by side. Use it to evaluate which model performs best for your specific use case."
    )

def display_prompt_area():
    """Display the prompt input area"""
    st.markdown("### Enter your prompt")
    
    prompt = st.text_area(
        "Prompt",
        value=st.session_state.current_prompt,
        height=150,
        placeholder="Enter your prompt here..."
    )
    
    if prompt != st.session_state.current_prompt:
        st.session_state.current_prompt = prompt
    
    # Compare button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Compare Models", disabled=st.session_state.compare_running):
            compare_responses()

def display_comparison_results():
    """Display the comparison results"""
    if not st.session_state.model_responses:
        return
    
    st.markdown("---")
    st.markdown("## Comparison Results")
    
    # Display based on selected view
    if st.session_state.comparison_view == "side_by_side":
        display_side_by_side_comparison()
    else:
        display_tabbed_comparison()
    
    # Display metrics
    st.markdown("---")
    display_comparison_metrics()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Initialize router (for helpers and cost calculation)
    router = initialize_router(config)
    
    if router is None:
        st.error("Failed to initialize the router. Please check your configuration.")
        return
    
    # Display sidebar
    display_sidebar(config, router)
    
    # Main area
    st.title("LLM Model Comparison")
    
    st.markdown(
        "Compare responses from multiple language models side by side. "
        "Select the models you want to compare in the sidebar, enter your prompt below, "
        "and click 'Compare Models' to see how each model responds."
    )
    
    # Display prompt input area
    display_prompt_area()
    
    # Display comparison results
    display_comparison_results()

if __name__ == "__main__":
    main() 