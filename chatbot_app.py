#!/usr/bin/env python3
"""
OpenRouter Chatbot App

A Streamlit application that provides a chat interface with rule-based routing
to select the most appropriate model based on prompt type and length.
"""

import os
import sys
import yaml
import time
import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the rule-based router and other utilities
try:
    from src.utils.rule_based_router import RuleBasedRouter
    from src.utils.cost_tracker import CostTracker
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Page config
st.set_page_config(
    page_title="OpenRouter LLM Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .model-tag {
        font-size: 0.8em;
        color: #888;
        margin-bottom: 0px;
    }
    .metrics-container {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #4CAF50;
    }
    .chat-container {
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #1E88E5;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #7E57C2;
    }
    .small-text {
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
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "metrics" not in st.session_state:
        st.session_state.metrics = []
    
    if "conversation_cost" not in st.session_state:
        st.session_state.conversation_cost = 0.0
    
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = True
    
    if "manual_model_selection" not in st.session_state:
        st.session_state.manual_model_selection = False
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

def format_message(message, model=None, metrics=None):
    """Format a message with model name and metrics"""
    if message["role"] == "user":
        return st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        # For assistant messages, add model name and metrics
        model_display = f"<div class='model-tag'>Model: {model}</div>" if model else ""
        
        message_html = f"<div class='assistant-message'>{model_display}{message['content']}</div>"
        
        # Add metrics if available
        if metrics and st.session_state.show_metrics:
            prompt_tokens = metrics.get("usage_stats", {}).get("prompt_tokens", 0)
            completion_tokens = metrics.get("usage_stats", {}).get("completion_tokens", 0)
            total_tokens = metrics.get("usage_stats", {}).get("total_tokens", 0)
            latency = metrics.get("latency", 0)
            cost = metrics.get("cost", 0)
            
            metrics_html = f"""
            <div class='metrics-container small-text'>
                <div class='token-usage'>Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {total_tokens}</div>
                <div class='cost'>Cost: ${cost:.6f}</div>
                <div class='latency'>Latency: {latency:.2f}s</div>
            </div>
            """
            message_html = metrics_html + message_html
        
        return st.markdown(message_html, unsafe_allow_html=True)

def display_chat_messages():
    """Display all chat messages with formatting"""
    if not st.session_state.messages:
        st.markdown("### Start a conversation with the AI assistant")
        return
    
    for i, message in enumerate(st.session_state.messages):
        # Skip system messages in the display
        if message["role"] == "system":
            continue
        
        # Get metrics for assistant messages
        metrics = None
        model = None
        if message["role"] == "assistant" and i // 2 < len(st.session_state.metrics):
            metrics = st.session_state.metrics[i // 2]
            model = metrics.get("model", "Unknown")
        
        # Display the message with metrics
        format_message(message, model, metrics)

def display_cost_summary():
    """Display a summary of the conversation cost and token usage"""
    if not st.session_state.metrics:
        return
    
    # Calculate total metrics
    total_prompt_tokens = sum(m.get("usage_stats", {}).get("prompt_tokens", 0) for m in st.session_state.metrics)
    total_completion_tokens = sum(m.get("usage_stats", {}).get("completion_tokens", 0) for m in st.session_state.metrics)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_cost = st.session_state.conversation_cost
    
    # Generate model usage stats
    model_stats = {}
    for metric in st.session_state.metrics:
        model = metric.get("model", "Unknown")
        tokens = metric.get("usage_stats", {}).get("total_tokens", 0)
        cost = metric.get("cost", 0)
        
        if model not in model_stats:
            model_stats[model] = {"tokens": 0, "cost": 0.0, "count": 0}
        
        model_stats[model]["tokens"] += tokens
        model_stats[model]["cost"] += cost
        model_stats[model]["count"] += 1
    
    # Create a DataFrame for metrics display
    metrics_df = pd.DataFrame([
        {"Metric": "Total Messages", "Value": len(st.session_state.metrics)},
        {"Metric": "Total Tokens", "Value": total_tokens},
        {"Metric": "Prompt Tokens", "Value": total_prompt_tokens},
        {"Metric": "Completion Tokens", "Value": total_completion_tokens},
        {"Metric": "Total Cost", "Value": f"${total_cost:.6f}"}
    ])
    
    # Create a DataFrame for model usage
    model_df = pd.DataFrame([
        {"Model": model, "Messages": stats["count"], "Tokens": stats["tokens"], "Cost": f"${stats['cost']:.6f}"}
        for model, stats in model_stats.items()
    ])
    
    # Display the summary
    st.markdown("### Conversation Summary")
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        if len(model_stats) > 0:
            # Create data for pie chart
            pie_data = pd.DataFrame([
                {"Model": model, "Tokens": stats["tokens"]}
                for model, stats in model_stats.items()
            ])
            
            fig = px.pie(pie_data, values="Tokens", names="Model", title="Token Usage by Model")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display model usage
    st.markdown("### Model Usage")
    st.dataframe(model_df, hide_index=True)

def display_sidebar(config, router):
    """Display and handle sidebar elements"""
    st.sidebar.title("OpenRouter Chatbot")
    
    # API Key status
    if "OPENROUTER_API_KEY" in os.environ:
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.error("❌ API Key not found")
        st.sidebar.info("Set OPENROUTER_API_KEY in your environment or .env file")
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    
    # Toggle for manual model selection
    st.session_state.manual_model_selection = st.sidebar.checkbox(
        "Manual Model Selection", 
        value=st.session_state.manual_model_selection
    )
    
    if st.session_state.manual_model_selection:
        # Get available models from config
        available_models = sorted(config.get("models", {}).keys())
        
        if available_models:
            # Set default to the first model if none selected
            if not st.session_state.selected_model:
                st.session_state.selected_model = available_models[0]
                
            # Model dropdown
            st.session_state.selected_model = st.sidebar.selectbox(
                "Select Model", 
                available_models,
                index=available_models.index(st.session_state.selected_model)
            )
            
            # Show model info
            model_info = config.get("models", {}).get(st.session_state.selected_model, {})
            if model_info:
                st.sidebar.markdown(f"**Provider:** {model_info.get('provider', 'Unknown')}")
                st.sidebar.markdown(f"**Cost:** ${model_info.get('cost_per_1k_tokens', 0):.6f} per 1K tokens")
                st.sidebar.markdown(f"**Context Length:** {model_info.get('context_length', 'Unknown')}")
                
                strengths = model_info.get("strengths", [])
                if strengths:
                    st.sidebar.markdown("**Strengths:**")
                    for s in strengths:
                        st.sidebar.markdown(f"- {s}")
    else:
        st.sidebar.markdown("Using rule-based model selection")
        st.sidebar.markdown("Model will be selected based on prompt type and length:")
        st.sidebar.markdown("- **Code**: For programming and technical questions")
        st.sidebar.markdown("- **Summary**: For summarizing or condensing information")
        st.sidebar.markdown("- **Question**: For general questions and inquiries")
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.markdown("### Settings")
    
    # Toggle for showing metrics
    st.session_state.show_metrics = st.sidebar.checkbox(
        "Show Response Metrics", 
        value=st.session_state.show_metrics
    )
    
    # System prompt
    st.sidebar.markdown("### System Prompt")
    
    # Default system prompt
    default_system_prompt = "You are a helpful AI assistant."
    
    # Get the system prompt from session state or set the default
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = default_system_prompt
    
    # System prompt input
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100
    )
    
    # Update system prompt in session state
    st.session_state.system_prompt = system_prompt
    
    # Reset button for chat
    if st.sidebar.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.metrics = []
        st.session_state.conversation_cost = 0.0
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This chatbot uses rule-based routing to select the most appropriate model "
        "based on prompt type and length. It leverages multiple AI models through "
        "the OpenRouter API."
    )

def process_user_input(router):
    """Process user input and get model response"""
    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Prepare complete message history including system prompt
        full_messages = [
            {"role": "system", "content": st.session_state.system_prompt}
        ] + st.session_state.messages
        
        # Display "thinking" spinner while processing
        with st.spinner("Thinking..."):
            try:
                # Use selected model if manual selection is enabled
                model_override = None
                if st.session_state.manual_model_selection and st.session_state.selected_model:
                    model_override = st.session_state.selected_model
                
                # Get response from router
                start_time = time.time()
                response_text, metrics = router.send_prompt(full_messages, model_id=model_override)
                total_time = time.time() - start_time
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Calculate and add cost
                model = metrics.get("model", "unknown")
                tokens = metrics.get("token_count", 0)
                cost = router.calculate_cost(tokens, model)
                
                # Add cost to metrics
                metrics["cost"] = cost
                
                # Update total conversation cost
                st.session_state.conversation_cost += cost
                
                # Store metrics
                st.session_state.metrics.append(metrics)
                
                # Rerun to update the UI
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
                # Log error message for debugging
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"}
                )
                
                # Rerun to update the UI
                st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Load configuration
    config = load_config()
    
    # Initialize router
    router = initialize_router(config)
    
    if router is None:
        st.error("Failed to initialize the router. Please check your configuration.")
        return
    
    # Display sidebar
    display_sidebar(config, router)
    
    # Main area
    st.title("OpenRouter Chatbot")
    
    # Display chat messages
    display_chat_messages()
    
    # Process user input
    process_user_input(router)
    
    # Display cost and usage summary
    if st.session_state.metrics:
        st.markdown("---")
        display_cost_summary()

if __name__ == "__main__":
    main() 