#!/usr/bin/env python3
"""
Unified OpenRouter Application

A Streamlit application that combines the chatbot, model comparison tool,
and cost dashboard into a single unified interface.
"""

import os
import sys
import yaml
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the required modules
try:
    from src.utils.rule_based_router import RuleBasedRouter
    from src.utils.cost_tracker import CostTracker
    from src.api.openrouter_client_enhanced import send_prompt_to_openrouter
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Import functionality from existing applications
import model_comparison
import cost_dashboard
from chatbot_app import (
    load_config as load_chatbot_config,
    initialize_router,
    initialize_session_state as initialize_chatbot_state,
    display_chat_messages,
    process_user_input,
    display_cost_summary
)

# Page config
st.set_page_config(
    page_title="OpenRouter LLM Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load all styles from individual apps
st.markdown("""
<style>
    /* Chatbot styles */
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

    /* Model comparison styles */
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
    .provider-tag {
        display: inline-block;
        background-color: #bbdefb;
        color: #0d47a1;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
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

    /* Unify the sidebar for all tabs */
    .sidebar-shared {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }

    /* Make the tabs more prominent */
    .main-tabs {
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_unified_config():
    """Load configuration from config.yaml for all components"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

def initialize_app_state():
    """Initialize all session state variables needed across apps"""
    # Initialize chatbot state
    initialize_chatbot_state()
    # Initialize model comparison state
    model_comparison.initialize_session_state()
    # Set current app if not set
    if "current_app" not in st.session_state:
        st.session_state.current_app = "chatbot"

def display_shared_sidebar(config, router):
    """Display shared sidebar elements for all apps"""
    st.sidebar.title("OpenRouter LLM Suite")
    
    # API Key status
    if "OPENROUTER_API_KEY" in os.environ:
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.error("❌ API Key not found")
        st.sidebar.info("Set OPENROUTER_API_KEY in your environment or .env file")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("Use the tabs above to switch between applications:")
    st.sidebar.markdown("- **Chatbot**: Interact with AI models")
    st.sidebar.markdown("- **Model Comparison**: Compare responses from multiple models")
    st.sidebar.markdown("- **Cost Dashboard**: Monitor your API usage and costs")
    
    st.sidebar.markdown("---")
    
    # App-specific sidebar content based on current tab
    if st.session_state.current_app == "chatbot":
        st.sidebar.markdown("### Chatbot Settings")
        st.sidebar.markdown("(Settings can be added here)")
    if st.session_state.current_app == "model_comparison":
        # Show model comparison sidebar using function from model_comparison.py
        model_comparison.display_sidebar(config, router)
    
    elif st.session_state.current_app == "cost_dashboard":
        # No sidebar needed for cost dashboard as it's all in the main view
        st.sidebar.markdown("### Cost Dashboard")
        st.sidebar.info("The cost dashboard shows analytics for your OpenRouter API usage.")
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This application combines a chatbot, model comparison tool, and "
        "cost dashboard into a single integrated interface. It leverages multiple "
        "AI models through the OpenRouter API."
    )

def on_tab_change():
    """Handle tab change actions"""
    # Store current app in session state based on selected tab
    tab_index_to_app = {
        0: "chatbot",
        1: "model_comparison",
        2: "cost_dashboard"
    }
    
    for i in range(3):
        if st.session_state.get(f"tab_{i}", False):
            st.session_state.current_app = tab_index_to_app[i]
            break

def main():
    """Main application function"""
    # Initialize session state
    initialize_app_state()
    
    # Load configuration
    config = load_unified_config()
    
    # Initialize router
    router = initialize_router(config)
    
    tab1, tab2, tab3 = st.tabs([
        "💬 Chatbot",
        "🔍 Model Comparison", 
        "📊 Cost Dashboard"
    ])
    
    # Update current app based on selected tab
    for i, tab in enumerate([tab1, tab2, tab3]):
        if tab.checkbox("", key=f"tab_{i}", value=st.session_state.current_app == ["chatbot", "model_comparison", "cost_dashboard"][i], label_visibility="collapsed"):
            on_tab_change()
    
    # Display sidebar
    display_shared_sidebar(config, router)
    
    # Render appropriate app based on selected tab
    with tab1:
        if st.session_state.current_app == "chatbot":
            st.title("OpenRouter Chatbot")
            display_chat_messages()
            process_user_input(router)
            if st.session_state.metrics:
                st.markdown("---")
                display_cost_summary()
    
    with tab2:
        if st.session_state.current_app == "model_comparison":
            model_comparison.main()
    
    with tab3:
        if st.session_state.current_app == "cost_dashboard":
            cost_dashboard.main()

if __name__ == "__main__":
    main() 