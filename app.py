import streamlit as st
import yaml
import os
from router import ModelRouter
from src.components.chat_ui import ChatUI
from src.config.config_loader import load_config

def main():
    # Set up page configuration
    st.set_page_config(
        page_title="Multi-LLM Chat with OpenRouter",
        page_icon="🤖",
        layout="wide",
    )
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Header section
    st.title("Multi-LLM Chat with OpenRouter")
    st.markdown("""
    This app intelligently routes your prompts to the best LLM for the task at hand.
    Different prompt types are automatically sent to the most appropriate model.
    """)
    
    # Load model configurations from YAML
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()
    
    # Initialize the model router
    try:
        router = ModelRouter(config)
    except Exception as e:
        st.error(f"Error initializing model router: {str(e)}")
        if "OPENROUTER_API_KEY" not in os.environ:
            st.warning("OPENROUTER_API_KEY environment variable not set. Please set it and restart the app.")
        st.stop()
    
    # Initialize and display the chat UI
    chat_ui = ChatUI(router)
    chat_ui.render()

if __name__ == "__main__":
    main() 