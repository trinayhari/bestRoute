#!/usr/bin/env python3
"""
Demo for the Advanced Router

This script demonstrates how to use the AdvancedRouter to intelligently
route prompts to different models based on metrics and prompt content.
"""

import os
import json
import time
from typing import Dict, List, Any
import streamlit as st
from src.utils.advanced_router import AdvancedRouter
from src.config.config_loader import load_config

def main():
    st.set_page_config(
        page_title="Advanced LLM Router Demo",
        page_icon="🧠",
        layout="wide",
    )
    
    st.title("Advanced LLM Router Demo")
    st.markdown("""
    This demo shows how the advanced router selects the most appropriate model
    based on the prompt content and historical performance metrics.
    """)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        st.stop()
    
    # Add optimization target to config if not present
    if "optimization_target" not in config:
        config["optimization_target"] = "balanced"
    
    # Initialize the advanced router
    try:
        router = AdvancedRouter(config)
    except Exception as e:
        st.error(f"Error initializing router: {str(e)}")
        if "OPENROUTER_API_KEY" not in os.environ:
            st.warning("OPENROUTER_API_KEY environment variable not set. Please set it and restart the app.")
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.subheader("Settings")
        
        # Optimization target
        optimization_target = st.selectbox(
            "Optimization Target",
            options=["balanced", "speed", "cost", "quality"],
            index=["balanced", "speed", "cost", "quality"].index(config.get("optimization_target", "balanced"))
        )
        
        if optimization_target != router.optimization_target:
            router.optimization_target = optimization_target
            router._init_router_params()
            st.success(f"Optimization target changed to: {optimization_target}")
        
        # Show all available models
        st.subheader("Available Models")
        for model_id, info in router.models.items():
            with st.expander(info.get("name", model_id)):
                st.write(f"**Provider:** {info.get('provider', 'Unknown')}")
                st.write(f"**Cost:** ${info.get('cost_per_1k_tokens', 0):.5f} per 1K tokens")
                st.write(f"**Strengths:** {', '.join(info.get('strengths', []))}")
    
    # Chat interface
    st.subheader("Chat")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {}
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show model info for assistant messages
            if message["role"] == "assistant" and "model" in message:
                with st.expander("Message details"):
                    st.write(f"**Model:** {message.get('model', 'Unknown')}")
                    st.write(f"**Tokens:** {message.get('tokens', 'Unknown')}")
                    st.write(f"**Latency:** {message.get('latency', 'Unknown'):.2f}s")
                    st.write(f"**Cost:** ${message.get('cost', 0):.5f}")
    
    # Chat input
    prompt = st.chat_input("Enter your message here")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Create messages array for API
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        
        # Use the router to select model and get response
        with st.spinner("Thinking..."):
            try:
                # Show which model was selected
                selected_model = router.select_model(prompt)
                model_name = router.models.get(selected_model, {}).get("name", selected_model)
                
                st.info(f"Selected model: {model_name}")
                
                # Get response
                start_time = time.time()
                response_text, usage_stats = router.send_prompt(messages)
                total_time = time.time() - start_time
                
                # Get cost information
                cost_per_1k = router.models.get(selected_model, {}).get("cost_per_1k_tokens", 0)
                total_tokens = usage_stats.get("total_tokens", 0)
                estimated_cost = (total_tokens * cost_per_1k) / 1000
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "model": selected_model,
                    "tokens": total_tokens,
                    "latency": total_time,
                    "cost": estimated_cost
                })
                
                # Update metrics display
                if selected_model not in st.session_state.model_metrics:
                    st.session_state.model_metrics[selected_model] = []
                
                st.session_state.model_metrics[selected_model].append({
                    "tokens": total_tokens,
                    "latency": total_time,
                    "cost": estimated_cost
                })
                
                # Display assistant message
                with st.chat_message("assistant"):
                    st.write(response_text)
                    with st.expander("Message details"):
                        st.write(f"**Model:** {model_name}")
                        st.write(f"**Tokens:** {total_tokens}")
                        st.write(f"**Latency:** {total_time:.2f}s")
                        st.write(f"**Cost:** ${estimated_cost:.5f}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Metrics visualization
    if st.session_state.model_metrics:
        st.subheader("Model Performance Metrics")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### Average Latency")
            for model_id, metrics in st.session_state.model_metrics.items():
                if metrics:
                    avg_latency = sum(m["latency"] for m in metrics) / len(metrics)
                    model_name = router.models.get(model_id, {}).get("name", model_id)
                    st.metric(model_name, f"{avg_latency:.2f}s")
        
        with col2:
            st.write("### Average Tokens")
            for model_id, metrics in st.session_state.model_metrics.items():
                if metrics:
                    avg_tokens = sum(m["tokens"] for m in metrics) / len(metrics)
                    model_name = router.models.get(model_id, {}).get("name", model_id)
                    st.metric(model_name, f"{int(avg_tokens)}")
        
        with col3:
            st.write("### Total Cost")
            for model_id, metrics in st.session_state.model_metrics.items():
                if metrics:
                    total_cost = sum(m["cost"] for m in metrics)
                    model_name = router.models.get(model_id, {}).get("name", model_id)
                    st.metric(model_name, f"${total_cost:.5f}")

if __name__ == "__main__":
    main() 