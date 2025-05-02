#!/usr/bin/env python3
"""
OpenRouter LLM Suite - Chatbot

A chat interface with rule-based routing to select the most appropriate model based on prompt type and length.
"""

import os
import sys
import yaml
import time
import streamlit as st
from datetime import datetime
import copy

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    page_title="OpenRouter LLM Suite - Chatbot",
    page_icon="💬",
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
        position: relative;
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
    .rerun-button {
        position: absolute;
        top: 5px;
        right: 5px;
        font-size: 0.8em;
        padding: 2px 8px;
        background-color: #f0f0f0;
        border-radius: 12px;
        color: #555;
        border: 1px solid #ddd;
        cursor: pointer;
    }
    .rerun-button:hover {
        background-color: #e0e0e0;
    }
    .rerun-message {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #FFC107;
    }
    .rerun-tag {
        display: inline-block;
        background-color: #fff0c2;
        color: #9a7b00;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 0.7em;
        margin-right: 5px;
    }
    .model-dropdown {
        margin-bottom: 10px;
    }
    /* New styles for routing explanations */
    .routing-explanation {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-top: 5px;
        margin-bottom: 15px;
        border-left: 3px solid #9575CD;
        font-size: 0.9em;
    }
    .routing-tag {
        display: inline-block;
        background-color: #e8eaf6;
        color: #5c6bc0;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    .strategy-balanced {
        color: #5c6bc0;
    }
    .strategy-cost {
        color: #43a047;
    }
    .strategy-speed {
        color: #fb8c00;
    }
    .strategy-quality {
        color: #ec407a;
    }
    /* Strategy badge styling */
    .strategy-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: bold;
        margin-left: 5px;
    }
    .badge-balanced {
        background-color: #e8eaf6;
        color: #3949ab;
    }
    .badge-cost {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .badge-speed {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .badge-quality {
        background-color: #fce4ec;
        color: #c2185b;
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
        
    if "rerun_message_index" not in st.session_state:
        st.session_state.rerun_message_index = None
        
    if "rerun_model" not in st.session_state:
        st.session_state.rerun_model = None
        
    if "rerun_responses" not in st.session_state:
        st.session_state.rerun_responses = {}
        
    if "routing_strategy" not in st.session_state:
        st.session_state.routing_strategy = "balanced"
        
    if "show_routing_explanation" not in st.session_state:
        st.session_state.show_routing_explanation = True

def display_chat_messages(config, router):
    """Display chat messages from history"""
    if not st.session_state.messages:
        # Welcome message
        st.info("👋 Welcome to the OpenRouter Chatbot! Send a message to get started.")
        return
    
    # Get available models for rerun dropdowns
    available_models = sorted(config.get("models", {}).keys())
    
    # Display all messages
    message_containers = []
    
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            # Create unique keys for this message's components
            rerun_button_key = f"rerun_button_{i}"
            message_container = st.container()
            message_containers.append(message_container)
            
            with message_container:
                # Display user message with a rerun button
                st.markdown(
                    f'<div class="user-message">{content}'
                    f'<button class="rerun-button" id="{rerun_button_key}">Try with different model</button>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Javascript to handle button click
                st.markdown(f"""
                <script>
                    document.getElementById("{rerun_button_key}").addEventListener("click", function() {{
                        window.parent.postMessage({{
                            type: "streamlit:setComponentValue",
                            value: {i},
                            dataType: "json",
                            key: "rerun_message_index"
                        }}, "*");
                    }});
                </script>
                """, unsafe_allow_html=True)
                
                # Check if this message is selected for rerun
                if st.session_state.rerun_message_index == i:
                    st.markdown("#### Try with a different model")
                    
                    # Model selection dropdown
                    selected_model = st.selectbox(
                        "Select model to try",
                        options=available_models,
                        format_func=lambda x: f"{config.get('models', {}).get(x, {}).get('name', x)} ({x})",
                        key=f"rerun_model_select_{i}"
                    )
                    
                    # Rerun button
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if st.button("Run with this model", key=f"execute_rerun_{i}"):
                            # Store the selected model and trigger rerun
                            st.session_state.rerun_model = selected_model
                            
                            # Get the message and all previous messages to maintain context
                            messages_context = []
                            for j in range(0, i+1):
                                if st.session_state.messages[j]["role"] == "user":
                                    messages_context.append(st.session_state.messages[j])
                            
                            # Execute rerun
                            execute_rerun(router, messages_context, i)
                    
                    with col2:
                        if st.button("Cancel", key=f"cancel_rerun_{i}"):
                            st.session_state.rerun_message_index = None
                            st.rerun()
        else:
            # This is an assistant message
            # Check if it's a rerun response
            is_rerun = False
            rerun_model = None
            
            # Find if this is a response to a rerun
            for msg_idx, responses in st.session_state.rerun_responses.items():
                if i == msg_idx + 1:  # This is the original response
                    continue
                    
                for model_id, response_idx in responses.items():
                    if i == response_idx:
                        is_rerun = True
                        rerun_model = model_id
                        break
            
            if is_rerun and rerun_model:
                # Display rerun message with model info
                model_name = config.get("models", {}).get(rerun_model, {}).get("name", rerun_model)
                
                st.markdown(
                    f'<div class="rerun-message">'
                    f'<span class="rerun-tag">Alternative response using {model_name}</span>'
                    f'{content}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            else:
                # Display regular assistant message
                st.markdown(f'<div class="assistant-message">{content}</div>', unsafe_allow_html=True)
            
            # Show metrics for this assistant message if available
            message_index = i // 2
            if st.session_state.show_metrics and message_index < len(st.session_state.metrics):
                metrics = st.session_state.metrics[message_index]
                
                # Format metrics
                model = metrics.get("model", "Unknown model")
                tokens = metrics.get("token_count", 0)
                prompt_tokens = metrics.get("prompt_tokens", 0)
                completion_tokens = metrics.get("completion_tokens", 0)
                latency = metrics.get("latency", 0)
                cost = metrics.get("cost", 0)
                
                # Get routing explanation if available
                routing_explanation = metrics.get("routing_explanation", None)
                strategy = routing_explanation.get("strategy", "balanced") if routing_explanation else "balanced"
                
                # Create strategy badge
                strategy_badge = f'<span class="strategy-badge badge-{strategy}">{strategy.capitalize()}</span>'
                
                # Create HTML for metrics
                metrics_html = f"""
                <div class="metrics-container">
                    <div class="model-tag">Model: {model} {strategy_badge if not is_rerun and not st.session_state.manual_model_selection else ""}</div>
                    <div class="token-usage">Tokens: {prompt_tokens} (prompt) + {completion_tokens} (completion) = {tokens}</div>
                    <div class="cost">Cost: ${cost:.6f}</div>
                    <div class="latency">Latency: {latency:.2f}s</div>
                </div>
                """
                
                # Display metrics
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Show routing explanation if available and enabled
                if st.session_state.show_routing_explanation and not is_rerun and not st.session_state.manual_model_selection:
                    if routing_explanation:
                        explanation_text = routing_explanation.get("explanation", "No explanation available.")
                        
                        # Create expandable section for routing explanation
                        with st.expander("Why was this model selected?", expanded=False):
                            # Format explanation text with appropriate styling
                            formatted_explanation = explanation_text.replace("\n", "<br>")
                            
                            # Display the formatted explanation
                            st.markdown(f'<div class="routing-explanation">{formatted_explanation}</div>', unsafe_allow_html=True)
                            
                            # Add additional information about matched patterns if available
                            matched_patterns = routing_explanation.get("matched_patterns", {})
                            if matched_patterns:
                                st.markdown("#### Pattern Matches")
                                pattern_data = {
                                    "Category": list(matched_patterns.keys()),
                                    "Matches": list(matched_patterns.values())
                                }
                                st.dataframe(pattern_data, use_container_width=True)

def execute_rerun(router, messages_context, user_msg_index):
    """Execute a rerun of a user message with a different model"""
    if not st.session_state.rerun_model:
        st.error("No model selected for rerun.")
        return
    
    # System prompt needs to be included
    full_messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ] + messages_context
    
    # Get the user message that we're rerunning
    user_message = messages_context[-1]["content"]
    
    with st.spinner(f"Getting response from {st.session_state.rerun_model}..."):
        try:
            # Get response from the selected model
            start_time = time.time()
            response_text, metrics = router.send_prompt(
                full_messages, 
                model_id=st.session_state.rerun_model
            )
            total_time = time.time() - start_time
            
            # Calculate and add cost
            model = metrics.get("model", "unknown")
            tokens = metrics.get("token_count", 0)
            cost = router.calculate_cost(tokens, model)
            
            # Add cost to metrics
            metrics["cost"] = cost
            
            # Update total conversation cost
            st.session_state.conversation_cost += cost
            
            # Add assistant message to chat at the end
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Store metrics
            st.session_state.metrics.append(metrics)
            
            # Add to rerun responses mapping
            if user_msg_index not in st.session_state.rerun_responses:
                st.session_state.rerun_responses[user_msg_index] = {}
                
            st.session_state.rerun_responses[user_msg_index][st.session_state.rerun_model] = len(st.session_state.messages) - 1
            
            # Reset rerun state
            st.session_state.rerun_message_index = None
            st.session_state.rerun_model = None
            
            # Rerun to update the UI
            st.rerun()
            
        except Exception as e:
            # Add error message to chat
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            
            # Reset rerun state
            st.session_state.rerun_message_index = None
            st.session_state.rerun_model = None

def display_cost_summary():
    """Display summary of conversation cost and token usage"""
    if not st.session_state.metrics:
        return
    
    # Calculate totals
    total_tokens = sum(m.get("token_count", 0) for m in st.session_state.metrics)
    total_prompt_tokens = sum(m.get("prompt_tokens", 0) for m in st.session_state.metrics)
    total_completion_tokens = sum(m.get("completion_tokens", 0) for m in st.session_state.metrics)
    total_cost = st.session_state.conversation_cost
    
    # Show summary
    st.markdown("### Conversation Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost", f"${total_cost:.6f}")
    
    with col2:
        st.metric("Total Tokens", f"{total_tokens:,}")
    
    with col3:
        # Calculate the actual messages excluding reruns
        original_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Unique Prompts", f"{original_messages}")
    
    # Add a breakdown of token usage
    token_data = {
        "Category": ["Prompt", "Completion", "Total"],
        "Tokens": [total_prompt_tokens, total_completion_tokens, total_tokens],
        "Percentage": [
            f"{(total_prompt_tokens / total_tokens) * 100:.1f}%" if total_tokens > 0 else "0%",
            f"{(total_completion_tokens / total_tokens) * 100:.1f}%" if total_tokens > 0 else "0%",
            "100%"
        ]
    }
    
    st.markdown("#### Token Usage Breakdown")
    st.dataframe(token_data, use_container_width=True)
    
    # Add model usage breakdown if there are reruns
    if st.session_state.rerun_responses:
        st.markdown("#### Model Usage")
        
        # Count responses by model
        model_usage = {}
        for metrics in st.session_state.metrics:
            model = metrics.get("model", "Unknown")
            if model not in model_usage:
                model_usage[model] = {
                    "count": 0,
                    "tokens": 0,
                    "cost": 0
                }
            
            model_usage[model]["count"] += 1
            model_usage[model]["tokens"] += metrics.get("token_count", 0)
            model_usage[model]["cost"] += metrics.get("cost", 0)
        
        # Create data for display
        model_data = []
        for model, stats in model_usage.items():
            model_data.append({
                "Model": model,
                "Responses": stats["count"],
                "Total Tokens": f"{stats['tokens']:,}",
                "Total Cost": f"${stats['cost']:.6f}",
                "Avg Tokens/Response": f"{int(stats['tokens'] / stats['count']):,}" if stats['count'] > 0 else "0"
            })
        
        # Display as table
        st.dataframe(model_data, use_container_width=True)

def display_sidebar(config, router):
    """Display and handle sidebar elements"""
    st.sidebar.title("Chatbot Options")
    
    # Model selection
    st.sidebar.markdown("### Model Selection")
    
    # Toggle for manual model selection
    st.session_state.manual_model_selection = st.sidebar.checkbox(
        "Manual Model Selection", 
        value=st.session_state.manual_model_selection
    )
    
    if st.session_state.manual_model_selection:
        # Get available models
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
        # Routing strategy selection
        st.sidebar.markdown("### Routing Strategy")
        
        # Add a description for each strategy
        st.sidebar.markdown("""
        Select a routing strategy to prioritize different aspects:
        """)
        
        # Radio buttons for strategy selection
        strategy_options = {
            "balanced": "Balanced (default) - Optimizes for overall performance",
            "cost": "Cost-first - Prioritizes cheaper models to minimize expenses",
            "speed": "Speed-first - Prioritizes faster models for quicker responses",
            "quality": "Quality-first - Prioritizes high-quality models for best results"
        }
        
        selected_strategy = st.sidebar.radio(
            "Select Strategy",
            options=list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            index=list(strategy_options.keys()).index(st.session_state.routing_strategy)
        )
        
        # Update routing strategy if changed
        if selected_strategy != st.session_state.routing_strategy:
            st.session_state.routing_strategy = selected_strategy
            if hasattr(router, 'set_routing_strategy'):
                router.set_routing_strategy(selected_strategy)
                st.sidebar.success(f"Routing strategy set to: {selected_strategy}")
        
        # Show explanation of rule-based model selection
        st.sidebar.markdown("Model will be selected based on prompt type and length:")
        st.sidebar.markdown("- **Code**: For programming and technical questions")
        st.sidebar.markdown("- **Summary**: For summarizing or condensing information")
        st.sidebar.markdown("- **Question**: For general questions and inquiries")
        
        # Toggle for showing routing explanation
        st.session_state.show_routing_explanation = st.sidebar.checkbox(
            "Show Routing Explanations", 
            value=st.session_state.show_routing_explanation
        )
    
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
        st.session_state.rerun_responses = {}
        st.session_state.rerun_message_index = None
        st.session_state.rerun_model = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This chatbot uses rule-based routing to select the most appropriate model "
        "based on prompt type and length. You can also try the same prompt with "
        "different models by clicking the 'Try with different model' button next to any message."
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
                else:
                    # Apply current routing strategy
                    router.set_routing_strategy(st.session_state.routing_strategy)
                
                # Get response from router
                start_time = time.time()
                response_text, metrics = router.send_prompt(full_messages, model_id=model_override)
                total_time = time.time() - start_time
                
                # Get routing explanation if available
                if not model_override and hasattr(router, 'get_routing_explanation'):
                    routing_explanation = router.get_routing_explanation()
                    metrics["routing_explanation"] = routing_explanation
                
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
                # Add error message to chat
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
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
    st.title("💬 Chatbot")
    st.markdown(
        "Chat with AI models using rule-based routing or manual model selection. "
        "Try the same prompt with different models by clicking the 'Try with different model' button next to any message."
    )
    
    # Display chat messages
    display_chat_messages(config, router)
    
    # Process user input
    process_user_input(router)
    
    # Display cost and usage summary
    if st.session_state.metrics:
        st.markdown("---")
        display_cost_summary()

if __name__ == "__main__":
    main() 