import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.utils.router import ModelRouter

class ChatUI:
    def __init__(self, router: ModelRouter):
        self.router = router
        
        # Initialize session state for chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = self.router.default_model
            
        if "show_model_info" not in st.session_state:
            st.session_state.show_model_info = False
            
        if "total_cost" not in st.session_state:
            st.session_state.total_cost = 0.0
    
    def handle_message_submission(self, user_message: str) -> None:
        """Handle the submission of a new user message"""
        if not user_message.strip():
            return
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})
        
        # If auto-routing is enabled, determine the model
        if st.session_state.get("auto_routing", True):
            model_id = self.router.get_model_for_prompt(user_message)
        else:
            model_id = st.session_state.selected_model
            
        with st.spinner(f"Thinking... (using {self.router.models.get(model_id, {}).get('name', model_id)})"):
            # Create the message history in the format the API expects
            messages = [{"role": m["role"], "content": m["content"]} 
                      for m in st.session_state.messages]
            
            # Get the response from the model
            response = self.router.send_to_model(messages, model_id)
            
            if "error" in response:
                st.error(f"Error: {response['error']}")
                return
                
            # Extract the response text
            try:
                assistant_message = response["choices"][0]["message"]["content"]
                used_model = response.get("model", model_id)
                
                # Calculate and add cost
                prompt_tokens = response.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = response.get("usage", {}).get("completion_tokens", 0)
                model_info = self.router.models.get(used_model, {})
                cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
                
                estimated_cost = (prompt_tokens + completion_tokens) * cost_per_1k / 1000
                st.session_state.total_cost += estimated_cost
                
                # Add response to chat history with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "model": used_model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": estimated_cost
                })
                
            except (KeyError, IndexError) as e:
                st.error(f"Error processing response: {str(e)}")
                st.json(response)  # Show the raw response for debugging
    
    def render_sidebar(self) -> None:
        """Render the sidebar with model selection and settings"""
        with st.sidebar:
            st.subheader("Settings")
            
            # Auto-routing toggle
            auto_routing = st.toggle("Auto-route to best model", 
                                  value=st.session_state.get("auto_routing", True))
            st.session_state.auto_routing = auto_routing
            
            # Manual model selection (active when auto-routing is off)
            available_models = self.router.get_available_models()
            model_options = {m["name"]: m["id"] for m in available_models}
            
            selected_model_name = st.selectbox(
                "Select model" if not auto_routing else "Default fallback model",
                options=list(model_options.keys()),
                index=list(model_options.values()).index(st.session_state.selected_model) 
                      if st.session_state.selected_model in model_options.values() else 0,
                disabled=auto_routing
            )
            st.session_state.selected_model = model_options[selected_model_name]
            
            # Model information toggle
            show_info = st.toggle("Show model details", value=st.session_state.show_model_info)
            st.session_state.show_model_info = show_info
            
            if show_info:
                # Find the selected model info
                selected_model_info = next(
                    (m for m in available_models if m["id"] == st.session_state.selected_model), 
                    None
                )
                
                if selected_model_info:
                    st.subheader(f"About {selected_model_info['name']}")
                    st.write(selected_model_info["description"])
                    
                    st.write("#### Strengths")
                    for strength in selected_model_info.get("strengths", []):
                        st.write(f"- {strength}")
                    
                    st.write(f"Cost: ${selected_model_info.get('cost_per_1k_tokens', 'N/A')}/1K tokens")
                    st.write(f"Max context: {selected_model_info.get('max_tokens', 'N/A')} tokens")
            
            # Display session cost
            st.metric("Estimated session cost", f"${st.session_state.total_cost:.5f}")
            
            # Clear chat button
            if st.button("Clear chat"):
                st.session_state.messages = []
                st.session_state.total_cost = 0.0
                st.rerun()
    
    def render_chat(self) -> None:
        """Render the chat interface with message history"""
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show message metadata if it exists
                    if st.session_state.show_model_info and "model" in message:
                        with st.expander("Message details"):
                            st.write(f"Model: {self.router.models.get(message.get('model', ''), {}).get('name', message.get('model', 'Unknown'))}")
                            st.write(f"Tokens: {message.get('prompt_tokens', 0)} (prompt) + {message.get('completion_tokens', 0)} (completion)")
                            st.write(f"Cost: ${message.get('cost', 0):.5f}")
        
        # Chat input
        if prompt := st.chat_input("Enter your message here"):
            self.handle_message_submission(prompt)
            st.rerun()
    
    def render(self) -> None:
        """Render the complete chat UI"""
        self.render_sidebar()
        self.render_chat() 