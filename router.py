from typing import Dict, List, Any, Optional
import re
import logging
from model_client import OpenRouterClient

class ModelRouter:
    """
    Router for routing prompts to appropriate models based on content
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = config.get("models", {})
        self.prompt_types = config.get("prompt_types", {})
        self.default_model = config.get("default_model")
        self.client = OpenRouterClient()
        
        # Set up logging
        logging.basicConfig(
            filename="logs/router.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )
        
        if not self.default_model:
            raise ValueError("Default model must be specified in configuration")
        
    def get_model_for_prompt(self, prompt: str) -> str:
        """
        Select the appropriate model ID based on the content of the prompt
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            model_id: The ID of the selected model to use
        """
        # Check each prompt type's patterns
        for prompt_type, info in self.prompt_types.items():
            patterns = info.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    selected_model = info.get("preferred_model", self.default_model)
                    logging.info(f"Prompt matched '{prompt_type}' pattern. Selected model: {selected_model}")
                    return selected_model
        
        # If no match found, return the default model
        logging.info(f"No pattern matched. Using default model: {self.default_model}")
        return self.default_model
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available models from the configuration
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                "id": model_id,
                "name": info.get("name", model_id),
                "description": info.get("description", ""),
                "strengths": info.get("strengths", []),
                "cost_per_1k_tokens": info.get("cost_per_1k_tokens", 0),
                "max_tokens": info.get("max_tokens", 4096)
            }
            for model_id, info in self.models.items()
        ]
    
    def send_to_model(self, messages: List[Dict[str, str]], model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send the messages to the appropriate model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model_id: Optional model ID override
            
        Returns:
            Response from the model
        """
        # If model_id is not specified, determine it from the user's last message
        if not model_id and len(messages) > 0:
            user_messages = [m for m in messages if m["role"] == "user"]
            if user_messages:
                last_user_message = user_messages[-1]["content"]
                model_id = self.get_model_for_prompt(last_user_message)
        
        # Fall back to default model if still not determined
        model_id = model_id or self.default_model
        
        # Get model-specific parameters
        model_info = self.models.get(model_id, {})
        temperature = model_info.get("temperature", 0.7)
        max_tokens = model_info.get("max_tokens", 1000)
        
        # Log the model selection
        logging.info(f"Sending prompt to model: {model_id} (temp: {temperature}, max tokens: {max_tokens})")
        
        # Generate and return the response
        return self.client.generate_response(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens
        ) 