"""
Rule-Based Router for OpenRouter API

This module provides a rule-based router that selects the most appropriate model
based on prompt type (code, summary, question) and prompt length.
"""

import os
import re
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
import tiktoken
from datetime import datetime

from src.api.openrouter_client_enhanced import send_prompt_to_openrouter
from src.utils.cost_tracker import CostTracker
from src.utils.model_call_logger import log_model_call

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "rule_based_router.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger("rule_based_router")

class RuleBasedRouter:
    """
    Rule-based router that selects models based on prompt type and length
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the router
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.models = config.get("models", {})
        self.default_model = config.get("default_model")
        
        # Set up cost tracker
        self.cost_tracker = CostTracker(config)
        
        # Maximum token lengths for different categories
        self.length_categories = {
            "short": 500,      # 0-500 tokens
            "medium": 2000,    # 501-2000 tokens
            "long": 8000       # 2001-8000 tokens
        }
        
        # Routing strategy - default to 'balanced'
        self.routing_strategy = config.get("optimization_target", "balanced")
        
        # Define model preferences for different prompt types and lengths
        self.model_preferences = {
            "code": {
                "short": config.get("code_models", {}).get("short", "openai/gpt-4o"),
                "medium": config.get("code_models", {}).get("medium", "openai/gpt-4o"),
                "long": config.get("code_models", {}).get("long", "anthropic/claude-3-opus")
            },
            "summary": {
                "short": config.get("summary_models", {}).get("short", "anthropic/claude-3-haiku"),
                "medium": config.get("summary_models", {}).get("medium", "anthropic/claude-3-sonnet"),
                "long": config.get("summary_models", {}).get("long", "anthropic/claude-3-sonnet")
            },
            "question": {
                "short": config.get("question_models", {}).get("short", "anthropic/claude-3-haiku"),
                "medium": config.get("question_models", {}).get("medium", "anthropic/claude-3-haiku"),
                "long": config.get("question_models", {}).get("long", "anthropic/claude-3-sonnet")
            }
        }
        
        # Define strategy-specific model preferences
        self.strategy_preferences = self._initialize_strategy_preferences()
        
        if not self.default_model:
            self.default_model = "anthropic/claude-3-haiku"
            logger.warning(f"No default model specified, using {self.default_model}")
    
    def _initialize_strategy_preferences(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Initialize strategy-specific model preferences based on model characteristics
        
        Returns:
            Dictionary of strategies with model preferences
        """
        strategies = {
            "cost": {},
            "speed": {},
            "quality": {},
            "balanced": {}
        }
        
        # Create a mapping of models by category (cost, speed, quality)
        models_by_cost = sorted(self.models.keys(), key=lambda m: self.models[m].get("cost_per_1k_tokens", 999))
        models_by_context = sorted(self.models.keys(), key=lambda m: self.models[m].get("context_length", 0), reverse=True)
        
        # For speed, we're assuming smaller models are generally faster
        # In a real application, you might want to measure and record actual latency
        models_by_speed = sorted(self.models.keys(), key=lambda m: self.models[m].get("cost_per_1k_tokens", 999))
        
        # Quality is often correlated with model size/cost
        models_by_quality = sorted(self.models.keys(), key=lambda m: self.models[m].get("cost_per_1k_tokens", 0), reverse=True)
        
        # Create strategy preferences for each prompt type and length
        for prompt_type in ["code", "summary", "question"]:
            strategies["cost"][prompt_type] = {
                "short": models_by_cost[0] if models_by_cost else self.default_model,
                "medium": models_by_cost[0] if models_by_cost else self.default_model,
                "long": models_by_cost[0] if models_by_cost else self.default_model
            }
            
            strategies["speed"][prompt_type] = {
                "short": models_by_speed[0] if models_by_speed else self.default_model,
                "medium": models_by_speed[0] if models_by_speed else self.default_model,
                "long": models_by_speed[0] if models_by_speed else self.default_model
            }
            
            strategies["quality"][prompt_type] = {
                "short": models_by_quality[0] if models_by_quality else self.default_model,
                "medium": models_by_quality[0] if models_by_quality else self.default_model,
                "long": models_by_quality[1] if len(models_by_quality) > 1 else models_by_quality[0] if models_by_quality else self.default_model
            }
            
            # Balanced strategy uses the default model preferences
            strategies["balanced"][prompt_type] = self.model_preferences[prompt_type]
            
        return strategies
    
    def set_routing_strategy(self, strategy: str) -> None:
        """
        Set the routing strategy
        
        Args:
            strategy: Strategy to use ("balanced", "cost", "speed", "quality")
        """
        if strategy in ["balanced", "cost", "speed", "quality"]:
            self.routing_strategy = strategy
            logger.info(f"Routing strategy set to: {strategy}")
        else:
            logger.warning(f"Invalid strategy: {strategy}. Using 'balanced' instead.")
            self.routing_strategy = "balanced"
    
    def classify_prompt(self, prompt: str) -> str:
        """
        Classify the prompt type (code, summary, question)
        
        Args:
            prompt: The user's prompt text
            
        Returns:
            Prompt classification ("code", "summary", "question")
        """
        # Code patterns
        code_patterns = [
            r'```[a-z]*\n',         # Code blocks
            r'function\s+\w+\s*\(',  # Function declarations
            r'class\s+\w+',          # Class declarations
            r'def\s+\w+\s*\(',       # Python function declarations
            r'import\s+\w+',         # Import statements
            r'from\s+\w+\s+import',  # Python imports
            r'const\s+\w+\s*=',      # JavaScript constants
            r'let\s+\w+\s*=',        # JavaScript variables
            r'var\s+\w+\s*=',        # JavaScript variables (old style)
            r'public\s+\w+\s+\w+\(',  # Java/C# method declarations
            r'#include',              # C/C++ includes
            r'<script',               # HTML script tags
            r'<style',                # HTML style tags
            r'package\s+\w+',         # Java/Kotlin package declarations
            r'@\w+',                  # Decorators/annotations
            r'SELECT\s+.*\s+FROM',    # SQL queries
            r'CREATE\s+TABLE',        # SQL table creation
            r'\w+\s*\(\s*\)\s*{',     # Function body starter
            r'^\s*for\s*\(\s*\w+',    # For loops at start of line
            r'^\s*if\s*\(\s*\w+',     # If statements at start of line
            r'^\s*while\s*\('        # While loops at start of line
        ]
        
        # Summary patterns
        summary_patterns = [
            r'\bsummarize\b',
            r'\bsummary\b',
            r'\bsummarise\b',
            r'\bcondense\b',
            r'\brecap\b',
            r'\bshorten\b',
            r'\bsynthesize\b',
            r'\bsynopsis\b',
            r'\babbreviate\b',
            r'\bdigest\b',
            r'\btl;dr\b',
            r'\btldr\b',
            r'\bkey points\b',
            r'\bmain points\b',
            r'\bhighlight\b',
            r'\boverview\b',
            r'\bbriefing\b'
        ]
        
        # Question patterns
        question_patterns = [
            r'\?\s*$',                 # Ends with question mark
            r'^[Ww]hat\b',             # Starts with What
            r'^[Hh]ow\b',              # Starts with How
            r'^[Ww]hy\b',              # Starts with Why
            r'^[Ww]hen\b',             # Starts with When
            r'^[Ww]here\b',            # Starts with Where
            r'^[Ww]ho\b',              # Starts with Who
            r'^[Cc]an\b',              # Starts with Can
            r'^[Dd]o\b',               # Starts with Do
            r'^[Ii]s\b',               # Starts with Is
            r'^[Aa]re\b',              # Starts with Are
            r'^[Cc]ould\b',            # Starts with Could
            r'^[Ss]hould\b',           # Starts with Should
            r'\bexplain\b',            # Contains explain
            r'\belaborate\b',          # Contains elaborate
            r'\bdiscuss\b',            # Contains discuss
            r'\bdescribe\b',           # Contains describe
            r'\btell me\b',            # Contains tell me
            r'\bI need to know\b'      # Contains I need to know
        ]
        
        # Count pattern matches
        code_matches = sum(1 for pattern in code_patterns if re.search(pattern, prompt, re.IGNORECASE))
        summary_matches = sum(1 for pattern in summary_patterns if re.search(pattern, prompt, re.IGNORECASE))
        question_matches = sum(1 for pattern in question_patterns if re.search(pattern, prompt, re.IGNORECASE))
        
        # Determine the prompt type based on the most matches
        max_matches = max(code_matches, summary_matches, question_matches)
        
        # Save the matched patterns for explanation
        self.matched_patterns = {
            "code": code_matches,
            "summary": summary_matches,
            "question": question_matches
        }
        
        if max_matches == 0:
            # No clear pattern matched, check for code-like syntax
            if re.search(r'[{};()]', prompt) and len(prompt.split('\n')) > 3:
                self.prompt_detection_reason = "Code-like syntax detected with brackets, semicolons, or parentheses."
                return "code"
            # If prompt is long, default to summary
            elif len(prompt) > 1000:
                self.prompt_detection_reason = "Long text input without clear patterns detected, treating as a summary task."
                return "summary"
            # Default to question for shorter prompts
            else:
                self.prompt_detection_reason = "Short text input with no clear patterns, treating as a general question."
                return "question"
        
        if code_matches == max_matches:
            self.prompt_detection_reason = f"Detected {code_matches} code-related patterns in the prompt."
            return "code"
        elif summary_matches == max_matches:
            self.prompt_detection_reason = f"Detected {summary_matches} summary-related patterns in the prompt."
            return "summary"
        else:
            self.prompt_detection_reason = f"Detected {question_matches} question-related patterns in the prompt."
            return "question"
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in the text
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        try:
            # Try to use tiktoken for accurate counting
            encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding for most models
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error using tiktoken: {e}. Using approximate method.")
            # Fall back to approximation: ~4 chars per token for English text
            return len(text) // 4
    
    def determine_length_category(self, token_count: int) -> str:
        """
        Determine the length category of the prompt
        
        Args:
            token_count: Number of tokens in the prompt
            
        Returns:
            Length category ("short", "medium", "long")
        """
        if token_count <= self.length_categories["short"]:
            return "short"
        elif token_count <= self.length_categories["medium"]:
            return "medium"
        else:
            return "long"
    
    def select_model(self, prompt: str) -> str:
        """
        Select the appropriate model based on prompt type, length, and current routing strategy
        
        Args:
            prompt: The user's prompt
            
        Returns:
            Selected model ID and explanation
        """
        # Classify the prompt type
        prompt_type = self.classify_prompt(prompt)
        
        # Estimate token count
        token_count = self.estimate_token_count(prompt)
        
        # Determine length category
        length_category = self.determine_length_category(token_count)
        
        # Initialize variables for explanation
        strategy_used = self.routing_strategy
        model_candidates = {}
        model_reasons = {}
        
        # Get model suggestions from different strategies
        for strategy in ["balanced", "cost", "speed", "quality"]:
            strategy_model = self.strategy_preferences.get(strategy, {}).get(prompt_type, {}).get(length_category, self.default_model)
            model_candidates[strategy] = strategy_model
            
            # Generate reasons for each strategy
            if strategy == "cost":
                model_reasons[strategy] = f"{strategy_model} is most cost-effective at ${self.models.get(strategy_model, {}).get('cost_per_1k_tokens', 0):.6f} per 1K tokens."
            elif strategy == "speed":
                model_reasons[strategy] = f"{strategy_model} is optimized for faster responses."
            elif strategy == "quality":
                model_reasons[strategy] = f"{strategy_model} offers highest quality for {prompt_type} tasks."
            else:  # balanced
                model_reasons[strategy] = f"{strategy_model} provides a good balance of cost, speed, and quality for {prompt_type} tasks."
        
        # Select model based on current strategy
        model_id = model_candidates.get(self.routing_strategy, self.default_model)
        
        # Check if model exists in configuration, fall back to default if not
        if model_id not in self.models:
            logger.warning(f"Selected model {model_id} not found in configuration, using default.")
            model_id = self.default_model
            self.model_selection_explanation = f"Selected model {model_id} (default) because the initial selection was not found in configuration."
        else:
            # Create explanation
            self.model_selection_explanation = (
                f"Selected model: {model_id}\n"
                f"• Prompt type: {prompt_type} ({self.prompt_detection_reason})\n"
                f"• Length: {length_category} ({token_count} tokens)\n"
                f"• Strategy: {self.routing_strategy}\n"
                f"• Reason: {model_reasons.get(self.routing_strategy, 'No specific reason available.')}"
            )
            
        logger.info(f"Prompt classified as {prompt_type} ({length_category}), selected model: {model_id}")
        return model_id
    
    def get_routing_explanation(self) -> Dict[str, Any]:
        """
        Get human-readable explanation for the routing decision
        
        Returns:
            Dictionary with explanation details
        """
        return {
            "explanation": self.model_selection_explanation if hasattr(self, 'model_selection_explanation') else "No explanation available",
            "prompt_type": self.classify_prompt.__defaults__[0] if hasattr(self, 'classify_prompt') and self.classify_prompt.__defaults__ else "unknown",
            "strategy": self.routing_strategy,
            "matched_patterns": self.matched_patterns if hasattr(self, 'matched_patterns') else {}
        }
    
    def send_prompt(self, messages: List[Dict[str, str]], 
                    model_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Send the prompt to the selected model and log metrics
        
        Args:
            messages: List of message dictionaries
            model_id: Optional model ID override
        
        Returns:
            Tuple of (response_text, metrics)
        """
        # Get the user's prompt from the messages
        user_messages = [m for m in messages if m["role"] == "user"]
        if not user_messages:
            raise ValueError("No user message found in the provided messages")
        
        prompt = user_messages[-1]["content"]
        
        # Generate a unique identifier for this request
        prompt_id = str(uuid.uuid4())
        
        # Generate a session ID if not already in the metrics
        session_id = getattr(self, 'session_id', str(uuid.uuid4()))
        self.session_id = session_id
        
        # Track whether this is a manual selection
        manual_selection = model_id is not None
        
        # Select model if not specified
        if not model_id:
            # Store original routing strategy for logging
            original_strategy = self.routing_strategy
            # Select model based on prompt content and current strategy
            model_id = self.select_model(prompt)
        else:
            # If manually selected, we don't have routing explanations
            original_strategy = "manual"
            self.model_selection_explanation = f"Model {model_id} was manually selected by the user."
            # For manually selected models, we still want to classify the prompt type for logging
            prompt_type = self.classify_prompt(prompt)
            length_category = self.determine_length_category(self.estimate_token_count(prompt))
        
        # Get routing explanation for logging
        routing_explanation = self.get_routing_explanation() if hasattr(self, 'get_routing_explanation') else {}
        
        # Get model-specific parameters
        model_info = self.models.get(model_id, {})
        temperature = model_info.get("temperature", 0.7)
        
        # Use a reasonable max_tokens value (either from config or default to 1000)
        # This is the maximum number of tokens to generate in the response
        max_tokens = model_info.get("max_tokens", 1000)
        
        # Make sure max_tokens is reasonable (cap at 4000 to be safe)
        if max_tokens > 4000:
            logger.warning(f"Reducing max_tokens from {max_tokens} to 4000 to prevent API errors")
            max_tokens = 4000
        
        # Estimate prompt tokens
        prompt_tokens = self.estimate_token_count(prompt)
        logger.info(f"Estimated prompt tokens: {prompt_tokens}, max response tokens: {max_tokens}")
        
        # Record start time
        start_time = datetime.now()
        error_type = None
        
        try:
            # Send to OpenRouter
            response_text, usage_stats, latency = send_prompt_to_openrouter(
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Record end time and calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Classify prompt type for logging
            prompt_type = self.classify_prompt(prompt)
            length_category = self.determine_length_category(prompt_tokens)
            
            # Create metrics dictionary
            metrics = {
                "model": model_id,
                "prompt_type": prompt_type,
                "token_count": usage_stats.get("total_tokens", 0),
                "prompt_tokens": usage_stats.get("prompt_tokens", 0),
                "completion_tokens": usage_stats.get("completion_tokens", 0),
                "latency": latency,
                "duration": duration,
                "timestamp": end_time.isoformat(),
                "usage_stats": usage_stats,
                "routing_explanation": routing_explanation,
                "session_id": session_id,
                "prompt_id": prompt_id
            }
            
            # Log costs using the cost tracker
            self.cost_tracker.log_api_call(
                model=model_id,
                usage_stats=usage_stats
            )
            
            # Log the model call with routing explanation
            log_model_call(
                session_id=session_id,
                model_id=model_id,
                prompt_type=prompt_type,
                prompt_query=prompt,
                usage_stats=usage_stats,
                routing_explanation=routing_explanation,
                prompt_id=prompt_id,
                length_category=length_category,
                strategy=original_strategy,
                manual_selection=manual_selection,
                latency=latency,
                success=True,
                matched_patterns=getattr(self, 'matched_patterns', {}) if hasattr(self, 'matched_patterns') else {},
                additional_metadata={
                    "duration": duration,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
            # Log the request/response details
            self.log_interaction(prompt, response_text, metrics)
            
            return response_text, metrics
            
        except Exception as e:
            logger.error(f"Error sending prompt to {model_id}: {e}")
            error_type = str(e)
            
            # Log the failed call
            try:
                # Determine execution time even for failed calls
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Create basic usage stats for failed calls
                failed_usage_stats = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": prompt_tokens,
                    "cost": 0
                }
                
                # Log the failed model call
                log_model_call(
                    session_id=session_id,
                    model_id=model_id,
                    prompt_type=getattr(self, 'prompt_type', "unknown"),
                    prompt_query=prompt,
                    usage_stats=failed_usage_stats,
                    routing_explanation=routing_explanation,
                    prompt_id=prompt_id,
                    length_category=length_category if 'length_category' in locals() else None,
                    strategy=original_strategy,
                    manual_selection=manual_selection,
                    latency=duration,
                    success=False,
                    error_type=error_type,
                    matched_patterns=getattr(self, 'matched_patterns', {}) if hasattr(self, 'matched_patterns') else {},
                    additional_metadata={
                        "duration": duration,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
            except Exception as log_error:
                logger.error(f"Error logging failed call: {log_error}")
            
            # Check if the error is related to token limits
            error_str = str(e).lower()
            if "maximum context length" in error_str or "token" in error_str or "too long" in error_str:
                logger.warning("Token limit error detected. Trying with reduced max_tokens.")
                
                # Reduce max_tokens and try again
                reduced_max_tokens = max(max_tokens // 2, 500)  # Reduce by half but not below 500
                logger.info(f"Retrying with max_tokens={reduced_max_tokens}")
                
                try:
                    response_text, usage_stats, latency = send_prompt_to_openrouter(
                        messages=messages,
                        model=model_id,
                        temperature=temperature,
                        max_tokens=reduced_max_tokens
                    )
                    
                    # Record end time and calculate duration
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # Classify prompt type for logging
                    prompt_type = self.classify_prompt(prompt)
                    
                    # Create metrics dictionary
                    metrics = {
                        "model": model_id,
                        "prompt_type": prompt_type,
                        "token_count": usage_stats.get("total_tokens", 0),
                        "latency": latency,
                        "duration": duration,
                        "timestamp": end_time.isoformat(),
                        "usage_stats": usage_stats,
                        "routing_explanation": routing_explanation,
                        "session_id": session_id,
                        "prompt_id": prompt_id,
                        "retry": True  # Indicate this was a retry
                    }
                    
                    # Log costs using the cost tracker
                    self.cost_tracker.log_api_call(
                        model=model_id,
                        usage_stats=usage_stats
                    )
                    
                    # Log the successful retry
                    log_model_call(
                        session_id=session_id,
                        model_id=model_id,
                        prompt_type=prompt_type,
                        prompt_query=prompt,
                        usage_stats=usage_stats,
                        routing_explanation=routing_explanation,
                        prompt_id=prompt_id,
                        length_category=self.determine_length_category(prompt_tokens),
                        strategy=original_strategy,
                        manual_selection=manual_selection,
                        latency=latency,
                        success=True,
                        matched_patterns=getattr(self, 'matched_patterns', {}) if hasattr(self, 'matched_patterns') else {},
                        additional_metadata={
                            "duration": duration,
                            "temperature": temperature,
                            "max_tokens": reduced_max_tokens,
                            "retry": True
                        }
                    )
                    
                    # Log the request/response details
                    self.log_interaction(prompt, response_text, metrics)
                    
                    return response_text, metrics
                    
                except Exception as retry_error:
                    logger.error(f"Still failed after reducing max_tokens: {retry_error}")
                    # Log the failed retry
                    try:
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        failed_usage_stats = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "total_tokens": prompt_tokens,
                            "cost": 0
                        }
                        
                        log_model_call(
                            session_id=session_id,
                            model_id=model_id,
                            prompt_type=getattr(self, 'prompt_type', "unknown"),
                            prompt_query=prompt,
                            usage_stats=failed_usage_stats,
                            routing_explanation=routing_explanation,
                            prompt_id=prompt_id,
                            length_category=self.determine_length_category(prompt_tokens),
                            strategy=original_strategy,
                            manual_selection=manual_selection,
                            latency=duration,
                            success=False,
                            error_type=str(retry_error),
                            matched_patterns=getattr(self, 'matched_patterns', {}) if hasattr(self, 'matched_patterns') else {},
                            additional_metadata={
                                "duration": duration,
                                "temperature": temperature,
                                "max_tokens": reduced_max_tokens,
                                "retry": True
                            }
                        )
                    except Exception as log_error:
                        logger.error(f"Error logging failed retry: {log_error}")
                    # Continue to fallback model logic
            
            # Try fallback model if different from current model
            if model_id != self.default_model:
                logger.info(f"Trying fallback model: {self.default_model}")
                
                # Log the fallback attempt
                try:
                    log_model_call(
                        session_id=session_id,
                        model_id=model_id,
                        prompt_type=getattr(self, 'prompt_type', "unknown"),
                        prompt_query=prompt,
                        usage_stats={"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
                        routing_explanation={"explanation": f"Falling back to {self.default_model} after error with {model_id}"},
                        prompt_id=prompt_id,
                        length_category=self.determine_length_category(prompt_tokens) if 'prompt_tokens' in locals() else None,
                        strategy="fallback",
                        manual_selection=False,
                        latency=0,
                        success=False,
                        error_type=error_type,
                        additional_metadata={
                            "fallback_to": self.default_model,
                            "original_model": model_id
                        }
                    )
                except Exception as log_error:
                    logger.error(f"Error logging fallback attempt: {log_error}")
                    
                return self.send_prompt(messages, self.default_model)
            
            # Propagate the error if no fallback available
            raise
    
    def log_interaction(self, prompt: str, response: str, metrics: Dict[str, Any]) -> None:
        """
        Log the interaction details for analysis
        
        Args:
            prompt: The user's prompt
            response: The model's response
            metrics: Dictionary with request/response metrics
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_type": metrics.get("prompt_type", "unknown"),
            "model": metrics.get("model"),
            "prompt_tokens": metrics.get("usage_stats", {}).get("prompt_tokens", 0),
            "completion_tokens": metrics.get("usage_stats", {}).get("completion_tokens", 0),
            "total_tokens": metrics.get("usage_stats", {}).get("total_tokens", 0),
            "latency": metrics.get("latency", 0),
            "duration": metrics.get("duration", 0),
            "cost": self.calculate_cost(
                metrics.get("usage_stats", {}).get("total_tokens", 0),
                metrics.get("model")
            )
        }
        
        logger.info(f"Interaction logged: {log_entry}")
        
        # Optionally write to a separate log file for detailed analysis
        interactions_log = os.path.join("logs", "interactions.jsonl")
        try:
            with open(interactions_log, "a") as f:
                f.write(f"{str(log_entry)}\n")
        except Exception as e:
            logger.error(f"Error writing to interactions log: {e}")
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """
        Calculate the cost of an API call
        
        Args:
            tokens: Number of tokens used
            model: Model ID
            
        Returns:
            Cost in USD
        """
        model_info = self.models.get(model, {})
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
        return (tokens * cost_per_1k) / 1000 