"""
Model Call Logger for OpenRouter LLM Suite

This module provides functionality to log detailed information about each model call,
including routing decisions, model usage metrics, and other metadata.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "model_calls.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger("model_call_logger")

# Constants
MODEL_CALLS_LOG_FILE = os.path.join("logs", "model_calls.jsonl")
MODEL_CALLS_CSV_FILE = os.path.join("logs", "model_calls.csv")

class ModelCallLogger:
    """
    Logger class for tracking model calls, routing decisions, and performance metrics
    """
    
    def __init__(self, log_file: str = MODEL_CALLS_LOG_FILE, csv_file: str = MODEL_CALLS_CSV_FILE):
        """
        Initialize the model call logger
        
        Args:
            log_file: Path to the JSONL log file
            csv_file: Path to the CSV file for structured data
        """
        self.log_file = log_file
        self.csv_file = csv_file
        self._ensure_log_files_exist()
    
    def _ensure_log_files_exist(self) -> None:
        """Ensure the log files exist and create them with headers if needed"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            headers = [
                "timestamp", "session_id", "user_id", "prompt_id", "model_id", 
                "prompt_type", "length_category", "strategy", "manual_selection",
                "token_count", "prompt_tokens", "completion_tokens", 
                "latency", "cost", "success", "error_type", "query"
            ]
            pd.DataFrame(columns=headers).to_csv(self.csv_file, index=False)
            logger.info(f"Created new CSV log file at {self.csv_file}")
    
    def log_model_call(self, 
                      session_id: str,
                      model_id: str,
                      prompt_type: str,
                      prompt_query: str,
                      usage_stats: Dict[str, Any],
                      routing_explanation: Optional[Dict[str, Any]] = None,
                      user_id: Optional[str] = None,
                      prompt_id: Optional[str] = None,
                      length_category: Optional[str] = None,
                      strategy: str = "balanced",
                      manual_selection: bool = False,
                      latency: float = 0.0,
                      success: bool = True,
                      error_type: Optional[str] = None,
                      matched_patterns: Optional[Dict[str, int]] = None,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a model call with detailed information
        
        Args:
            session_id: Unique identifier for the chat session
            model_id: ID of the model used
            prompt_type: Type of prompt (code, summary, question)
            prompt_query: The actual query text (truncated if needed)
            usage_stats: Dictionary with token usage statistics
            routing_explanation: Explanation for routing decision (if available)
            user_id: ID of the user (if available)
            prompt_id: Unique identifier for the prompt
            length_category: Length category (short, medium, long)
            strategy: Routing strategy used
            manual_selection: Whether the model was manually selected
            latency: Response time in seconds
            success: Whether the call was successful
            error_type: Type of error if unsuccessful
            matched_patterns: Dictionary of pattern matches used for classification
            additional_metadata: Any additional information to log
        """
        timestamp = datetime.now().isoformat()
        
        # Create log entry dictionary
        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "user_id": user_id,
            "prompt_id": prompt_id or timestamp,
            "model_id": model_id,
            "prompt_type": prompt_type,
            "length_category": length_category,
            "strategy": strategy,
            "manual_selection": manual_selection,
            "token_count": usage_stats.get("total_tokens", 0),
            "prompt_tokens": usage_stats.get("prompt_tokens", 0),
            "completion_tokens": usage_stats.get("completion_tokens", 0),
            "latency": latency,
            "cost": usage_stats.get("cost", 0),
            "success": success,
            "error_type": error_type,
            "query": self._truncate_text(prompt_query, 100),  # Truncate query for CSV
            "full_query": prompt_query,  # Store full query in JSON
            "routing_explanation": routing_explanation,
            "matched_patterns": matched_patterns
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            log_entry.update(additional_metadata)
        
        # Log to console/file
        log_message = (
            f"Model call: {model_id} | "
            f"Type: {prompt_type} | "
            f"Strategy: {strategy} | "
            f"Tokens: {usage_stats.get('total_tokens', 0)} | "
            f"Cost: ${usage_stats.get('cost', 0):.6f} | "
            f"Latency: {latency:.2f}s | "
            f"Success: {success}"
        )
        if success:
            logger.info(log_message)
        else:
            logger.error(f"{log_message} | Error: {error_type}")
        
        # Write to JSONL file
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to JSONL log file: {e}")
        
        # Write to CSV file (with subset of fields)
        try:
            csv_entry = {
                "timestamp": timestamp,
                "session_id": session_id,
                "user_id": user_id,
                "prompt_id": prompt_id or timestamp,
                "model_id": model_id,
                "prompt_type": prompt_type,
                "length_category": length_category,
                "strategy": strategy,
                "manual_selection": manual_selection,
                "token_count": usage_stats.get("total_tokens", 0),
                "prompt_tokens": usage_stats.get("prompt_tokens", 0),
                "completion_tokens": usage_stats.get("completion_tokens", 0),
                "latency": latency,
                "cost": usage_stats.get("cost", 0),
                "success": success,
                "error_type": error_type,
                "query": self._truncate_text(prompt_query, 100)
            }
            # Append to CSV
            pd.DataFrame([csv_entry]).to_csv(self.csv_file, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Error writing to CSV log file: {e}")
    
    def _truncate_text(self, text: str, max_length: int = 100) -> str:
        """Truncate text to specified length for CSV storage"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def get_recent_calls(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent model calls
        
        Args:
            n: Number of recent calls to return
            
        Returns:
            List of recent call data
        """
        try:
            if os.path.exists(self.log_file):
                entries = []
                with open(self.log_file, "r") as f:
                    for line in f:
                        try:
                            entries.append(json.loads(line.strip()))
                        except:
                            pass
                return sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:n]
            return []
        except Exception as e:
            logger.error(f"Error reading model call log: {e}")
            return []
    
    def get_calls_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all model calls for a specific session
        
        Args:
            session_id: Session ID to filter by
            
        Returns:
            List of call data for the session
        """
        try:
            if os.path.exists(self.log_file):
                session_entries = []
                with open(self.log_file, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("session_id") == session_id:
                                session_entries.append(entry)
                        except:
                            pass
                return sorted(session_entries, key=lambda x: x.get("timestamp", ""))
            return []
        except Exception as e:
            logger.error(f"Error reading model call log: {e}")
            return []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of model usage
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                
                # Filter to successful calls only
                successful_df = df[df['success'] == True]
                
                stats = {
                    "total_calls": len(df),
                    "successful_calls": len(successful_df),
                    "error_rate": (len(df) - len(successful_df)) / len(df) if len(df) > 0 else 0,
                    "total_tokens": successful_df['token_count'].sum(),
                    "total_cost": successful_df['cost'].sum(),
                    "avg_latency": successful_df['latency'].mean(),
                    "calls_by_model": successful_df.groupby('model_id').size().to_dict(),
                    "calls_by_type": successful_df.groupby('prompt_type').size().to_dict(),
                    "calls_by_strategy": successful_df.groupby('strategy').size().to_dict(),
                    "total_sessions": successful_df['session_id'].nunique()
                }
                return stats
            return {"total_calls": 0}
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            return {"error": str(e)}
    
    def export_to_csv(self, output_file: str) -> bool:
        """
        Export the log data to a CSV file
        
        Args:
            output_file: Path to the output CSV file
            
        Returns:
            Success status
        """
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                df.to_csv(output_file, index=False)
                return True
            return False
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

# Singleton instance
model_call_logger = ModelCallLogger()

def log_model_call(*args, **kwargs):
    """Convenience function to log a model call using the singleton logger"""
    return model_call_logger.log_model_call(*args, **kwargs)

def get_recent_calls(n: int = 10):
    """Convenience function to get recent calls using the singleton logger"""
    return model_call_logger.get_recent_calls(n)

def get_calls_by_session(session_id: str):
    """Convenience function to get calls by session using the singleton logger"""
    return model_call_logger.get_calls_by_session(session_id)

def get_summary_stats():
    """Convenience function to get summary stats using the singleton logger"""
    return model_call_logger.get_summary_stats() 