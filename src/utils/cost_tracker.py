"""
Cost Tracker

A utility module for tracking and analyzing API usage costs.
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

# Set up logging
os.makedirs("logs", exist_ok=True)
cost_logger = logging.getLogger("cost_tracker")
cost_logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join("logs", "cost_tracking.log"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
cost_logger.addHandler(handler)

class CostTracker:
    """
    Track and analyze API usage costs across multiple models
    """
    
    def __init__(self, config: Dict[str, Any], log_file: str = "api_costs.csv"):
        """
        Initialize the cost tracker
        
        Args:
            config: Configuration dictionary with model pricing
            log_file: CSV file to log costs to
        """
        self.config = config
        self.models = config.get("models", {})
        self.log_file = os.path.join("logs", log_file)
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create the log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "model", "prompt_tokens", "completion_tokens", 
                    "total_tokens", "cost", "session_id"
                ])
        
        # Load existing cost data
        self.cost_data = self._load_cost_data()
        
        # Daily and session summaries
        self.daily_costs = {}
        self.session_costs = {}
        self.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        cost_logger.info(f"Cost tracker initialized with session ID: {self.current_session_id}")
    
    def _load_cost_data(self) -> pd.DataFrame:
        """Load existing cost data from CSV file"""
        try:
            if os.path.exists(self.log_file):
                return pd.read_csv(self.log_file)
            return pd.DataFrame(columns=[
                "timestamp", "model", "prompt_tokens", "completion_tokens", 
                "total_tokens", "cost", "session_id"
            ])
        except Exception as e:
            cost_logger.error(f"Error loading cost data: {e}")
            return pd.DataFrame(columns=[
                "timestamp", "model", "prompt_tokens", "completion_tokens", 
                "total_tokens", "cost", "session_id"
            ])
    
    def log_api_call(self, model: str, usage_stats: Dict[str, Any], session_id: Optional[str] = None) -> float:
        """
        Log an API call and calculate its cost
        
        Args:
            model: The model used
            usage_stats: Dictionary with token usage statistics
            session_id: Optional session identifier
        
        Returns:
            The calculated cost
        """
        # Get model pricing info
        model_info = self.models.get(model, {})
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0)
        
        # Extract token counts
        prompt_tokens = usage_stats.get("prompt_tokens", 0)
        completion_tokens = usage_stats.get("completion_tokens", 0)
        total_tokens = usage_stats.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        
        # Calculate cost
        cost = (total_tokens * cost_per_1k) / 1000
        
        # Use provided session ID or current session
        session = session_id or self.current_session_id
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "session_id": session
        }
        
        # Append to CSV file
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                log_entry["timestamp"],
                log_entry["model"],
                log_entry["prompt_tokens"],
                log_entry["completion_tokens"],
                log_entry["total_tokens"],
                log_entry["cost"],
                log_entry["session_id"]
            ])
        
        # Update in-memory data
        self.cost_data = self._load_cost_data()
        
        # Log the cost
        cost_logger.info(f"API call logged: model={model}, tokens={total_tokens}, cost=${cost:.6f}")
        
        return cost
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of costs for a specific session
        
        Args:
            session_id: Optional session ID (uses current session if not provided)
        
        Returns:
            Dictionary with cost summary
        """
        session = session_id or self.current_session_id
        session_data = self.cost_data[self.cost_data["session_id"] == session]
        
        if len(session_data) == 0:
            return {
                "session_id": session,
                "total_cost": 0,
                "total_tokens": 0,
                "calls": 0,
                "models": {}
            }
        
        # Calculate totals
        total_cost = session_data["cost"].sum()
        total_tokens = session_data["total_tokens"].sum()
        total_calls = len(session_data)
        
        # Calculate per-model stats
        model_stats = {}
        for model, group in session_data.groupby("model"):
            model_stats[model] = {
                "cost": group["cost"].sum(),
                "tokens": group["total_tokens"].sum(),
                "calls": len(group)
            }
        
        return {
            "session_id": session,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "calls": total_calls,
            "models": model_stats
        }
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of costs for a specific day
        
        Args:
            date: Optional date string in YYYY-MM-DD format (uses today if not provided)
        
        Returns:
            Dictionary with cost summary
        """
        # Use provided date or today
        target_date = date or datetime.now().strftime("%Y-%m-%d")
        
        # Filter data for the target date
        day_data = self.cost_data[self.cost_data["timestamp"].str.startswith(target_date)]
        
        if len(day_data) == 0:
            return {
                "date": target_date,
                "total_cost": 0,
                "total_tokens": 0,
                "calls": 0,
                "models": {}
            }
        
        # Calculate totals
        total_cost = day_data["cost"].sum()
        total_tokens = day_data["total_tokens"].sum()
        total_calls = len(day_data)
        
        # Calculate per-model stats
        model_stats = {}
        for model, group in day_data.groupby("model"):
            model_stats[model] = {
                "cost": group["cost"].sum(),
                "tokens": group["total_tokens"].sum(),
                "calls": len(group)
            }
        
        return {
            "date": target_date,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "calls": total_calls,
            "models": model_stats
        }
    
    def get_cost_trends(self, days: int = 30) -> pd.DataFrame:
        """
        Get cost trends over a number of days
        
        Args:
            days: Number of past days to include
        
        Returns:
            DataFrame with daily cost data
        """
        # Ensure timestamp is treated as datetime
        self.cost_data["date"] = pd.to_datetime(self.cost_data["timestamp"]).dt.date
        
        # Group by date
        daily_costs = self.cost_data.groupby("date").agg({
            "cost": "sum",
            "total_tokens": "sum",
            "model": "count"
        }).reset_index()
        
        # Rename columns
        daily_costs = daily_costs.rename(columns={"model": "calls"})
        
        # Sort by date and get last N days
        daily_costs = daily_costs.sort_values("date").tail(days)
        
        return daily_costs
    
    def export_cost_report(self, output_file: str = "cost_report.json") -> str:
        """
        Export a comprehensive cost report
        
        Args:
            output_file: JSON file to export to
        
        Returns:
            Path to the exported file
        """
        # Get current session and today's summary
        current_session = self.get_session_summary()
        today = self.get_daily_summary()
        
        # Get all-time stats
        all_time = {
            "total_cost": self.cost_data["cost"].sum(),
            "total_tokens": self.cost_data["total_tokens"].sum(),
            "total_calls": len(self.cost_data),
            "unique_models": self.cost_data["model"].nunique()
        }
        
        # Get per-model stats
        model_stats = {}
        for model, group in self.cost_data.groupby("model"):
            model_stats[model] = {
                "cost": group["cost"].sum(),
                "tokens": group["total_tokens"].sum(),
                "calls": len(group),
                "avg_tokens_per_call": group["total_tokens"].mean()
            }
        
        # Build report
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_session": current_session,
            "today": today,
            "all_time": all_time,
            "model_breakdown": model_stats
        }
        
        # Export to JSON
        output_path = os.path.join("logs", output_file)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return output_path

# Example usage
if __name__ == "__main__":
    # Load config
    import yaml
    
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {"models": {}}
    
    # Initialize tracker
    tracker = CostTracker(config)
    
    # Simulated API call
    tracker.log_api_call(
        model="anthropic/claude-3-haiku",
        usage_stats={
            "prompt_tokens": 100,
            "completion_tokens": 150,
            "total_tokens": 250
        }
    )
    
    # Print session summary
    print(json.dumps(tracker.get_session_summary(), indent=2))
    
    # Export report
    report_path = tracker.export_cost_report()
    print(f"Cost report exported to: {report_path}") 