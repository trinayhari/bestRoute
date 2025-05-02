#!/usr/bin/env python3
"""
OpenRouter LLM Suite - Model Call Analytics

A dashboard for analyzing model calls, routing decisions, and performance metrics.
"""

import os
import sys
import yaml
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model call logger utilities
try:
    from src.utils.model_call_logger import get_recent_calls, get_summary_stats
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info("Make sure all dependencies are installed and the project structure is correct.")
    sys.exit(1)

# Page config
st.set_page_config(
    page_title="OpenRouter LLM Suite - Model Call Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
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
    .badge-manual {
        background-color: #e0e0e0;
        color: #424242;
    }
    .badge-fallback {
        background-color: #ffebee;
        color: #c62828;
    }
    .routing-explanation {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-top: 5px;
        margin-bottom: 15px;
        border-left: 3px solid #9575CD;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

def load_model_call_data():
    """Load and process model call data from the log files"""
    MODEL_CALLS_LOG_FILE = os.path.join("logs", "model_calls.jsonl")
    
    if not os.path.exists(MODEL_CALLS_LOG_FILE):
        return None
    
    entries = []
    try:
        with open(MODEL_CALLS_LOG_FILE, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        st.error(f"Error loading model call data: {e}")
        return None
    
    if not entries:
        return None
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(entries)
    
    # Convert timestamp strings to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
    
    return df

def display_metrics_summary(stats):
    """Display summary metrics in cards"""
    if not stats:
        st.warning("No model call data available for analysis.")
        return
    
    # Create metrics cards in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total API Calls</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(stats.get("total_calls", 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Tokens Used</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(stats.get("total_tokens", 0))), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Cost</div>
            <div class="metric-value">${:.4f}</div>
        </div>
        """.format(stats.get("total_cost", 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Avg. Latency</div>
            <div class="metric-value">{:.2f}s</div>
        </div>
        """.format(stats.get("avg_latency", 0)), unsafe_allow_html=True)

def plot_model_usage(df):
    """Plot model usage distribution"""
    if df is None or df.empty:
        return
    
    st.subheader("Model Usage")
    
    try:
        # Count model occurrences
        model_counts = df['model_id'].value_counts().reset_index()
        model_counts.columns = ['Model', 'Count']
        
        # Prepare for plotting
        if len(model_counts) > 0:
            fig = px.bar(
                model_counts, 
                x='Model', 
                y='Count',
                title='API Calls by Model',
                color='Model',
                labels={'Count': 'Number of Calls'},
                height=400
            )
            fig.update_layout(xaxis_title="Model", yaxis_title="Number of Calls")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate percentage distribution
            model_counts['Percentage'] = (model_counts['Count'] / model_counts['Count'].sum() * 100).round(2)
            model_counts['Percentage'] = model_counts['Percentage'].apply(lambda x: f"{x}%")
            
            # Display as table
            st.dataframe(model_counts, use_container_width=True)
        else:
            st.info("No model usage data available.")
    except Exception as e:
        st.error(f"Error generating model usage plot: {e}")

def plot_strategy_distribution(df):
    """Plot routing strategy distribution"""
    if df is None or df.empty or 'strategy' not in df.columns:
        return
    
    st.subheader("Routing Strategy Distribution")
    
    try:
        # Count strategy occurrences
        strategy_counts = df['strategy'].value_counts().reset_index()
        strategy_counts.columns = ['Strategy', 'Count']
        
        # Create color map for strategies
        color_map = {
            'balanced': '#3949ab',
            'cost': '#2e7d32',
            'speed': '#ef6c00',
            'quality': '#c2185b',
            'manual': '#424242',
            'fallback': '#c62828'
        }
        
        # Get colors based on strategies in data
        colors = [color_map.get(s.lower(), '#808080') for s in strategy_counts['Strategy']]
        
        # Plot pie chart
        fig = px.pie(
            strategy_counts, 
            values='Count', 
            names='Strategy',
            title='Routing Strategy Distribution',
            color='Strategy',
            color_discrete_map={s: color_map.get(s.lower(), '#808080') for s in strategy_counts['Strategy']},
            height=400
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate percentage
        strategy_counts['Percentage'] = (strategy_counts['Count'] / strategy_counts['Count'].sum() * 100).round(2)
        strategy_counts['Percentage'] = strategy_counts['Percentage'].apply(lambda x: f"{x}%")
        
        # Display as table
        st.dataframe(strategy_counts, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating strategy distribution plot: {e}")

def plot_prompt_types(df):
    """Plot prompt type distribution"""
    if df is None or df.empty or 'prompt_type' not in df.columns:
        return
    
    st.subheader("Prompt Type Analysis")
    
    try:
        # Count prompt type occurrences
        prompt_type_counts = df['prompt_type'].value_counts().reset_index()
        prompt_type_counts.columns = ['Prompt Type', 'Count']
        
        # Plot bar chart
        fig = px.bar(
            prompt_type_counts, 
            x='Prompt Type', 
            y='Count',
            title='Distribution of Prompt Types',
            color='Prompt Type',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model selection by prompt type - create a heatmap
        if 'model_id' in df.columns:
            # Get counts of each model-prompt type combination
            model_prompt_counts = pd.crosstab(df['model_id'], df['prompt_type'])
            
            # Create heatmap
            fig = px.imshow(
                model_prompt_counts,
                labels=dict(x="Prompt Type", y="Model", color="Count"),
                title="Model Selection by Prompt Type",
                height=500,
                color_continuous_scale='YlGnBu'
            )
            fig.update_layout(
                xaxis_title="Prompt Type",
                yaxis_title="Model"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating prompt type plots: {e}")

def plot_cost_and_tokens_over_time(df):
    """Plot cost and token usage over time"""
    if df is None or df.empty:
        return
    
    st.subheader("Cost and Token Usage Over Time")
    
    try:
        # Ensure timestamp is in datetime format
        if 'timestamp' in df.columns:
            # Group by date
            df['date'] = df['timestamp'].dt.date
            daily_metrics = df.groupby('date').agg({
                'token_count': 'sum',
                'cost': 'sum',
                'model_id': 'count'
            }).reset_index()
            daily_metrics.columns = ['Date', 'Total Tokens', 'Total Cost', 'API Calls']
            
            # Create tabs for different metrics
            tab1, tab2, tab3 = st.tabs(["Cost Over Time", "Token Usage Over Time", "API Calls Over Time"])
            
            with tab1:
                fig = px.line(
                    daily_metrics, 
                    x='Date', 
                    y='Total Cost',
                    title='Daily Cost',
                    markers=True,
                    height=400
                )
                fig.update_layout(yaxis_title="Cost (USD)")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = px.line(
                    daily_metrics, 
                    x='Date', 
                    y='Total Tokens',
                    title='Daily Token Usage',
                    markers=True,
                    height=400
                )
                fig.update_layout(yaxis_title="Number of Tokens")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                fig = px.line(
                    daily_metrics, 
                    x='Date', 
                    y='API Calls',
                    title='Daily API Calls',
                    markers=True,
                    height=400
                )
                fig.update_layout(yaxis_title="Number of API Calls")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating time series plots: {e}")

def display_recent_calls(df, n=10):
    """Display detailed information about recent model calls"""
    if df is None or df.empty:
        return
    
    st.subheader("Recent Model Calls")
    
    try:
        # Take most recent n calls
        recent_df = df.head(n)
        
        # Display each call with detailed information
        for _, call in recent_df.iterrows():
            with st.expander(f"{call.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if pd.notna(call.get('timestamp')) else 'Unknown'} - {call.get('model_id', 'Unknown Model')}"):
                # Create two columns for metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Prompt Type:** {call.get('prompt_type', 'Unknown')}")
                    st.markdown(f"**Length Category:** {call.get('length_category', 'Unknown')}")
                    st.markdown(f"**Token Count:** {int(call.get('token_count', 0))}")
                    st.markdown(f"**Latency:** {call.get('latency', 0):.2f}s")
                
                with col2:
                    # Create colored badge for strategy
                    strategy = call.get('strategy', 'unknown')
                    strategy_badge = f'<span class="strategy-badge badge-{strategy.lower()}">{strategy.capitalize()}</span>'
                    st.markdown(f"**Strategy:** {strategy_badge}", unsafe_allow_html=True)
                    
                    st.markdown(f"**Manual Selection:** {'Yes' if call.get('manual_selection', False) else 'No'}")
                    st.markdown(f"**Success:** {'✅' if call.get('success', False) else '❌'}")
                    st.markdown(f"**Cost:** ${call.get('cost', 0):.6f}")
                
                # Show prompt query
                st.markdown("**Prompt:**")
                st.markdown(f"```{call.get('query', '').strip()}```")
                
                # Show routing explanation if available
                routing_explanation = call.get('routing_explanation', {})
                if routing_explanation and isinstance(routing_explanation, dict):
                    explanation_text = routing_explanation.get('explanation', 'No explanation available.')
                    
                    # Format explanation text
                    formatted_explanation = explanation_text.replace("\n", "<br>")
                    
                    st.markdown("**Routing Explanation:**")
                    st.markdown(f'<div class="routing-explanation">{formatted_explanation}</div>', unsafe_allow_html=True)
                
                # Display matched patterns if available
                matched_patterns = call.get('matched_patterns', {})
                if matched_patterns and isinstance(matched_patterns, dict):
                    # Filter to patterns with matches
                    matches = {k: v for k, v in matched_patterns.items() if v > 0}
                    if matches:
                        st.markdown("**Pattern Matches:**")
                        pattern_data = []
                        for pattern, count in matches.items():
                            pattern_data.append({"Pattern Type": pattern, "Matches": count})
                        st.dataframe(pattern_data)
    except Exception as e:
        st.error(f"Error displaying recent calls: {e}")

def sidebar_filters(df):
    """Create sidebar filters for the dashboard"""
    st.sidebar.title("Model Call Analytics")
    
    if df is None or df.empty:
        st.sidebar.warning("No model call data available.")
        return None
    
    st.sidebar.markdown("Use the filters below to analyze model call data:")
    
    # Date range filter
    if 'timestamp' in df.columns:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        # Default to last 7 days if enough data
        default_start = max(min_date, max_date - timedelta(days=7))
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Handle single date selection
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range
        
        # Filter DataFrame by date
        filtered_df = df[(df['timestamp'].dt.date >= start_date) & 
                         (df['timestamp'].dt.date <= end_date)]
    else:
        filtered_df = df
    
    # Model filter
    if 'model_id' in df.columns:
        available_models = sorted(df['model_id'].unique())
        selected_models = st.sidebar.multiselect(
            "Filter by Model",
            options=available_models,
            default=[]
        )
        
        if selected_models:
            filtered_df = filtered_df[filtered_df['model_id'].isin(selected_models)]
    
    # Strategy filter
    if 'strategy' in df.columns:
        available_strategies = sorted(df['strategy'].unique())
        selected_strategies = st.sidebar.multiselect(
            "Filter by Strategy",
            options=available_strategies,
            default=[]
        )
        
        if selected_strategies:
            filtered_df = filtered_df[filtered_df['strategy'].isin(selected_strategies)]
    
    # Prompt type filter
    if 'prompt_type' in df.columns:
        available_types = sorted(df['prompt_type'].unique())
        selected_types = st.sidebar.multiselect(
            "Filter by Prompt Type",
            options=available_types,
            default=[]
        )
        
        if selected_types:
            filtered_df = filtered_df[filtered_df['prompt_type'].isin(selected_types)]
    
    # Success/failure filter
    if 'success' in df.columns:
        success_option = st.sidebar.radio(
            "Filter by Success/Failure",
            options=["All", "Successful Only", "Failed Only"]
        )
        
        if success_option == "Successful Only":
            filtered_df = filtered_df[filtered_df['success'] == True]
        elif success_option == "Failed Only":
            filtered_df = filtered_df[filtered_df['success'] == False]
    
    # Show filter counts
    if len(filtered_df) < len(df):
        st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} model calls ({(len(filtered_df)/len(df)*100):.1f}%)")
    
    return filtered_df

def main():
    """Main function for the Model Call Analytics page"""
    st.title("📊 Model Call Analytics")
    st.markdown(
        "Analyze model call data, routing decisions, and performance metrics. "
        "Use the sidebar to filter the data."
    )
    
    # Load model call data
    df = load_model_call_data()
    
    if df is None or df.empty:
        st.warning("No model call data available. Make some API calls first to generate data.")
        # Show sample route
        st.info(
            "Once you use the chatbot and make some API calls, this dashboard will show detailed analytics "
            "about model usage, routing decisions, and performance metrics."
        )
        return
    
    # Apply sidebar filters
    filtered_df = sidebar_filters(df)
    
    if filtered_df is None or filtered_df.empty:
        st.warning("No data matches the current filters. Try adjusting your filter criteria.")
        return
    
    # Get summary stats from filtered data
    stats = {
        "total_calls": len(filtered_df),
        "successful_calls": len(filtered_df[filtered_df.get('success', True) == True]),
        "total_tokens": filtered_df['token_count'].sum(),
        "total_cost": filtered_df['cost'].sum(),
        "avg_latency": filtered_df['latency'].mean(),
    }
    
    # Display summary metrics
    display_metrics_summary(stats)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Usage", 
        "Routing Strategy", 
        "Prompt Types", 
        "Cost & Token Usage"
    ])
    
    with tab1:
        plot_model_usage(filtered_df)
    
    with tab2:
        plot_strategy_distribution(filtered_df)
    
    with tab3:
        plot_prompt_types(filtered_df)
    
    with tab4:
        plot_cost_and_tokens_over_time(filtered_df)
    
    # Display recent calls with detailed information
    display_recent_calls(filtered_df)
    
    # Add raw data view
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
    
    # Download option
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="model_calls_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main() 