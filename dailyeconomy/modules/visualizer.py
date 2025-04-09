# visualizer.py
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set Seaborn style
sns.set_style('whitegrid')
logger = logging.getLogger(__name__)

def create_line_graph(data, measurement_id, context, output_dir):
    """
    Create a line graph for a specific measurement.
    
    Args:
        data (pd.DataFrame): Processed measurement data
        measurement_id (str): Measurement identifier
        context (dict): Context information for this measurement
        output_dir (str): Directory to save visualization
    
    Returns:
        str: Path to saved visualization
    """
    try:
        # Get measurement context
        measurement_context = context.get(measurement_id, {})
        title = measurement_context.get('title', measurement_id)
        description = measurement_context.get('description', '')
        frequency = measurement_context.get('frequency', 'unknown')
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Plot the line
        sns.lineplot(data=data, x='date', y='value', ax=ax, linewidth=2)
        
        # Format x-axis to show years regardless of data frequency
        format_x_axis_years(ax, data)
        
        # Highlight the most recent data point
        highlight_latest_point(ax, data)
        
        # Highlight historical events
        highlight_historical_events(ax, data)
        
        # Set title and labels
        ax.set_title(f"{title}", fontsize=16, pad=20)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(measurement_context.get('units', 'Value'), fontsize=12)
        
        # Add description as a subtitle if available
        if description:
            plt.figtext(0.5, 0.01, description, wrap=True, 
                       horizontalalignment='center', fontsize=10)
        
        # Add frequency information
        freq_text = f"Update Frequency: {frequency.capitalize()}"
        plt.figtext(0.02, 0.02, freq_text, fontsize=9, style='italic')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{measurement_id}_trend.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved visualization for {measurement_id} to {filepath}")
        return filepath
    
    except Exception as e:
        logger.error(f"Error creating visualization for {measurement_id}: {e}")
        return None

def format_x_axis_years(ax, data):
    """
    Format x-axis to show only decade years (e.g., 1990, 2000, 2010) for a cleaner visualization.
    
    Args:
        ax (matplotlib.axes.Axes): Plot axis
        data (pd.DataFrame): Measurement data
    """
    # Use built-in YearLocator with base=10 to show only decade years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set minor locator based on data frequency (keep this for grid lines)
    if data['date'].nunique() > 50:  # High frequency data (e.g., daily)
        ax.xaxis.set_minor_locator(mdates.YearLocator(2))  # Every two years
    else:
        ax.xaxis.set_minor_locator(mdates.YearLocator())  # Every year
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Ensure x-axis spans the entire data range
    ax.set_xlim(data['date'].min(), data['date'].max())

def highlight_latest_point(ax, data):
    """
    Highlight the most recent data point and show its value.
    
    Args:
        ax (matplotlib.axes.Axes): Plot axis
        data (pd.DataFrame): Measurement data
    """
    # Get the most recent data point
    latest_data = data.loc[data['date'].idxmax()]
    latest_date = latest_data['date']
    latest_value = latest_data['value']
    
    # Highlight point
    ax.scatter(latest_date, latest_value, color='red', s=80, zorder=5, 
               label=f'Latest: {latest_value:.2f}')
    
    # Add text label
    ax.annotate(f'{latest_value:.2f}', 
                xy=(latest_date, latest_value),
                xytext=(10, 0), textcoords='offset points',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
    
    # Add date information
    latest_date_str = latest_date.strftime('%b %d, %Y')
    ax.annotate(f'Date: {latest_date_str}', 
                xy=(latest_date, latest_value),
                xytext=(10, -15), textcoords='offset points',
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))

def highlight_historical_events(ax, data):
    """
    Highlight important historical events (2008 crisis, 2020 COVID).
    
    Args:
        ax (matplotlib.axes.Axes): Plot axis
        data (pd.DataFrame): Measurement data
    """
    # Define historical events
    events = {
        '2008 Financial Crisis': pd.Timestamp('2008-09-15'),  # Lehman Brothers bankruptcy
        '2020 COVID-19': pd.Timestamp('2020-03-15')  # COVID-19 declared national emergency
    }
    
    # Check if data covers these periods
    for event_name, event_date in events.items():
        # Find closest data point to the event
        if data['date'].min() <= event_date <= data['date'].max():
            closest_idx = (data['date'] - event_date).abs().idxmin()
            event_point = data.loc[closest_idx]
            
            # Highlight point
            ax.axvline(x=event_date, color='gray', linestyle='--', alpha=0.7)
            
            # Add text label
            ax.annotate(event_name, 
                        xy=(event_date, ax.get_ylim()[1] * 0.95),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', rotation=90,
                        fontsize=9, alpha=0.7) 