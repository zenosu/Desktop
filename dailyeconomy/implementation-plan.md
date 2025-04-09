# Daily Economy Report Generator - Implementation Plan

## Project Overview

This system processes daily FRED (Federal Reserve Economic Data) measurements to generate a comprehensive economy report with visualizations, trend analyses, and AI-generated insights using Ollama with the deepseek-r1:70b model.

## Project Structure

```
/Users/zeno.su/nn/mcp-tools/daily_economy/
├── config.py                # Configuration settings
├── daily_economy_report.py  # Main script
├── modules/
│   ├── __init__.py
│   ├── data_loader.py       # Data loading functions
│   ├── visualizer.py        # Visualization functions
│   ├── analyzer.py          # Analysis functions
│   ├── llm_insights.py      # LLM integration
│   └── report_generator.py  # Report generation
├── templates/              # HTML/Report templates
│   └── report_template.html
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Helper utilities
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Core Components

### 1. Configuration (`config.py`)

```python
# Paths
DATA_DIR = "/Users/zeno.su/Desktop/dailyeconomy"  # Original data directory
CONTEXT_FILE = f"{DATA_DIR}/fred_context.txt"     # Context file
OUTPUT_DIR = "/Users/zeno.su/nn/mcp-tools/daily_economy/reports"  # Reports output

# LLM Configuration
OLLAMA_MODEL = "deepseek-r1:70b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama API

# Analysis Parameters
RECESSION_RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9
}

# Historical Events
HISTORICAL_EVENTS = {
    "financial_crisis": "2008-09-01",
    "covid_drop": "2020-03-01"
}

# Visualization Settings
VIZ_WIDTH = 10
VIZ_HEIGHT = 6
VIZ_DPI = 100
```

### 2. Data Loading Module (`modules/data_loader.py`)

```python
# data_loader.py
import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def load_fred_data(date_str, data_dir):
    """
    Load FRED data for a specific date.
    
    Args:
        date_str (str): Date string in YYYYMMDD format
        data_dir (str): Directory containing FRED data files
    
    Returns:
        dict: Mapping of measurement_id to pandas DataFrame
    """
    try:
        fred_file = os.path.join(data_dir, f"FRED_{date_str}")
        logger.info(f"Loading FRED data from {fred_file}")
        
        # Assuming the file contains multiple measurements in a structured format
        # This part would need to be adapted based on the actual file format
        measurements = {}
        
        # For CSV-like format with multiple measurements
        with open(fred_file, 'r') as f:
            current_measurement = None
            headers = None
            data_lines = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Assuming a format where measurements are separated by headers
                if line.startswith('#MEASUREMENT:'):
                    # Save previous measurement if exists
                    if current_measurement and headers and data_lines:
                        df = pd.DataFrame([l.split(',') for l in data_lines], columns=headers)
                        measurements[current_measurement] = df
                    
                    # Start new measurement
                    current_measurement = line.split(':')[1].strip()
                    headers = None
                    data_lines = []
                elif line.startswith('#HEADERS:'):
                    headers = line.split(':')[1].strip().split(',')
                elif not line.startswith('#') and headers:
                    data_lines.append(line)
            
            # Save the last measurement
            if current_measurement and headers and data_lines:
                df = pd.DataFrame([l.split(',') for l in data_lines], columns=headers)
                measurements[current_measurement] = df
        
        # Process each measurement DataFrame
        for measurement_id, df in measurements.items():
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Convert value column to float if possible
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
            measurements[measurement_id] = df
                
        return measurements
    
    except Exception as e:
        logger.error(f"Error loading FRED data: {e}")
        return {}

def load_context_data(context_file):
    """
    Load context data from fred_context.txt.
    
    Args:
        context_file (str): Path to fred_context.txt
    
    Returns:
        dict: Mapping of measurement_id to context information
    """
    try:
        logger.info(f"Loading context data from {context_file}")
        context = {}
        
        with open(context_file, 'r') as f:
            current_measurement = None
            current_data = {}
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Assuming each measurement has a section with properties
                if line.startswith('MEASUREMENT:'):
                    # Save previous measurement if exists
                    if current_measurement and current_data:
                        context[current_measurement] = current_data
                    
                    # Start new measurement
                    current_measurement = line.split(':')[1].strip()
                    current_data = {}
                elif ':' in line and current_measurement:
                    key, value = line.split(':', 1)
                    current_data[key.strip()] = value.strip()
            
            # Save the last measurement
            if current_measurement and current_data:
                context[current_measurement] = current_data
        
        return context
    
    except Exception as e:
        logger.error(f"Error loading context data: {e}")
        return {}

def preprocess_measurement(data, measurement_id, context):
    """
    Preprocess a specific measurement DataFrame.
    
    Args:
        data (pd.DataFrame): Raw measurement data
        measurement_id (str): Measurement identifier
        context (dict): Context information for this measurement
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    try:
        df = data.copy()
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        # Handle missing values
        if 'value' in df.columns:
            df['value'] = df['value'].interpolate(method='linear')
        
        # Add year column for easier analysis
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
        
        # Add frequency information from context
        frequency = context.get(measurement_id, {}).get('frequency', 'unknown')
        df['frequency'] = frequency
        
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing measurement {measurement_id}: {e}")
        return data  # Return original data on error
```

### 3. Visualization Module (`modules/visualizer.py`)

```python
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
    Format x-axis to show years regardless of data frequency.
    
    Args:
        ax (matplotlib.axes.Axes): Plot axis
        data (pd.DataFrame): Measurement data
    """
    # Set major locator to years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set minor locator based on data frequency
    if data['date'].nunique() > 50:  # High frequency data (e.g., daily)
        ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))  # Quarters
    else:
        ax.xaxis.set_minor_locator(mdates.MonthLocator())  # Months
    
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
```

### 4. Analysis Module (`modules/analyzer.py`)

```python
# analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def analyze_recent_trend(data, measurement_id, context):
    """
    Analyze recent trend based on the update frequency.
    
    Args:
        data (pd.DataFrame): Processed measurement data
        measurement_id (str): Measurement identifier
        context (dict): Context information for measurements
    
    Returns:
        dict: Trend analysis results
    """
    try:
        # Get measurement context and frequency
        measurement_context = context.get(measurement_id, {})
        frequency = measurement_context.get('frequency', 'monthly').lower()
        
        # Ensure data is sorted by date
        df = data.sort_values('date').copy()
        
        # Get latest data point
        latest_date = df['date'].max()
        latest_point = df[df['date'] == latest_date].iloc[0]
        latest_value = latest_point['value']
        
        # Get previous data point based on frequency
        if frequency == 'daily':
            previous_date = latest_date - timedelta(days=1)
            window_size = 7  # 1 week window for trend
        elif frequency == 'weekly':
            previous_date = latest_date - timedelta(weeks=1)
            window_size = 4  # 4 weeks window for trend
        elif frequency == 'monthly':
            # Approximate previous month
            previous_date = latest_date - timedelta(days=30)
            window_size = 3  # 3 months window for trend
        elif frequency == 'quarterly':
            # Approximate previous quarter
            previous_date = latest_date - timedelta(days=90)
            window_size = 4  # 4 quarters window for trend
        else:  # Default to annual
            previous_date = latest_date - timedelta(days=365)
            window_size = 3  # 3 years window for trend
        
        # Find actual previous data point (closest to calculated previous date)
        previous_idx = (df['date'] - previous_date).abs().idxmin()
        previous_point = df.loc[previous_idx]
        previous_value = previous_point['value']
        
        # Calculate changes
        absolute_change = latest_value - previous_value
        if previous_value != 0:
            percentage_change = (absolute_change / abs(previous_value)) * 100
        else:
            percentage_change = float('inf') if absolute_change > 0 else float('-inf')
        
        # Calculate trend direction
        if absolute_change > 0:
            trend_direction = "increasing"
        elif absolute_change < 0:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calculate recent trend statistics
        recent_window = df.tail(window_size)
        mean_value = recent_window['value'].mean()
        std_dev = recent_window['value'].std()
        
        # Determine if latest value is an outlier
        z_score = (latest_value - mean_value) / std_dev if std_dev != 0 else 0
        is_outlier = abs(z_score) > 2  # Z-score threshold for outlier
        
        # Compile results
        results = {
            'measurement_id': measurement_id,
            'latest_date': latest_date,
            'latest_value': latest_value,
            'previous_date': previous_point['date'],
            'previous_value': previous_value,
            'absolute_change': absolute_change,
            'percentage_change': percentage_change,
            'trend_direction': trend_direction,
            'frequency': frequency,
            'mean_value': mean_value,
            'std_dev': std_dev,
            'z_score': z_score,
            'is_outlier': is_outlier
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error analyzing recent trend for {measurement_id}: {e}")
        return {'measurement_id': measurement_id, 'error': str(e)}

def historical_comparison(data, measurement_id, events=None):
    """
    Compare current trend with historical economic downturns.
    
    Args:
        data (pd.DataFrame): Processed measurement data
        measurement_id (str): Measurement identifier
        events (dict): Historical events with dates
    
    Returns:
        dict: Historical comparison results
    """
    try:
        # Default events if none provided
        if events is None:
            events = {
                '2008_crisis': pd.Timestamp('2008-09-15'),  # Lehman Brothers bankruptcy
                'covid_drop': pd.Timestamp('2020-03-15')  # COVID-19 emergency declaration
            }
        
        # Ensure data is sorted by date
        df = data.sort_values('date').copy()
        
        # Get latest data
        latest_date = df['date'].max()
        latest_value = df[df['date'] == latest_date]['value'].iloc[0]
        
        # Prepare results container
        comparisons = {
            'measurement_id': measurement_id,
            'latest_date': latest_date,
            'latest_value': latest_value,
            'comparisons': {}
        }
        
        # Loop through historical events
        for event_name, event_date in events.items():
            # Check if data covers this event
            if df['date'].min() <= event_date <= df['date'].max():
                # Find the event data point (closest to event date)
                event_idx = (df['date'] - event_date).abs().idxmin()
                event_point = df.loc[event_idx]
                event_value = event_point['value']
                
                # Calculate pre-event baseline (6 months before)
                pre_event_date = event_date - timedelta(days=180)
                pre_event_idx = (df['date'] - pre_event_date).abs().idxmin()
                pre_event_value = df.loc[pre_event_idx]['value']
                
                # Calculate post-event trough (within 1 year after event)
                post_event_window = df[(df['date'] >= event_date) & 
                                      (df['date'] <= event_date + timedelta(days=365))]
                
                if not post_event_window.empty:
                    # Find minimum value after event (the trough)
                    trough_idx = post_event_window['value'].idxmin()
                    trough_point = df.loc[trough_idx]
                    trough_value = trough_point['value']
                    trough_date = trough_point['date']
                    
                    # Calculate recovery period
                    recovery_window = df[(df['date'] >= trough_date) & 
                                        (df['date'] <= latest_date)]
                    
                    # Check if it has recovered to pre-event level
                    recovered_points = recovery_window[recovery_window['value'] >= pre_event_value]
                    has_recovered = not recovered_points.empty
                    
                    if has_recovered:
                        recovery_date = recovered_points.iloc[0]['date']
                        recovery_days = (recovery_date - trough_date).days
                    else:
                        recovery_date = None
                        recovery_days = None
                    
                    # Calculate drop percentages
                    event_drop_pct = ((event_value - pre_event_value) / pre_event_value) * 100
                    trough_drop_pct = ((trough_value - pre_event_value) / pre_event_value) * 100
                    current_vs_pre_pct = ((latest_value - pre_event_value) / pre_event_value) * 100
                    current_vs_trough_pct = ((latest_value - trough_value) / trough_value) * 100
                    
                    # Store comparison results
                    comparisons['comparisons'][event_name] = {
                        'event_date': event_date,
                        'event_value': event_value,
                        'pre_event_date': df.loc[pre_event_idx]['date'],
                        'pre_event_value': pre_event_value,
                        'trough_date': trough_date,
                        'trough_value': trough_value,
                        'event_drop_pct': event_drop_pct,
                        'trough_drop_pct': trough_drop_pct,
                        'has_recovered': has_recovered,
                        'recovery_date': recovery_date,
                        'recovery_days': recovery_days,
                        'current_vs_pre_pct': current_vs_pre_pct,
                        'current_vs_trough_pct': current_vs_trough_pct
                    }
        
        return comparisons
    
    except Exception as e:
        logger.error(f"Error comparing historical events for {measurement_id}: {e}")
        return {'measurement_id': measurement_id, 'error': str(e)}

def calculate_recession_indicators(all_measurements_data, all_analyses, context):
    """
    Calculate recession risk indicators based on all measurements.
    
    Args:
        all_measurements_data (dict): All processed measurement data
        all_analyses (dict): All analysis results
        context (dict): Context information for measurements
    
    Returns:
        dict: Recession risk assessment
    """
    try:
        # Define key recession indicators and their weights
        # These would ideally be calibrated based on historical data
        indicator_weights = {
            'GDP': 0.20,             # Gross Domestic Product
            'UNRATE': 0.15,          # Unemployment Rate
            'PAYEMS': 0.10,          # Nonfarm Payroll
            'INDPRO': 0.10,          # Industrial Production
            'T10Y2Y': 0.10,          # 10Y-2Y Treasury Yield Spread
            'T10Y3M': 0.10,          # 10Y-3M Treasury Yield Spread
            'UMCSENT': 0.05,         # Consumer Sentiment
            'RSAFS': 0.05,           # Retail Sales
            'HOUST': 0.05,           # Housing Starts
            'PERMIT': 0.05,          # Building Permits
            'CPILFESL': 0.05         # Core Inflation
        }
        
        # Initialize risk scores
        risk_scores = {}
        available_indicators = 0
        total_weighted_score = 0
        
        # Calculate individual risk scores for each indicator
        for measurement_id, weight in indicator_weights.items():
            if measurement_id in all_analyses:
                analysis = all_analyses[measurement_id]
                
                # Skip if error in analysis
                if 'error' in analysis:
                    continue
                
                # Different logic for different indicators
                if measurement_id in ['GDP', 'PAYEMS', 'INDPRO', 'UMCSENT', 'RSAFS', 'HOUST', 'PERMIT']:
                    # For these, negative growth is bad
                    if analysis['percentage_change'] < -1.0:
                        indicator_risk = min(1.0, abs(analysis['percentage_change']) / 5.0)
                    else:
                        indicator_risk = max(0.0, -analysis['percentage_change'] / 5.0)
                        
                elif measurement_id in ['UNRATE', 'CPILFESL']:
                    # For these, positive growth is bad
                    if analysis['percentage_change'] > 1.0:
                        indicator_risk = min(1.0, analysis['percentage_change'] / 5.0)
                    else:
                        indicator_risk = max(0.0, analysis['percentage_change'] / 5.0)
                        
                elif measurement_id in ['T10Y2Y', 'T10Y3M']:
                    # For yield curve, negative is bad (inversion)
                    if analysis['latest_value'] < 0:
                        indicator_risk = min(1.0, abs(analysis['latest_value']) * 2)
                    else:
                        indicator_risk = max(0.0, (0.5 - analysis['latest_value']) / 0.5)
                
                # Store individual risk and add to weighted total
                risk_scores[measurement_id] = indicator_risk
                total_weighted_score += indicator_risk * weight
                available_indicators += weight
        
        # Calculate overall risk score (normalized to available indicators)
        if available_indicators > 0:
            overall_risk_score = total_weighted_score / available_indicators
        else:
            overall_risk_score = 0.5  # Default to medium if no indicators available
        
        # Determine risk category
        if overall_risk_score < 0.3:
            risk_category = "Low"
        elif overall_risk_score < 0.6:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        # Compile results
        results = {
            'overall_risk_score': overall_risk_score,
            'risk_category': risk_category,
            'individual_risks': risk_scores,
            'available_indicators': list(risk_scores.keys()),
            'missing_indicators': [m for m in indicator_weights.keys() if m not in risk_scores]
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error calculating recession indicators: {e}")
        return {'error': str(e)}
```

### 5. LLM Integration Module (`modules/llm_insights.py`)

```python
# llm_insights.py
import requests
import json
import logging
import time

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, model_name, api_url="http://localhost:11434/api/generate"):
        """
        Initialize the Ollama client.
        
        Args:
            model_name (str): Name of the Ollama model to use
            api_url (str): URL of the Ollama API
        """
        self.model_name = model_name
        self.api_url = api_url
        
    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=60):
        """
        Generate text from the model.
        
        Args:
            prompt (str): The prompt text
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            timeout (int): Request timeout in seconds
            
        Returns:
            str: Generated text
        """
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(self.api_url, json=data, timeout=timeout)
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Failed to generate response (Status {response.status_code})"
                
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return f"Error: {str(e)}"
        except json.JSONDecodeError:
            logger.error("Error decoding JSON response")
            return "Error: Invalid response format from API"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Error: {str(e)}"

def generate_measurement_insight(measurement_data, analysis_results, historical_comparison, context):
    """
    Generate insight for a specific measurement using LLM.
    
    Args:
        measurement_data (pd.DataFrame): Processed measurement data
        analysis_results (dict): Analysis results for this measurement
        historical_comparison (dict): Historical comparison results
        context (dict): Context information for this measurement
    
    Returns:
        str: Generated insight text
    """
    try:
        # Get measurement context
        measurement_id = analysis_results['measurement_id']
        measurement_context = context.get(measurement_id, {})
        title = measurement_context.get('title', measurement_id)
        description = measurement_context.get('description', '')
        frequency = measurement_context.get('frequency', 'monthly')
        
        # Format the prompt
        prompt = f"""
        You are an expert financial and economic analyst. Analyze the following economic data for {title} and provide concise, insightful analysis:
        
        ## MEASUREMENT INFORMATION
        Measurement: {title}
        ID: {measurement_id}
        Description: {description}
        Update Frequency: {frequency}
        
        ## RECENT TREND ANALYSIS
        Latest Date: {analysis_results.get('latest_date')}
        Latest Value: {analysis_results.get('latest_value')}
        Previous Date: {analysis_results.get('previous_date')}
        Previous Value: {analysis_results.get('previous_value')}
        Change: {analysis_results.get('absolute_change')} ({analysis_results.get('percentage_change'):.2f}%)
        Trend Direction: {analysis_results.get('trend_direction')}
        """
        
        # Add historical comparison information if available
        comparisons = historical_comparison.get('comparisons', {})
        for event_name, comparison in comparisons.items():
            event_title = "2008 Financial Crisis" if event_name == "2008_crisis" else "2020 COVID-19 Pandemic"
            prompt += f"""
            {event_title}:
            - Pre-{event_title} Value: {comparison.get('pre_event_value')} (Date: {comparison.get('pre_event_date')})
            - During {event_title} Value: {comparison.get('event_value')} (Date: {comparison.get('event_date')})
            - Lowest Value: {comparison.get('trough_value')} (Date: {comparison.get('trough_date')})
            - Drop from Pre-Event to Trough: {comparison.get('trough_drop_pct'):.2f}%
            - Current Value vs Pre-Event: {comparison.get('current_vs_pre_pct'):.2f}%
            - Has Fully Recovered: {"Yes" if comparison.get('has_recovered') else "No"}
            """
            if comparison.get('has_recovered') and comparison.get('recovery_days'):
                prompt += f"- Recovery Time: {comparison.get('recovery_days')} days\n"
        
        # Add analysis request
        prompt += """
        ## ANALYSIS REQUEST
        Please provide a concise analysis of this economic data in 2-3 paragraphs covering:
        
        1. The recent trend and its significance, considering the update frequency
        2. Comparison with both the 2008 financial crisis and 2020 COVID-19 pandemic
        3. What this measurement suggests about current economic conditions
        
        Be specific, data-driven, and insightful. Avoid generic statements.
        """
        
        # Create Ollama client and generate insight
        client = OllamaClient(model_name="deepseek-r1:70b")
        insight = client.generate(prompt, max_tokens=1024, temperature=0.7)
        
        return insight
    
    except Exception as e:
        logger.error(f"Error generating insight for {measurement_id}: {e}")
        return f"Error generating insight: {str(e)}"

def generate_overall_summary(all_analyses, recession_indicators, context):
    """
    Generate overall economy summary using LLM.
    
    Args:
        all_analyses (dict): All analysis results
        recession_indicators (dict): Recession risk assessment
        context (dict): Context information for measurements
    
    Returns:
        str: Generated summary text
    """
    try:
        # Extract key measurements and their trends
        key_measurements = {}
        for measurement_id, analysis in all_analyses.items():
            if 'error' not in analysis:
                measurement_context = context.get(measurement_id, {})
                title = measurement_context.get('title', measurement_id)
                key_measurements[title] = {
                    'id': measurement_id,
                    'latest_value': analysis.get('latest_value'),
                    'change_pct': analysis.get('percentage_change'),
                    'trend': analysis.get('trend_direction')
                }
        
        # Format recession risk information
        risk_score = recession_indicators.get('overall_risk_score', 0.5)
        risk_category = recession_indicators.get('risk_category', 'Medium')
        individual_risks = recession_indicators.get('individual_risks', {})
        
        # Format individual risks for the prompt
        risk_details = ""
        for measurement_id, score in individual_risks.items():
            measurement_context = context.get(measurement_id, {})
            title = measurement_context.get('title', measurement_id)
            risk_details += f"- {title} ({measurement_id}): {score:.2f}\n"
        
        # Format the prompt
        prompt = f"""
        You are an expert financial and economic analyst. Provide a comprehensive assessment of the current economic situation and recession risk based on these economic indicators:
        
        ## KEY MEASUREMENTS
        """
        
        # Add key measurements
        for title, info in key_measurements.items():
            prompt += f"- {title} ({info['id']}): {info['latest_value']} ({info['change_pct']:.2f}%, {info['trend']})\n"
        
        # Add recession risk assessment
        prompt += f"""
        ## RECESSION RISK ASSESSMENT
        Overall Risk Score: {risk_score:.2f}
        Risk Category: {risk_category}
        
        Individual Indicator Risks:
        {risk_details}
        
        ## ANALYSIS REQUEST
        Based on the economic data presented, please provide:
        
        1. A comprehensive summary (about 3-4 paragraphs) of the current state of the economy
        2. Analysis of strengths and weaknesses in different sectors
        3. An assessment of the recession risk (High/Medium/Low) with justification
        4. Key indicators to watch closely in the near future
        
        Be specific, data-driven, and insightful. Avoid generic statements.
        """
        
        # Create Ollama client and generate summary
        client = OllamaClient(model_name="deepseek-r1:70b")
        summary = client.generate(prompt, max_tokens=2048, temperature=0.7)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        return f"Error generating summary: {str(e)}"
```

### 6. Report Generation Module (`modules/report_generator.py`)

```python
# report_generator.py
import os
import jinja2
import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_report(date_str, measurements_data, analyses, historical_comparisons, 
                   insights, visualization_paths, recession_indicators, 
                   overall_summary, context, output_dir):
    """
    Generate a complete HTML report.
    
    Args:
        date_str (str): Date string in YYYYMMDD format
        measurements_data (dict): All processed measurement data
        analyses (dict): Analysis results for each measurement
        historical_comparisons (dict): Historical comparison results
        insights (dict): LLM-generated insights for each measurement
        visualization_paths (dict): Paths to visualization images
        recession_indicators (dict): Recession risk assessment
        overall_summary (str): Overall economy summary
        context (dict): Context information for measurements
        output_dir (str): Directory to save the report
    
    Returns:
        str: Path to final report
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Format date for display
        display_date = datetime.datetime.strptime(date_str, '%Y%m%d').strftime('%B %d, %Y')
        
        # Set up Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), '../templates')
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Load template
        template = env.get_template('report_template.html')
        
        # Prepare measurements sections
        measurement_sections = []
        measurement_trend_table = []
        
        # Sort measurements by their risk contribution if available
        if 'individual_risks' in recession_indicators:
            sorted_measurements = sorted(
                analyses.keys(),
                key=lambda m: recession_indicators['individual_risks'].get(m, 0),
                reverse=True
            )
        else:
            sorted_measurements = sorted(analyses.keys())
        
        # Generate measurement sections
        for measurement_id in sorted_measurements:
            if measurement_id in analyses and 'error' not in analyses[measurement_id]:
                section = create_measurement_section(
                    measurement_id,
                    measurements_data.get(measurement_id, {}),
                    analyses.get(measurement_id, {}),
                    historical_comparisons.get(measurement_id, {}),
                    insights.get(measurement_id, "No insight available."),
                    visualization_paths.get(measurement_id),
                    context
                )
                measurement_sections.append(section)
                
                # Add to trend table
                analysis = analyses[measurement_id]
                measurement_context = context.get(measurement_id, {})
                title = measurement_context.get('title', measurement_id)
                
                trend_entry = {
                    'id': measurement_id,
                    'title': title,
                    'latest_value': f"{analysis.get('latest_value', 'N/A'):.2f}",
                    'change': f"{analysis.get('percentage_change', 0):.2f}%",
                    'direction': analysis.get('trend_direction', 'stable'),
                    'date': analysis.get('latest_date', 'N/A').strftime('%Y-%m-%d') if isinstance(analysis.get('latest_date'), datetime.datetime) else 'N/A',
                    'frequency': analysis.get('frequency', 'monthly').capitalize(),
                    'risk': recession_indicators.get('individual_risks', {}).get(measurement_id, 'N/A')
                }
                measurement_trend_table.append(trend_entry)
        
        # Render the template
        report_html = template.render(
            date=display_date,
            report_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            measurement_sections=measurement_sections,
            trend_table=measurement_trend_table,
            recession_risk={
                'score': f"{recession_indicators.get('overall_risk_score', 0):.2f}",
                'category': recession_indicators.get('risk_category', 'Medium'),
                'indicators': len(recession_indicators.get('individual_risks', {}))
            },
            overall_summary=overall_summary
        )
        
        # Save the report
        report_path = os.path.join(output_dir, f"Economy_Report_{date_str}.html")
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Generated report saved to {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def create_measurement_section(measurement_id, data, analysis, historical_comparison, 
                              insight, visualization_path, context):
    """
    Create HTML/markdown for a single measurement section.
    
    Args:
        measurement_id (str): Measurement identifier
        data (pd.DataFrame): Measurement data
        analysis (dict): Analysis results
        historical_comparison (dict): Historical comparison results
        insight (str): LLM-generated insight
        visualization_path (str): Path to visualization image
        context (dict): Context information
    
    Returns:
        dict: Formatted section data
    """
    try:
        # Get measurement context
        measurement_context = context.get(measurement_id, {})
        title = measurement_context.get('title', measurement_id)
        description = measurement_context.get('description', '')
        frequency = measurement_context.get('frequency', 'monthly').capitalize()
        units = measurement_context.get('units', '')
        
        # Format change with appropriate sign
        percentage_change = analysis.get('percentage_change', 0)
        if percentage_change > 0:
            change_formatted = f"+{percentage_change:.2f}%"
        else:
            change_formatted = f"{percentage_change:.2f}%"
        
        # Format historical comparisons
        comparisons = historical_comparison.get('comparisons', {})
        historical_data = []
        
        for event_name, comparison in comparisons.items():
            event_title = "2008 Financial Crisis" if event_name == "2008_crisis" else "2020 COVID-19 Pandemic"
            
            recovery_status = "Fully recovered" if comparison.get('has_recovered') else "Not yet recovered"
            if comparison.get('has_recovered') and comparison.get('recovery_days'):
                recovery_status += f" (took {comparison.get('recovery_days')} days)"
            
            historical_data.append({
                'event': event_title,
                'pre_value': f"{comparison.get('pre_event_value'):.2f}",
                'during_value': f"{comparison.get('event_value'):.2f}",
                'trough_value': f"{comparison.get('trough_value'):.2f}",
                'trough_drop': f"{comparison.get('trough_drop_pct'):.2f}%",
                'current_vs_pre': f"{comparison.get('current_vs_pre_pct'):.2f}%",
                'recovery_status': recovery_status
            })
        
        # Determine visualization path relative to the report
        if visualization_path:
            # Convert to relative path if needed
            viz_filename = os.path.basename(visualization_path)
            vis_path = f"visualizations/{viz_filename}"
        else:
            vis_path = ""
        
        # Create section
        section = {
            'id': measurement_id,
            'title': title,
            'description': description,
            'frequency': frequency,
            'units': units,
            'latest_value': f"{analysis.get('latest_value'):.2f}",
            'previous_value': f"{analysis.get('previous_value'):.2f}",
            'change': change_formatted,
            'trend_direction': analysis.get('trend_direction', 'stable').capitalize(),
            'latest_date': analysis.get('latest_date').strftime('%Y-%m-%d') if isinstance(analysis.get('latest_date'), datetime.datetime) else 'N/A',
            'visualization': vis_path,
            'historical': historical_data,
            'insight': insight
        }
        
        return section
    
    except Exception as e:
        logger.error(f"Error creating section for {measurement_id}: {e}")
        return {
            'id': measurement_id,
            'title': measurement_id,
            'error': f"Error: {str(e)}"
        }
```

### 7. Main Script (`daily_economy_report.py`)

```python
#!/usr/bin/env python3
# daily_economy_report.py
import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
import shutil

# Import modules
from modules.data_loader import load_fred_data, load_context_data, preprocess_measurement
from modules.visualizer import create_line_graph
from modules.analyzer import analyze_recent_trend, historical_comparison, calculate_recession_indicators
from modules.llm_insights import generate_measurement_insight, generate_overall_summary
from modules.report_generator import generate_report

# Import configuration
import config

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.OUTPUT_DIR, 'economy_report.log'))
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a daily economic report from FRED data.')
    
    parser.add_argument('--date', type=str, 
                        default=datetime.datetime.now().strftime('%Y%m%d'),
                        help='Date in YYYYMMDD format for the report (default: today)')
    
    parser.add_argument('--data-dir', type=str,
                        default=config.DATA_DIR,
                        help=f'Directory containing FRED data (default: {config.DATA_DIR})')
    
    parser.add_argument('--context-file', type=str,
                        default=config.CONTEXT_FILE,
                        help=f'Path to the context file (default: {config.CONTEXT_FILE})')
    
    parser.add_argument('--output-dir', type=str,
                        default=config.OUTPUT_DIR,
                        help=f'Directory to save the report (default: {config.OUTPUT_DIR})')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def main():
    """Main function to generate the economic report."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting daily economy report generation for date: {args.date}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualization directory
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Load data and context
        logger.info(f"Loading FRED data from {args.data_dir}")
        all_measurements_data = load_fred_data(args.date, args.data_dir)
        
        if not all_measurements_data:
            logger.error(f"No FRED data found for date {args.date}")
            return 1
        
        logger.info(f"Loading context information from {args.context_file}")
        context = load_context_data(args.context_file)
        
        if not context:
            logger.error(f"No context information found in {args.context_file}")
            return 1
        
        # Process each measurement
        processed_data = {}
        analyses = {}
        historical_comparisons = {}
        insights = {}
        visualization_paths = {}
        
        logger.info(f"Processing {len(all_measurements_data)} measurements")
        for measurement_id, data in all_measurements_data.items():
            logger.info(f"Processing measurement: {measurement_id}")
            
            # Preprocess
            processed_data[measurement_id] = preprocess_measurement(data, measurement_id, context)
            
            # Analyze
            analyses[measurement_id] = analyze_recent_trend(
                processed_data[measurement_id], measurement_id, context
            )
            
            # Historical comparison
            historical_comparisons[measurement_id] = historical_comparison(
                processed_data[measurement_id], measurement_id
            )
            
            # Visualization
            visualization_paths[measurement_id] = create_line_graph(
                processed_data[measurement_id], measurement_id, context, viz_dir
            )
            
            # Generate insight
            insights[measurement_id] = generate_measurement_insight(
                processed_data[measurement_id],
                analyses[measurement_id],
                historical_comparisons[measurement_id],
                context
            )
        
        # Calculate recession indicators
        logger.info("Calculating recession risk indicators")
        recession_indicators = calculate_recession_indicators(
            processed_data, analyses, context
        )
        
        # Generate overall summary
        logger.info("Generating overall economic summary")
        overall_summary = generate_overall_summary(
            analyses, recession_indicators, context
        )
        
        # Generate report
        logger.info("Generating final report")
        report_path = generate_report(
            args.date,
            processed_data,
            analyses,
            historical_comparisons,
            insights,
            visualization_paths,
            recession_indicators,
            overall_summary,
            context,
            args.output_dir
        )
        
        if report_path:
            logger.info(f"Report successfully generated: {report_path}")
            print(f"\nReport successfully generated: {report_path}")
            return 0
        else:
            logger.error("Failed to generate report")
            return 1
    
    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 8. HTML Template (`templates/report_template.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Economy Report - {{ date }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 20px;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .report-date {
            color: #7f8c8d;
            font-style: italic;
        }
        .summary-section {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            border-left: 5px solid #3498db;
        }
        .risk-indicator {
            display: flex;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .risk-low {
            background-color: #d5f5e3;
            border-left: 5px solid #2ecc71;
        }
        .risk-medium {
            background-color: #fef9e7;
            border-left: 5px solid #f1c40f;
        }
        .risk-high {
            background-color: #fadbd8;
            border-left: 5px solid #e74c3c;
        }
        .risk-value {
            font-size: 24px;
            font-weight: bold;
            margin-right: 20px;
        }
        .risk-details {
            flex-grow: 1;
        }
        .overview-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        .overview-table th, .overview-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .overview-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .trend-up {
            color: #2ecc71;
        }
        .trend-down {
            color: #e74c3c;
        }
        .trend-stable {
            color: #3498db;
        }
        .measurement-section {
            margin-bottom: 50px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .measurement-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .measurement-title {
            margin: 0;
            color: #2c3e50;
        }
        .measurement-value {
            font-size: 18px;
            font-weight: bold;
        }
        .measurement-visualization {
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            display: block;
        }
        .historical-comparison {
            margin-top: 20px;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
        .historical-table {
            width: 100%;
            border-collapse: collapse;
        }
        .historical-table th, .historical-table td {
            padding: 8px 10px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        .insight-box {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fc;
            border-radius: 5px;
            border-left: 5px solid #3498db;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <header>
        <h1>Daily Economy Report</h1>
        <h2>{{ date }}</h2>
        <p class="report-date">Generated on {{ report_date }}</p>
    </header>
    
    <section class="summary-section">
        <h2>Economic Overview</h2>
        
        <div class="risk-indicator risk-{{ recession_risk.category|lower }}">
            <div class="risk-value">
                {{ recession_risk.category }} Risk
                <div style="font-size: 14px; font-weight: normal;">({{ recession_risk.score }})</div>
            </div>
            <div class="risk-details">
                <p>Based on analysis of {{ recession_risk.indicators }} economic indicators.</p>
            </div>
        </div>
        
        <div class="summary-content">
            {{ overall_summary|safe }}
        </div>
    </section>
    
    <h2>Economic Indicators Overview</h2>
    <table class="overview-table">
        <thead>
            <tr>
                <th>Indicator</th>
                <th>Latest Value</th>
                <th>Change</th>
                <th>Date</th>
                <th>Frequency</th>
                <th>Risk Score</th>
            </tr>
        </thead>
        <tbody>
            {% for item in trend_table %}
            <tr>
                <td>{{ item.title }} ({{ item.id }})</td>
                <td>{{ item.latest_value }}</td>
                <td class="trend-{{ item.direction }}">{{ item.change }}</td>
                <td>{{ item.date }}</td>
                <td>{{ item.frequency }}</td>
                <td>{{ item.risk }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <h2>Detailed Analysis</h2>
    
    {% for section in measurement_sections %}
    <div class="measurement-section">
        <div class="measurement-header">
            <h3 class="measurement-title">{{ section.title }} ({{ section.id }})</h3>
            <div class="measurement-value">
                {{ section.latest_value }} {{ section.units }}
                <span class="trend-{{ section.trend_direction|lower }}">{{ section.change }}</span>
            </div>
        </div>
        
        <p>{{ section.description }}</p>
        <p><strong>Update Frequency:</strong> {{ section.frequency }}</p>
        <p><strong>Latest Date:</strong> {{ section.latest_date }}</p>
        
        {% if section.visualization %}
        <img src="{{ section.visualization }}" alt="{{ section.title }} Trend" class="measurement-visualization">
        {% endif %}
        
        <div class="insight-box">
            <h4>Analysis & Insight</h4>
            {{ section.insight|safe }}
        </div>
        
        {% if section.historical %}
        <div class="historical-comparison">
            <h4>Historical Comparison</h4>
            <table class="historical-table">
                <thead>
                    <tr>
                        <th>Event</th>
                        <th>Pre-Event</th>
                        <th>During Event</th>
                        <th>Lowest Point</th>
                        <th>Drop %</th>
                        <th>Current vs Pre</th>
                        <th>Recovery Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for event in section.historical %}
                    <tr>
                        <td>{{ event.event }}</td>
                        <td>{{ event.pre_value }}</td>
                        <td>{{ event.during_value }}</td>
                        <td>{{ event.trough_value }}</td>
                        <td>{{ event.trough_drop }}</td>
                        <td>{{ event.current_vs_pre }}</td>
                        <td>{{ event.recovery_status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
    {% endfor %}
    
    <footer>
        <p>This report is generated automatically using FRED economic data.</p>
    </footer>
</body>
</html>
```

### 9. Dependencies (`requirements.txt`)

```
pandas>=2.0.0
numpy>=1.22.0
matplotlib>=3.5.0
seaborn>=0.12.0
jinja2>=3.1.0
requests>=2.28.0
python-dotenv>=1.0.0
```

## Process Flow

1. **Data Loading**:
   - Read FRED data files with measurements
   - Parse context information for each measurement
   - Preprocess measurement data

2. **Analysis**:
   - Analyze recent trends based on measurement frequency
   - Compare with historical economic downturns (2008, 2020)
   - Calculate recession risk indicators

3. **Visualization**:
   - Create line graphs for each measurement
   - Highlight recent data points and historical events
   - Format axes to display years regardless of data frequency

4. **Insight Generation**:
   - Use Ollama with deepseek-r1:70b to generate insights for each measurement
   - Create an overall economy summary with recession risk assessment
   - Format prompts with relevant context and analysis results

5. **Report Generation**:
   - Compile all analyses, visualizations, and insights
   - Generate a comprehensive HTML report
   - Include tables and comparisons for each measurement

## Usage Instructions

1. **Setup**:
   ```bash
   # Create the project structure
   mkdir -p /Users/zeno.su/nn/mcp-tools/daily_economy/{modules,templates,utils,reports}
   
   # Create the necessary files (as listed in the implementation)
   
   # Make the main script executable
   chmod +x /Users/zeno.su/nn/mcp-tools/daily_economy/daily_economy_report.py
   
   # Install dependencies
   pip install -r /Users/zeno.su/nn/mcp-tools/daily_economy/requirements.txt
   ```

2. **Running the Report Generator**:
   ```bash
   cd /Users/zeno.su/nn/mcp-tools/daily_economy
   
   # Generate a report for the latest data
   ./daily_economy_report.py --date 20250407
   
   # With verbose logging
   ./daily_economy_report.py --date 20250407 --verbose
   
   # Custom paths
   ./daily_economy_report.py --date 20250407 --data-dir /path/to/data --context-file /path/to/context.txt --output-dir /path/to/output
   ```

3. **Output**:
   - HTML report at `/Users/zeno.su/nn/mcp-tools/daily_economy/reports/Economy_Report_20250407.html`
   - Visualizations in `/Users/zeno.su/nn/mcp-tools/daily_economy/reports/visualizations/`
   - Log file at `/Users/zeno.su/nn/mcp-tools/daily_economy/reports/economy_report.log`
