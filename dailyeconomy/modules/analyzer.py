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
    Compare current change trends with historical economic downturns (2008 crisis and 2020 COVID).
    Analyzes rate and direction of change to identify pattern similarities.
    
    Args:
        data (pd.DataFrame): Processed measurement data
        measurement_id (str): Measurement identifier
        events (dict): Historical events with dates
    
    Returns:
        dict: Historical comparison results including trend similarity analysis
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

        # Get data from 3 months ago for current trend calculation
        months_3_ago = latest_date - timedelta(days=90)
        months_3_ago_idx = (df['date'] - months_3_ago).abs().idxmin()
        months_3_ago_value = df.loc[months_3_ago_idx]['value']
        
        # Calculate current 3-month change
        current_3month_change = ((latest_value - months_3_ago_value) / months_3_ago_value) * 100
        
        # Add a 6-month lookback as well
        months_6_ago = latest_date - timedelta(days=180)
        months_6_ago_idx = (df['date'] - months_6_ago).abs().idxmin()
        months_6_ago_value = df.loc[months_6_ago_idx]['value']
        
        # Calculate current 6-month change
        current_6month_change = ((latest_value - months_6_ago_value) / months_6_ago_value) * 100
        
        # Prepare results container
        comparisons = {
            'measurement_id': measurement_id,
            'latest_date': latest_date,
            'latest_value': latest_value,
            'current_trend': {
                '3month_change_pct': current_3month_change,
                '6month_change_pct': current_6month_change,
                '3month_date': df.loc[months_3_ago_idx]['date'],
                '6month_date': df.loc[months_6_ago_idx]['date'],
                '3month_value': months_3_ago_value,
                '6month_value': months_6_ago_value
            },
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
                
                # Calculate 3-month pre-event value
                pre_event_3m_date = event_date - timedelta(days=90)
                pre_event_3m_idx = (df['date'] - pre_event_3m_date).abs().idxmin()
                pre_event_3m_value = df.loc[pre_event_3m_idx]['value']
                
                # Calculate post-event trough (within 1 year after event)
                post_event_window = df[(df['date'] >= event_date) & 
                                       (df['date'] <= event_date + timedelta(days=365))]
                
                if not post_event_window.empty:
                    # Find minimum value after event (the trough)
                    trough_idx = post_event_window['value'].idxmin()
                    trough_point = df.loc[trough_idx]
                    trough_value = trough_point['value']
                    trough_date = trough_point['date']
                    
                    # Calculate change rates during the event period
                    event_3month_change = ((event_value - pre_event_3m_value) / pre_event_3m_value) * 100
                    event_6month_change = ((event_value - pre_event_value) / pre_event_value) * 100
                    event_to_trough_change = ((trough_value - event_value) / event_value) * 100
                    pre_to_trough_change = ((trough_value - pre_event_value) / pre_event_value) * 100
                    
                    # Calculate the time it took to reach the trough
                    days_to_trough = (trough_date - event_date).days
                    
                    # Calculate additional time points for more detailed trend analysis
                    post_1m_date = event_date + timedelta(days=30)
                    post_3m_date = event_date + timedelta(days=90)
                    
                    # Find closest data points to these dates
                    post_1m_idx = (df['date'] - post_1m_date).abs().idxmin()
                    post_3m_idx = (df['date'] - post_3m_date).abs().idxmin()
                    
                    post_1m_value = df.loc[post_1m_idx]['value']
                    post_3m_value = df.loc[post_3m_idx]['value']
                    
                    # Calculate change rates for these periods
                    event_to_1m_change = ((post_1m_value - event_value) / event_value) * 100
                    event_to_3m_change = ((post_3m_value - event_value) / event_value) * 100
                    
                    # Analyze trend similarity with current situation
                    # Compare direction and magnitude of change
                    
                    # 3-month change similarity (direction)
                    same_3m_direction = (current_3month_change > 0) == (event_3month_change > 0)
                    
                    # 6-month change similarity (direction)
                    same_6m_direction = (current_6month_change > 0) == (event_6month_change > 0)
                    
                    # Magnitude similarity (how close are the percentage changes)
                    # Lower values mean more similar
                    magnitude_3m_diff = abs(current_3month_change - event_3month_change)
                    magnitude_6m_diff = abs(current_6month_change - event_6month_change)
                    
                    # Calculate an overall similarity score (0-100, higher means more similar)
                    direction_score = (same_3m_direction * 25) + (same_6m_direction * 25)
                    
                    # Magnitude score - higher when differences are smaller
                    # Scale from 0-50 based on percentage difference
                    magnitude_3m_score = max(0, 25 - min(25, magnitude_3m_diff / 2))
                    magnitude_6m_score = max(0, 25 - min(25, magnitude_6m_diff / 2))
                    
                    similarity_score = direction_score + magnitude_3m_score + magnitude_6m_score
                    
                    # Interpret the similarity
                    if similarity_score >= 75:
                        similarity_assessment = "Very similar trend patterns"
                    elif similarity_score >= 50:
                        similarity_assessment = "Moderately similar trend patterns"
                    elif similarity_score >= 25:
                        similarity_assessment = "Somewhat similar trend patterns"
                    else:
                        similarity_assessment = "Different trend patterns"
                    
                    # Store comparison results with focus on trend analysis
                    comparisons['comparisons'][event_name] = {
                        'event_date': event_date,
                        'event_value': event_value,
                        'pre_event_date': df.loc[pre_event_idx]['date'],
                        'pre_event_value': pre_event_value,
                        'pre_event_3m_date': df.loc[pre_event_3m_idx]['date'],
                        'pre_event_3m_value': pre_event_3m_value,
                        'trough_date': trough_date,
                        'trough_value': trough_value,
                        'event_3month_change': event_3month_change,
                        'event_6month_change': event_6month_change,
                        'event_to_trough_change': event_to_trough_change,
                        'pre_to_trough_change': pre_to_trough_change,
                        'days_to_trough': days_to_trough,
                        'post_1m_change': event_to_1m_change,
                        'post_3m_change': event_to_3m_change,
                        'similarity': {
                            'score': similarity_score,
                            'assessment': similarity_assessment,
                            'same_3m_direction': same_3m_direction,
                            'same_6m_direction': same_6m_direction,
                            'magnitude_3m_diff': magnitude_3m_diff,
                            'magnitude_6m_diff': magnitude_6m_diff
                        }
                    }
        
        return comparisons
    
    except Exception as e:
        logger.error(f"Error comparing historical events for {measurement_id}: {e}")
        return {'measurement_id': measurement_id, 'error': str(e)}

def calculate_recession_indicators(all_measurements_data, all_analyses, context):
    """
    Calculate recession risk indicators based on all available measurements.
    Assigns categorical risk levels (Low, Medium, High) directly for each indicator.
    Only calculates risk for data points from the past 14 days.
    
    Args:
        all_measurements_data (dict): All processed measurement data
        all_analyses (dict): All analysis results
        context (dict): Context information for measurements
    
    Returns:
        dict: Recession risk assessment
    """
    try:
        # Initialize risk scores with categorical values
        risk_scores = {}
        risk_points = {'Low': 0, 'Medium': 1, 'High': 2}  # Points for weighted calculation
        total_weighted_points = 0
        available_indicators = 0
        
        # Define weights based on specification
        weights = {
            'NFCI': 3.0,
            'T10Y2Y': 3.0,
            'DGS2': 2.0
        }
        
        # Calculate cutoff date (14 days before current date)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=14)
        
        # Calculate individual risk scores for each available indicator
        for measurement_id, analysis in all_analyses.items():
            # Skip if error in analysis
            if 'error' in analysis:
                continue
                
            # Skip if data is older than 14 days
            latest_date = analysis.get('latest_date')
            if latest_date is None or not isinstance(latest_date, (pd.Timestamp, datetime)) or latest_date < cutoff_date:
                logger.info(f"Skipping risk calculation for {measurement_id}: data from {latest_date} is before cutoff date {cutoff_date}")
                continue
            
            # Get latest value for assessment
            latest_value = analysis.get('latest_value')
            
            # Determine weight (default to 1.0 if not specified)
            weight = weights.get(measurement_id, 1.0)
            
            # Assign categorical risk based on new criteria
            if measurement_id == 'NFCI':
                # NFCI: less than -0.35 is low, -0.35 to 0 is medium, greater than 0 is high
                if latest_value < -0.35:
                    indicator_risk = 'Low'
                elif latest_value <= 0:
                    indicator_risk = 'Medium'
                else:
                    indicator_risk = 'High'
            
            elif measurement_id == 'T10Y2Y':
                # T10Y2Y: greater than 25bps is low, 25 to -75bps is medium, lower than -75bps is high
                if latest_value > 0.25:
                    indicator_risk = 'Low'
                elif latest_value >= -0.75:
                    indicator_risk = 'Medium'
                else:
                    indicator_risk = 'High'
            
            elif measurement_id == 'DGS2':
                # DGS2: less than 3% is low, 3-5% is medium, greater than 5% is high
                if latest_value < 3.0:
                    indicator_risk = 'Low'
                elif latest_value <= 5.0:
                    indicator_risk = 'Medium'
                else:
                    indicator_risk = 'High'
            
            else:
                # For other measurements, use the percentage change approach
                percentage_change = analysis.get('percentage_change', 0)
                
                # Get measurement context
                measurement_context = context.get(measurement_id, {})
                direction = measurement_context.get('recession_direction', None)
                
                # Determine direction if not defined
                if direction is None:
                    if measurement_id in ['NFCI']:
                        direction = 'positive'
                    elif measurement_id in ['T10Y2Y']:
                        direction = 'negative'
                    elif measurement_id in ['ICSA']:
                        direction = 'positive'
                    elif measurement_id in ['RSAFS']:
                        direction = 'negative'
                    elif measurement_id in ['CPIAUCSL']:
                        direction = 'positive'
                    elif measurement_id in ['DGS10']:
                        direction = 'positive'
                    elif measurement_id in ['CSUSHPINSA']:
                        direction = 'negative'
                    else:
                        direction = 'negative'
                
                # Calculate risk based on direction and percentage change
                if direction == 'negative':
                    if percentage_change < -3.0:
                        indicator_risk = 'High'
                    elif percentage_change < -1.0:
                        indicator_risk = 'Medium'
                    else:
                        indicator_risk = 'Low'
                else:  # direction == 'positive'
                    if percentage_change > 3.0:
                        indicator_risk = 'High'
                    elif percentage_change > 1.0:
                        indicator_risk = 'Medium'
                    else:
                        indicator_risk = 'Low'
            
            # Store categorical risk
            risk_scores[measurement_id] = indicator_risk
            
            # Calculate weighted points for overall risk determination
            risk_point_value = risk_points.get(indicator_risk, 1)  # Default to Medium (1) if unknown
            total_weighted_points += risk_point_value * weight
            available_indicators += weight
        
        # Calculate overall risk category
        if available_indicators > 0:
            average_points = total_weighted_points / available_indicators
            
            # Determine overall risk category
            if average_points < 0.67:
                risk_category = "Low"
            elif average_points < 1.33:
                risk_category = "Medium"
            else:
                risk_category = "High"
        else:
            risk_category = "Medium"  # Default to medium if no indicators available
        
        # Compile results
        results = {
            'overall_points': round(total_weighted_points / available_indicators, 2) if available_indicators > 0 else 1.0,
            'risk_category': risk_category,
            'individual_risks': risk_scores,
            'available_indicators': list(risk_scores.keys()),
            'missing_indicators': [],
            'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
            'weights': weights
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Error calculating recession indicators: {e}")
        return {'error': str(e)} 