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
        
    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=180, retries=2, retry_delay=5):
        """
        Generate text from the model.
        
        Args:
            prompt (str): The prompt text
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            timeout (int): Request timeout in seconds
            retries (int): Number of retry attempts if the request fails
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            str: Generated text
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False  # Explicitly set stream to False
        }
        
        # Check if Ollama service is available before making request
        try:
            health_check = requests.get("http://localhost:11434/api/health", timeout=5)
            if health_check.status_code != 200:
                logger.error(f"Ollama service is not healthy: {health_check.status_code}")
                return f"Error: Ollama service is not available (Status {health_check.status_code})"
        except requests.RequestException as e:
            logger.error(f"Ollama service is not available: {e}")
            return "Error: Ollama service is not available. Please ensure Ollama is running with the required model."
        
        # Implement retry logic
        attempt = 0
        last_error = None
        
        while attempt <= retries:
            try:
                logger.info(f"Sending request to Ollama API (attempt {attempt+1}/{retries+1})")
                
                response = requests.post(self.api_url, json=data, timeout=timeout)
                
                if response.status_code == 200:
                    # Log the full response for debugging
                    logger.debug(f"API response: {response.text[:500]}...")
                    try:
                        # Use a more defensive approach to extract the response text
                        response_json = response.json()
                        if isinstance(response_json, dict):
                            return response_json.get('response', '')
                        else:
                            logger.error(f"Unexpected response format: {response_json}")
                            return "Error: Unexpected response format from API"
                    except json.JSONDecodeError as json_err:
                        logger.error(f"JSON decode error: {json_err} - Raw response: {response.text[:500]}")
                        # Try to extract response from the text if JSON parsing fails
                        if "response" in response.text:
                            try:
                                # Try to extract just the first JSON object
                                first_json_end = response.text.find("}\n")
                                if first_json_end > 0:
                                    first_json = response.text[:first_json_end+1]
                                    return json.loads(first_json).get('response', '')
                            except Exception:
                                pass
                        return "Error: Invalid response format from API"
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    last_error = f"Failed to generate response (Status {response.status_code})"
                    
            except requests.RequestException as e:
                logger.error(f"Request error (attempt {attempt+1}): {e}")
                last_error = str(e)
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt+1}): {e}")
                last_error = str(e)
            
            # If we're here, the request failed
            attempt += 1
            if attempt <= retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            
        # If we're here, all retries failed
        return f"Error: All attempts failed - {last_error}"

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
        
        # Safely format values with null checks
        latest_date = analysis_results.get('latest_date', 'N/A')
        latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        
        previous_date = analysis_results.get('previous_date', 'N/A')
        previous_date_str = previous_date.strftime('%Y-%m-%d') if hasattr(previous_date, 'strftime') else str(previous_date)
        
        latest_value = analysis_results.get('latest_value', 'N/A')
        previous_value = analysis_results.get('previous_value', 'N/A')
        absolute_change = analysis_results.get('absolute_change', 'N/A')
        
        # Handle percentage change carefully
        percentage_change = analysis_results.get('percentage_change', 0)
        percentage_change_str = f"{percentage_change:.2f}%" if percentage_change is not None else "N/A"
        
        trend_direction = analysis_results.get('trend_direction', 'stable')
        
        # Format the prompt
        prompt = f"""
        You are an expert financial and economic analyst. Analyze the following economic data for {title} and provide concise, insightful analysis:
        
        ## MEASUREMENT INFORMATION
        Measurement: {title}
        ID: {measurement_id}
        Description: {description}
        Update Frequency: {frequency}
        
        ## RECENT TREND ANALYSIS
        Latest Date: {latest_date_str}
        Latest Value: {latest_value}
        Previous Date: {previous_date_str}
        Previous Value: {previous_value}
        Change: {absolute_change} ({percentage_change_str})
        Trend Direction: {trend_direction}
        """
        
        # Add historical comparison information if available
        current_trend = historical_comparison.get('current_trend', {})
        if current_trend:
            prompt += f"""
            ## CURRENT TREND
            3-Month Change: {current_trend.get('3month_change_pct', 'N/A'):.2f}%
            6-Month Change: {current_trend.get('6month_change_pct', 'N/A'):.2f}%
            """
            
        # Add trend similarity analysis for historical events
        comparisons = historical_comparison.get('comparisons', {})
        if comparisons:
            prompt += "\n## HISTORICAL COMPARISONS\n"
            
        for event_name, comparison in comparisons.items():
            event_title = "2008 Financial Crisis" if event_name == "2008_crisis" else "2020 COVID-19 Pandemic"
            
            # Get dates and safely format them
            pre_event_date = comparison.get('pre_event_date', 'N/A')
            pre_event_date_str = pre_event_date.strftime('%Y-%m-%d') if hasattr(pre_event_date, 'strftime') else str(pre_event_date)
            
            event_date = comparison.get('event_date', 'N/A')
            event_date_str = event_date.strftime('%Y-%m-%d') if hasattr(event_date, 'strftime') else str(event_date)
            
            trough_date = comparison.get('trough_date', 'N/A')
            trough_date_str = trough_date.strftime('%Y-%m-%d') if hasattr(trough_date, 'strftime') else str(trough_date)
            
            # Get similarity metrics
            similarity = comparison.get('similarity', {})
            similarity_score = similarity.get('score', 'N/A')
            similarity_assessment = similarity.get('assessment', 'Unknown similarity')
            
            # Safe format for all numerical values
            pre_event_value = comparison.get('pre_event_value', 'N/A')
            event_value = comparison.get('event_value', 'N/A')
            trough_value = comparison.get('trough_value', 'N/A')
            event_3month_change = comparison.get('event_3month_change', 'N/A')
            event_6month_change = comparison.get('event_6month_change', 'N/A')
            event_to_trough_change = comparison.get('event_to_trough_change', 'N/A')
            
            # Format the numerical values correctly
            event_3month_change_str = f"{event_3month_change:.2f}%" if isinstance(event_3month_change, (int, float)) else str(event_3month_change)
            event_6month_change_str = f"{event_6month_change:.2f}%" if isinstance(event_6month_change, (int, float)) else str(event_6month_change)
            event_to_trough_change_str = f"{event_to_trough_change:.2f}%" if isinstance(event_to_trough_change, (int, float)) else str(event_to_trough_change)
            similarity_score_str = f"{similarity_score:.1f}/100" if isinstance(similarity_score, (int, float)) else str(similarity_score)
            
            prompt += f"""
            {event_title}:
            - Pre-Event Value: {pre_event_value} (Date: {pre_event_date_str})
            - During Event Value: {event_value} (Date: {event_date_str})
            - Lowest Value: {trough_value} (Date: {trough_date_str})
            - 3-Month Change: {event_3month_change_str}
            - 6-Month Change: {event_6month_change_str}
            - Event to Trough Change: {event_to_trough_change_str}
            - Similarity to Current Trend: {similarity_score_str} - {similarity_assessment}
            """
        
        # Add analysis request
        prompt += """
        ## ANALYSIS REQUEST
        Please provide a concise analysis of this economic data in 2-3 paragraphs covering:
        
        1. The recent trend and its significance, considering the update frequency
        2. How the current trend compares to the historical crisis periods (2008 & 2020)
        3. What this measurement suggests about current economic conditions
        
        Be specific, data-driven, and insightful. Avoid generic statements.
        """
        
        # Create Ollama client and generate insight
        client = OllamaClient(model_name="deepseek-r1:70b")
        insight = client.generate(prompt, max_tokens=1024, temperature=0.7)
        
        # Check if the insight contains an error message
        if insight.startswith("Error:"):
            logger.warning(f"Using fallback insight for {measurement_id} due to LLM error: {insight}")
            return generate_fallback_insight(measurement_id, analysis_results, historical_comparison, context)
        
        return insight
    
    except Exception as e:
        # Get measurement_id in a safe way, in case it wasn't assigned in the try block
        measurement_id = analysis_results.get('measurement_id', 'unknown')
        logger.error(f"Error generating insight for {measurement_id}: {e}")
        return generate_fallback_insight(measurement_id, analysis_results, historical_comparison, context)

def generate_fallback_insight(measurement_id, analysis_results, historical_comparison, context):
    """
    Generate a fallback insight when the LLM is unavailable.
    
    Args:
        measurement_id (str): Measurement identifier
        analysis_results (dict): Analysis results for this measurement
        historical_comparison (dict): Historical comparison results
        context (dict): Context information for this measurement
    
    Returns:
        str: Fallback insight text
    """
    try:
        # Get measurement context
        measurement_context = context.get(measurement_id, {})
        title = measurement_context.get('title', measurement_id)
        
        # Get recent trend information
        latest_value = analysis_results.get('latest_value', 'N/A')
        percentage_change = analysis_results.get('percentage_change', 0)
        trend_direction = analysis_results.get('trend_direction', 'stable')
        
        # Format a basic insight based on the trend
        if trend_direction == 'increasing':
            trend_text = f"The {title} has shown an increase of {percentage_change:.2f}% recently."
        elif trend_direction == 'decreasing':
            trend_text = f"The {title} has shown a decrease of {abs(percentage_change):.2f}% recently."
        else:
            trend_text = f"The {title} has remained relatively stable recently."
        
        # Add current trend information if available
        current_trend_text = ""
        if 'current_trend' in historical_comparison:
            ct = historical_comparison['current_trend']
            change_3m = ct.get('3month_change_pct', None)
            change_6m = ct.get('6month_change_pct', None)
            
            if change_3m is not None and change_6m is not None:
                current_trend_text = f"\nOver the past 3 months, it has changed by {change_3m:.2f}%, and over 6 months by {change_6m:.2f}%."
        
        # Add historical comparison if available
        historical_text = ""
        comparisons = historical_comparison.get('comparisons', {})
        
        for event_name, comparison in comparisons.items():
            event_title = "2008 Financial Crisis" if event_name == "2008_crisis" else "2020 COVID-19 Pandemic"
            
            # Get similarity assessment
            similarity = comparison.get('similarity', {})
            similarity_score = similarity.get('score', 0)
            similarity_assessment = similarity.get('assessment', 'Unknown similarity')
            
            # Create comparison text
            if similarity_score >= 75:
                compare_text = f"The current trend shows very strong similarities to the {event_title} period."
            elif similarity_score >= 50:
                compare_text = f"The current trend shows moderate similarities to the {event_title} period."
            elif similarity_score >= 25:
                compare_text = f"The current trend shows some similarities to the {event_title} period."
            else:
                compare_text = f"The current trend is quite different from the {event_title} period."
                
            # Add change rate comparisons
            event_3m_change = comparison.get('event_3month_change', None)
            current_3m_change = historical_comparison.get('current_trend', {}).get('3month_change_pct', None)
            
            if event_3m_change is not None and current_3m_change is not None:
                if (event_3m_change > 0) == (current_3m_change > 0):
                    direction_text = "same direction"
                else:
                    direction_text = "opposite direction"
                    
                compare_text += f" The 3-month change rates are moving in the {direction_text}."
            
            historical_text += compare_text + " "
        
        # Combine the insights
        fallback_insight = f"""
        {trend_text}{current_trend_text}
        
        {historical_text}
        
        This data should be monitored closely for changes that could indicate shifting economic conditions.
        """
        
        return fallback_insight
    
    except Exception as e:
        logger.error(f"Error generating fallback insight for {measurement_id}: {e}")
        return f"Unable to generate insight for {measurement_id} due to an error. Latest value: {analysis_results.get('latest_value', 'N/A')}"

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
        risk_points = recession_indicators.get('overall_points', 1.0)
        risk_category = recession_indicators.get('risk_category', 'Medium')
        individual_risks = recession_indicators.get('individual_risks', {})
        
        # Format individual risks for the prompt
        risk_details = ""
        for measurement_id, risk_level in individual_risks.items():
            measurement_context = context.get(measurement_id, {})
            title = measurement_context.get('title', measurement_id)
            risk_details += f"- {title} ({measurement_id}): {risk_level}\n"
        
        # Format the prompt
        prompt = f"""
        You are an expert financial and economic analyst. Generate a comprehensive summary of the current economic situation based on the following data:
        
        ## RECESSION RISK ASSESSMENT
        Overall Risk Category: {risk_category}
        Weighted Points: {risk_points}
        
        Individual Indicator Risks:
        {risk_details}
        
        ## KEY MEASUREMENTS
        """
        
        # Add key measurements to the prompt
        for title, details in key_measurements.items():
            trend_symbol = "↑" if details['trend'] == 'increasing' else "↓" if details['trend'] == 'decreasing' else "→"
            prompt += f"- {title} ({details['id']}): {details['latest_value']:.2f} ({details['change_pct']:.2f}% {trend_symbol})\n"
        
        # Add analysis request
        prompt += """
        ## ANALYSIS REQUEST
        Please provide a concise economic summary in 2-3 paragraphs covering:
        
        1. The overall economic situation based on these indicators
        2. The key areas of strength or concern
        3. The recession risk assessment and what it suggests about the economy
        
        Be specific, data-driven, and insightful. Avoid vague or generic statements.
        """
        
        # Create Ollama client and generate summary
        client = OllamaClient(model_name="deepseek-r1:70b")
        summary = client.generate(prompt, max_tokens=1536, temperature=0.7)
        
        # Check if the summary contains an error message
        if summary.startswith("Error:"):
            logger.warning(f"Using fallback summary due to LLM error: {summary}")
            return generate_fallback_summary(all_analyses, recession_indicators, context)
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        return generate_fallback_summary(all_analyses, recession_indicators, context)

def generate_fallback_summary(all_analyses, recession_indicators, context):
    """
    Generate a fallback summary when the LLM is unavailable.
    
    Args:
        all_analyses (dict): All analysis results
        recession_indicators (dict): Recession risk assessment
        context (dict): Context information for measurements
    
    Returns:
        str: Fallback summary text
    """
    try:
        # Get recession risk information
        risk_points = recession_indicators.get('overall_points', 1.0)
        risk_category = recession_indicators.get('risk_category', 'Medium')
        
        # Count trends by direction
        trends = {'increasing': 0, 'decreasing': 0, 'stable': 0}
        for analysis in all_analyses.values():
            if 'error' not in analysis:
                trend = analysis.get('trend_direction', 'stable')
                trends[trend] = trends.get(trend, 0) + 1
        
        # Generate a basic summary
        summary = f"""
        # Economic Summary
        
        Based on the analyzed indicators, the economy currently shows a **{risk_category} Risk** level (weighted points: {risk_points}). 
        
        Of the measured indicators, {trends['increasing']} are increasing, {trends['decreasing']} are decreasing, and {trends['stable']} are stable. 
        """
        
        # Add information about specific indicators
        key_indicators = []
        for measurement_id, analysis in all_analyses.items():
            if 'error' not in analysis and measurement_id in context:
                measurement_context = context.get(measurement_id, {})
                title = measurement_context.get('title', measurement_id)
                value = analysis.get('latest_value', 'N/A')
                change = analysis.get('percentage_change', 0)
                
                if abs(change) > 2.0:  # Only highlight significant changes
                    direction = "increased" if change > 0 else "decreased"
                    key_indicators.append(f"The {title} has {direction} by {abs(change):.2f}% to {value:.2f}.")
        
        if key_indicators:
            summary += "\n\n**Key Indicator Changes:**\n\n"
            summary += "\n".join(f"- {indicator}" for indicator in key_indicators[:3])
        
        summary += "\n\nNote: This is an automated summary based on the available data. For more detailed insights, please ensure the LLM service is properly configured and running."
        
        return summary.strip()
    
    except Exception as e:
        logger.error(f"Error generating fallback summary: {e}")
        return "Economic summary temporarily unavailable. Please check system logs for details." 