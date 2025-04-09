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
        
        # Gather data update information
        data_update_info = []
        
        # Sort measurements by their risk contribution if available
        if 'individual_risks' in recession_indicators:
            # Map risk levels to numeric values for sorting
            risk_level_to_value = {'High': 3, 'Medium': 2, 'Low': 1, 'N/A': 0}
            
            sorted_measurements = sorted(
                analyses.keys(),
                key=lambda m: risk_level_to_value.get(recession_indicators['individual_risks'].get(m, 'N/A'), 0),
                reverse=True
            )
        else:
            sorted_measurements = sorted(analyses.keys())
        
        # Generate measurement sections
        for measurement_id in sorted_measurements:
            if measurement_id in analyses and 'error' not in analyses[measurement_id]:
                analysis = analyses[measurement_id]
                measurement_context = context.get(measurement_id, {})
                title = measurement_context.get('title', measurement_id)
                frequency = measurement_context.get('frequency', 'monthly').capitalize()
                
                # Add to data update info
                latest_date = analysis.get('latest_date')
                if isinstance(latest_date, datetime.datetime):
                    data_update_info.append({
                        'id': measurement_id,
                        'title': title,
                        'date': latest_date.strftime('%Y-%m-%d'),
                        'frequency': frequency
                    })
                
                section = create_measurement_section(
                    measurement_id,
                    measurements_data.get(measurement_id, {}),
                    analysis,
                    historical_comparisons.get(measurement_id, {}),
                    insights.get(measurement_id, "No insight available."),
                    visualization_paths.get(measurement_id),
                    context
                )
                measurement_sections.append(section)
                
                # Add to trend table
                # Determine change explanation based on measurement type
                change_explanation = explain_change_meaning(measurement_id, analysis.get('percentage_change', 0))
                
                # Determine risk calculation explanation
                risk_explanation = explain_risk_calculation(measurement_id)
                
                trend_entry = {
                    'id': measurement_id,
                    'title': title,
                    'latest_value': f"{analysis.get('latest_value', 'N/A'):.2f}",
                    'change': f"{analysis.get('percentage_change', 0):.2f}%",
                    'change_explanation': change_explanation,
                    'direction': analysis.get('trend_direction', 'stable'),
                    'date': analysis.get('latest_date', 'N/A').strftime('%Y-%m-%d') if isinstance(analysis.get('latest_date'), datetime.datetime) else 'N/A',
                    'frequency': frequency,
                    'risk': recession_indicators.get('individual_risks', {}).get(measurement_id, 'N/A'),
                    'risk_explanation': risk_explanation
                }
                measurement_trend_table.append(trend_entry)
        
        # Render the template
        report_html = template.render(
            date=display_date,
            report_date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            measurement_sections=measurement_sections,
            trend_table=measurement_trend_table,
            recession_risk={
                'points': recession_indicators.get('overall_points', 1.0),
                'category': recession_indicators.get('risk_category', 'Medium'),
                'indicators': len(recession_indicators.get('individual_risks', {})),
                'cutoff_date': recession_indicators.get('cutoff_date', 'Unknown'),
                'weights': recession_indicators.get('weights', {})
            },
            overall_summary=overall_summary,
            data_update_info=data_update_info
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
        
        # Format current trend data
        current_trend = {}
        if 'current_trend' in historical_comparison:
            ct = historical_comparison['current_trend']
            current_trend = {
                '3month_change': f"{ct.get('3month_change_pct', 0):.2f}%",
                '6month_change': f"{ct.get('6month_change_pct', 0):.2f}%",
                '3month_date': ct.get('3month_date').strftime('%Y-%m-%d') if isinstance(ct.get('3month_date'), datetime.datetime) else 'N/A',
                '6month_date': ct.get('6month_date').strftime('%Y-%m-%d') if isinstance(ct.get('6month_date'), datetime.datetime) else 'N/A'
            }
        
        # Format historical comparisons with focus on trend similarity
        comparisons = historical_comparison.get('comparisons', {})
        historical_data = []
        
        for event_name, comparison in comparisons.items():
            event_title = "2008 Financial Crisis" if event_name == "2008_crisis" else "2020 COVID-19 Pandemic"
            
            # Get similarity assessment
            similarity = comparison.get('similarity', {})
            similarity_score = similarity.get('score', 0)
            similarity_assessment = similarity.get('assessment', 'Unknown similarity')
            
            # Format trend comparison data
            historical_data.append({
                'event': event_title,
                'pre_value': f"{comparison.get('pre_event_value'):.2f}",
                'during_value': f"{comparison.get('event_value'):.2f}",
                'trough_value': f"{comparison.get('trough_value'):.2f}",
                'event_3m_change': f"{comparison.get('event_3month_change'):.2f}%",
                'event_6m_change': f"{comparison.get('event_6month_change'):.2f}%",
                'to_trough_change': f"{comparison.get('event_to_trough_change'):.2f}%",
                'days_to_trough': comparison.get('days_to_trough', 'N/A'),
                'similarity_score': f"{similarity_score:.1f}/100",
                'similarity_assessment': similarity_assessment,
                'same_direction_3m': "Yes" if similarity.get('same_3m_direction', False) else "No",
                'same_direction_6m': "Yes" if similarity.get('same_6m_direction', False) else "No"
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
            'current_trend': current_trend,
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

def explain_change_meaning(measurement_id, percentage_change):
    """
    Explain what the percentage change means for a specific measurement.
    
    Args:
        measurement_id (str): The measurement identifier
        percentage_change (float): The percentage change value
    
    Returns:
        str: Explanation of what the change means
    """
    # Round percentage for display
    rounded_change = round(percentage_change, 2)
    change_direction = "increase" if percentage_change > 0 else "decrease"
    
    explanations = {
        'NFCI': f"A {rounded_change}% {change_direction} in the National Financial Conditions Index. Higher values indicate tighter financial conditions.",
        'ICSA': f"A {rounded_change}% {change_direction} in initial unemployment claims, indicating {'worsening' if percentage_change > 0 else 'improving'} job market conditions.",
        'RSAFS': f"A {rounded_change}% {change_direction} in retail sales, reflecting {'stronger' if percentage_change > 0 else 'weaker'} consumer spending.",
        'CPIAUCSL': f"A {rounded_change}% {change_direction} in consumer price index, indicating {'rising' if percentage_change > 0 else 'falling'} inflation.",
        'DGS2': f"A {rounded_change}% {change_direction} in 2-year Treasury yield, reflecting changing interest rate expectations.",
        'DGS10': f"A {rounded_change}% {change_direction} in 10-year Treasury yield, affecting long-term borrowing costs.",
        'T10Y2Y': f"A {rounded_change}% {change_direction} in the 10-year minus 2-year Treasury yield spread. Negative values indicate yield curve inversion, a recession signal.",
        'CSUSHPINSA': f"A {rounded_change}% {change_direction} in the Case-Shiller Home Price Index, reflecting {'appreciation' if percentage_change > 0 else 'depreciation'} in housing values."
    }
    
    # Default explanation if measurement ID not found
    return explanations.get(
        measurement_id, 
        f"A {rounded_change}% {change_direction} from previous period."
    )

def explain_risk_calculation(measurement_id):
    """
    Explain how risk is calculated for a specific measurement.
    
    Args:
        measurement_id (str): The measurement identifier
    
    Returns:
        str: Explanation of risk calculation
    """
    risk_explanations = {
        'NFCI': "Less than -0.35 is low risk, -0.35 to 0 is medium risk, greater than 0 is high risk. (Weight: 3)",
        'T10Y2Y': "Greater than 25bps is low risk, 25 to -75bps is medium risk, less than -75bps is high risk. (Weight: 3)",
        'DGS2': "Less than 3% is low risk, 3-5% is medium risk, greater than 5% is high risk. (Weight: 2)",
        'ICSA': "Risk rises with increasing unemployment claims. (Weight: 1)",
        'RSAFS': "Risk rises with declining retail sales. (Weight: 1)",
        'CPIAUCSL': "Risk rises with increasing inflation. (Weight: 1)",
        'DGS10': "Risk rises with rising long-term interest rates. (Weight: 1)",
        'CSUSHPINSA': "Risk rises with declining home prices. (Weight: 1)"
    }
    
    # Default explanation if measurement ID not found
    return risk_explanations.get(
        measurement_id, 
        "Risk based on historical patterns of this indicator. (Weight: 1)"
    ) 