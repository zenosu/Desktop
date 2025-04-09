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

# Import utilities
from utils.helpers import setup_logging, validate_date_format, save_metadata

# Import configuration
import config

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
    setup_logging(args.output_dir, log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting daily economy report generation for date: {args.date}")
    
    try:
        # Validate date format
        if not validate_date_format(args.date):
            logger.error(f"Invalid date format: {args.date}. Expected YYYYMMDD.")
            return 1
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create visualization directory
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Check if we're dealing with new format (directory of CSV files) or old format
        fred_dir_path = os.path.join(args.data_dir, f"FRED_{args.date}")
        if os.path.isdir(fred_dir_path):
            logger.info(f"Using new CSV format from directory: {fred_dir_path}")
            data_source = args.data_dir
        else:
            # Check if direct file exists
            fred_file_path = fred_dir_path  # Same path, but expecting a file not dir
            if os.path.isfile(fred_file_path):
                logger.info(f"Using original file format: {fred_file_path}")
                data_source = args.data_dir
            else:
                # If neither exists, use the parent directory as data source
                logger.info(f"Neither directory nor file format found at {fred_dir_path}, using parent directory")
                data_source = os.path.dirname(args.data_dir)
        
        # Load data and context
        logger.info(f"Loading FRED data from {data_source}")
        all_measurements_data = load_fred_data(args.date, data_source)
        
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
        
        # Save metadata
        metadata = {
            'date': args.date,
            'measurements_count': len(processed_data),
            'data_dir': data_source,
            'context_file': args.context_file,
            'output_dir': args.output_dir,
            'recession_risk': {
                'score': recession_indicators.get('overall_risk_score', 0),
                'category': recession_indicators.get('risk_category', 'Medium')
            }
        }
        save_metadata(args.output_dir, args.date, metadata)
        
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