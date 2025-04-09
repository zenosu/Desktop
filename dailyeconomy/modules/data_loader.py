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
        # For the new format, look for CSV files in a directory named after the date
        fred_dir = os.path.join(data_dir, f"FRED_{date_str}")
        logger.info(f"Loading FRED data from directory {fred_dir}")
        
        measurements = {}
        
        # Check if the path is a directory or a file
        if os.path.isdir(fred_dir):
            # It's a directory - load individual CSV files
            csv_files = [f for f in os.listdir(fred_dir) if f.endswith(f'_{date_str}.csv')]
            
            for csv_file in csv_files:
                # Extract measurement ID from filename (e.g., "CPIAUCSL_20250407.csv" -> "CPIAUCSL")
                measurement_id = csv_file.split('_')[0]
                file_path = os.path.join(fred_dir, csv_file)
                
                logger.info(f"Loading measurement {measurement_id} from {file_path}")
                
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    
                    # Standardize column names
                    if 'observation_date' in df.columns and measurement_id in df.columns:
                        df = df.rename(columns={'observation_date': 'date', measurement_id: 'value'})
                    
                    # Convert date to datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Convert value to numeric
                    if 'value' in df.columns:
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    
                    measurements[measurement_id] = df
                    
                except Exception as e:
                    logger.error(f"Error loading measurement {measurement_id}: {e}")
        else:
            # Try the original format - a single file with multiple measurements
            logger.info(f"Directory not found. Trying to load from file {fred_dir}")
            
            if os.path.isfile(fred_dir):
                with open(fred_dir, 'r') as f:
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
            else:
                logger.error(f"Neither directory nor file found for {date_str}")
                
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