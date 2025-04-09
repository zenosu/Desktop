# helpers.py
import os
import logging
import datetime
import json
from pathlib import Path

def setup_logging(output_dir, log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        output_dir (str): Directory for log files
        log_level (int): Logging level
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, 'economy_report.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Suppress overly verbose loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def save_metadata(output_dir, date_str, metadata):
    """
    Save processing metadata for tracking and troubleshooting.
    
    Args:
        output_dir (str): Directory to save metadata
        date_str (str): Date string in YYYYMMDD format
        metadata (dict): Metadata to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_file = os.path.join(output_dir, f"metadata_{date_str}.json")
    
    # Add timestamp
    metadata['timestamp'] = datetime.datetime.now().isoformat()
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def validate_date_format(date_str):
    """
    Validate date string format (YYYYMMDD).
    
    Args:
        date_str (str): Date string to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        datetime.datetime.strptime(date_str, '%Y%m%d')
        return True
    except ValueError:
        return False

def get_cached_data_path(date_str, cache_dir):
    """
    Get path for cached processed data.
    
    Args:
        date_str (str): Date string in YYYYMMDD format
        cache_dir (str): Cache directory
    
    Returns:
        str: Path to cached data file
    """
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"processed_data_{date_str}.json")

def check_cached_data(date_str, cache_dir):
    """
    Check if cached processed data exists.
    
    Args:
        date_str (str): Date string in YYYYMMDD format
        cache_dir (str): Cache directory
    
    Returns:
        bool: True if cache exists, False otherwise
    """
    cache_path = get_cached_data_path(date_str, cache_dir)
    return os.path.exists(cache_path)

def format_timestamp(timestamp):
    """
    Format timestamp for display.
    
    Args:
        timestamp (datetime): Timestamp to format
    
    Returns:
        str: Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S") 