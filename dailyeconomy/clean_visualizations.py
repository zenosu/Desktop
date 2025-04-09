#!/usr/bin/env python3
# clean_visualizations.py
import os
import sys
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def get_available_measurements(fred_folder):
    """Get measurement IDs from CSV files in the FRED folder."""
    try:
        if not os.path.isdir(fred_folder):
            logger.error(f"FRED folder {fred_folder} not found")
            return []
            
        # Extract measurement IDs from filenames
        csv_files = [f for f in os.listdir(fred_folder) if f.endswith('.csv')]
        measurements = [f.split('_')[0] for f in csv_files]
        logger.info(f"Found {len(measurements)} measurements: {', '.join(measurements)}")
        return measurements
    except Exception as e:
        logger.error(f"Error getting measurements: {e}")
        return []

def clean_visualizations(visualizations_dir, valid_measurements):
    """Remove visualization files that don't match valid measurements."""
    try:
        if not os.path.isdir(visualizations_dir):
            logger.error(f"Visualizations directory {visualizations_dir} not found")
            return False
            
        # Get all png files in the directory
        png_files = [f for f in os.listdir(visualizations_dir) if f.endswith('.png')]
        
        # Identify files to remove
        files_to_remove = []
        for png_file in png_files:
            # Extract measurement ID from filename (e.g., "GDP_trend.png" -> "GDP")
            measurement_id = png_file.split('_')[0]
            if measurement_id not in valid_measurements:
                files_to_remove.append(png_file)
        
        # Remove files
        for file_name in files_to_remove:
            file_path = os.path.join(visualizations_dir, file_name)
            logger.info(f"Removing obsolete visualization: {file_name}")
            os.remove(file_path)
            
        logger.info(f"Removed {len(files_to_remove)} obsolete visualization files")
        return True
    except Exception as e:
        logger.error(f"Error cleaning visualizations: {e}")
        return False

def main():
    """Main function."""
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fred_folder = os.path.join(base_dir, "FRED_20250407")
    visualizations_dir = os.path.join(base_dir, "reports", "visualizations")
    
    # Get valid measurements
    measurements = get_available_measurements(fred_folder)
    if not measurements:
        logger.error("No measurements found, exiting")
        return 1
        
    # Clean visualizations
    success = clean_visualizations(visualizations_dir, measurements)
    if not success:
        logger.error("Failed to clean visualizations")
        return 1
        
    logger.info("Visualization cleanup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 