import os
import requests
import datetime

def download_fred_data(series_id, target_dir):
    today_date = datetime.datetime.now().strftime('%Y%m%d')
    new_filename = f"{series_id}_{today_date}.csv"
    target_file = os.path.join(target_dir, new_filename)
    
    # Direct download URL for CSV
    download_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    
    try:
        # Make the request
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Save the content to the target file
        with open(target_file, 'wb') as f:
            f.write(response.content)
        
        print(f"Successfully downloaded and saved to: {target_file}")
        return True
        
    except Exception as e:
        print(f"Error downloading {series_id}: {str(e)}")
        return False

def main():
    # List of series IDs to download
    series_ids = [
        "NFCI", "ICSA", "RSAFS", "CPIAUCSL", 
        "DGS2", "DGS10", "T10Y2Y", "CSUSHPINSA"
    ]
    
    # Get today's date for folder creation
    today_date = datetime.datetime.now().strftime('%Y%m%d')
    folder_name = f"FRED_{today_date}"
    
    # Setup target directory
    home_dir = os.path.expanduser("~")
    base_dir = os.path.join(home_dir, "Desktop", "dailyeconomy")
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create date-specific folder
    target_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created folder: {target_dir}")
    
    # Process each series ID
    results = []
    for series_id in series_ids:
        success = download_fred_data(series_id, target_dir)
        results.append((series_id, success))
    
    # Print summary
    print("\nDownload Summary:")
    for series_id, success in results:
        status = "Success" if success else "Failed"
        print(f"{series_id}: {status}")
    
    # Check if all downloads were successful
    all_success = all(success for _, success in results)
    print(f"\nAll downloads {'successful' if all_success else 'not successful'}")
    print(f"Files saved in: {target_dir}")

if __name__ == "__main__":
    main()