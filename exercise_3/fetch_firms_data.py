import os
import requests
import pandas as pd
import numpy as np

def fetch_or_mock_firms_data(output_file='hotspots.csv'):
    api_key = os.environ.get('FIRMS_MAP_KEY')
    
    if api_key:
        # Request data for a specific bounding box (e.g., California region)
        bbox = "-120,30,-110,40" 
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{bbox}/1"
        print(f"Fetching FIRMS data from API: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_file, 'w') as f:
                f.write(response.text)
            print("Data saved to", output_file)
            return
        except Exception as e:
            print(f"Failed to fetch data: {e}. Falling back to synthetic data.")
    else:
        print("FIRMS_MAP_KEY environment variable not found.")
        print("Generating synthetic hotspot data for testing purposes.")
        
    # Generate synthetic hotspot data to ensure the simulation can be tested
    np.random.seed(42)
    # Simulate a region from lat 30 to 40, lon -120 to -110
    num_hotspots = 100
    lats = np.random.uniform(30.5, 39.5, num_hotspots)
    lons = np.random.uniform(-119.5, -110.5, num_hotspots)
    brightness = np.random.uniform(300, 400, num_hotspots)
    
    df = pd.DataFrame({'latitude': lats, 'longitude': lons, 'brightness': brightness})
    df.to_csv(output_file, index=False)
    print("Synthetic data saved to", output_file)

if __name__ == '__main__':
    fetch_or_mock_firms_data()
