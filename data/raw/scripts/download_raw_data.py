"""
Antarctic Glacier Boundary Image Downloader - MULTISPECTRAL DATA VERSION
Downloads full multispectral satellite imagery (all key bands).
This data is UNPROCESSED and saved as Int16.
It is NOT VISIBLE in standard image viewers but is required for ML.
"""

import ee
import os
import time
from pathlib import Path
import requests


# Initialize Earth Engine
try:
    ee.Initialize(project = 'glacier-probe-model-475519')
except:
    ee.Authenticate()
    ee.Initialize(project = 'glacier-probe-model-475519')


# Configuration
# ← MODIFIED: Changed output directory
OUTPUT_DIR = 'antarctic_glacier_dataset_multispectral'
NUM_IMAGES = 1000
SCALE = 30  # Resolution in meters

# Antarctic Glacier Boundary Regions (VERY SMALL - centered on calving fronts)
# Coordinates: [lon_min, lat_min, lon_max, lat_max]
# Reduced to 0.3-0.5 degree tiles to avoid edge artifacts
ANTARCTIC_REGIONS = {
    # Larsen Ice Shelf (smaller, centered tiles)
    'Larsen_C_North': [-61.2, -67.3, -60.7, -67.0],
    'Larsen_C_Central': [-61.2, -67.8, -60.7, -67.5],
    'Larsen_C_South': [-61.2, -68.3, -60.7, -68.0],
    'Larsen_B_Remnant': [-59.2, -65.5, -58.8, -65.2],
    
    # Wilkins Ice Shelf
    'Wilkins_North': [-73.7, -69.8, -73.2, -69.5],
    'Wilkins_South': [-73.7, -70.3, -73.2, -70.0],
    
    # George VI Ice Shelf
    'George_VI_North': [-69.2, -71.3, -68.7, -71.0],
    'George_VI_South': [-69.2, -71.8, -68.7, -71.5],
    
    # Pine Island Glacier (CRITICAL - very small focused area)
    'Pine_Island_Front': [-101.2, -75.0, -100.7, -74.7],
    'Pine_Island_East': [-100.7, -74.9, -100.2, -74.6],
    
    # Thwaites Glacier (CRITICAL)
    'Thwaites_Front': [-106.2, -75.0, -105.7, -74.7],
    'Thwaites_East': [-105.7, -74.9, -105.2, -74.6],
    
    # Dotson Ice Shelf
    'Dotson_Front': [-113.2, -74.1, -112.7, -73.8],
    
    # Crosson Ice Shelf
    'Crosson_Front': [-109.2, -74.1, -108.7, -73.8],
    
    # Ross Ice Shelf (smaller centered tiles)
    'Ross_North_West': [171.5, -77.5, 172.0, -77.2],
    'Ross_North_East': [173.5, -77.5, 174.0, -77.2],
    'Ross_East_Front': [-173.5, -78.3, -173.0, -78.0],
    'McMurdo_Sound': [164.5, -77.5, 165.0, -77.2],
    
    # Filchner-Ronne Ice Shelf
    'Filchner_West': [-40.5, -78.0, -40.0, -77.7],
    'Filchner_East': [-39.0, -78.0, -38.5, -77.7],
    'Ronne_North': [-58.5, -78.5, -58.0, -78.2],
    
    # Totten Glacier (East Antarctica)
    'Totten_Front': [116.8, -66.5, 117.3, -66.2],
    'Totten_Shelf': [117.3, -66.5, 117.8, -66.2],
    
    # Moscow University Ice Shelf
    'Moscow_Uni_Front': [93.8, -66.5, 94.3, -66.2],
    
    # Shackleton Ice Shelf
    'Shackleton_Front': [95.8, -65.5, 96.3, -65.2],
    
    # Mertz Glacier
    'Mertz_Front': [144.8, -67.0, 145.3, -66.7],
    
    # Ninnis Glacier
    'Ninnis_Front': [147.8, -68.0, 148.3, -67.7],
    
    # Cook Ice Shelf
    'Cook_Front': [150.8, -69.0, 151.3, -68.7],
    
    # Fimbul Ice Shelf
    'Fimbul_Front': [-0.5, -70.5, 0.0, -70.2],
    
    # Riiser-Larsen Ice Shelf
    'Riiser_Larsen': [11.5, -72.5, 12.0, -72.2],
}

def create_output_dirs():
    """Create output directory structure"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    # ← MODIFIED: Changed subfolder names for clarity
    Path(f"{OUTPUT_DIR}/multispectral_images").mkdir(exist_ok=True)
    Path(f"{OUTPUT_DIR}/metadata").mkdir(exist_ok=True)
    print(f"✓ Created output directories in {OUTPUT_DIR}/")

def get_satellite_collection(start_date, end_date, roi, max_cloud_cover=80):
    """Get Sentinel-2 or Landsat collection"""
    # Sentinel-2 (better coverage)
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
    
    # Landsat 8/9
    landsat89 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
    
    s2_count = sentinel2.size().getInfo()
    l8_count = landsat89.size().getInfo()
    
    if s2_count > 0:
        return sentinel2, 'sentinel2'
    elif l8_count > 0:
        return landsat89, 'landsat'
    else:
        return None, None

def download_raw_image(image, region, filename, satellite_type):
    """
    # ← MODIFIED: This function now downloads RAW, UNPROCESSED multispectral data.
    # It saves the data as Int16, not 8-bit.
    # This data is NOT viewable in standard viewers, but is correct for ML.
    """
    try:
        if satellite_type == 'sentinel2':
            # ← MODIFIED: Select all 10m and 20m bands
            bands_to_select = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            raw_image = image.select(bands_to_select)
            
            # ← MODIFIED: REMOVED all processing (.divide, .clamp, .toByte)
            # We want the original 0-10000 range, which will be saved as Int16
            processed = raw_image.toInt16() # Keep original values
        
        else: # 'landsat'
            # ← MODIFIED: Select all SR bands (B1-B7)
            bands_to_select = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            raw_image = image.select(bands_to_select)
            
            # ← MODIFIED: REMOVED all processing and scaling.
            # We will download the original scaled Int16 data from Collection 2
            processed = raw_image.toInt16() # Keep original values

        # Get download URL
        url = processed.getDownloadURL({
            'region': region,
            'scale': SCALE,
            'format': 'GEO_TIFF',
            'crs': 'EPSG:3031'
        })
        
        # Download file
        response = requests.get(url, timeout=300)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"✗ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if 'too large' in error_msg.lower() or 'size' in error_msg.lower():
            print(f"✗ Too large (region size issue)")
        else:
            print(f"✗ Error: {error_msg[:50]}")
        return False

def save_metadata(image, filename, region_name, satellite_type):
    """Save image metadata"""
    try:
        props = image.getInfo()['properties']
        metadata_file = filename.replace('.tif', '_metadata.txt')
        
        with open(metadata_file, 'w') as f:
            f.write(f"Region: {region_name}\n")
            f.write(f"Satellite: {satellite_type.upper()}\n")
            # ← MODIFIED: Updated image type
            f.write(f"Image Type: MULTISPECTRAL (Scientific Data)\n") 
            
            if satellite_type == 'sentinel2':
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                f.write(f"Date: {date}\n")
                f.write(f"Cloud Cover: {props.get('CLOUDY_PIXEL_PERCENTAGE', 'N/A')}%\n")
                f.write(f"Product ID: {props.get('PRODUCT_ID', 'N/A')}\n")
                # ← MODIFIED: Updated band list
                f.write(f"Bands: ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']\n")
                # ← MODIFIED: Updated values and processing
                f.write(f"Values: Raw (0-10000 range)\n")
                f.write(f"Processing: None. Saved as Int16.\n")
            else:
                f.write(f"Date: {props.get('DATE_ACQUIRED', 'N/A')}\n")
                f.write(f"Cloud Cover: {props.get('CLOUD_COVER', 'N/A')}%\n")
                f.write(f"Scene ID: {props.get('LANDSAT_SCENE_ID', 'N/A')}\n")
                # ← MODIFIED: Updated band list
                f.write(f"Bands: ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']\n")
                # ← MODIFIED: Updated values and processing
                f.write(f"Values: Raw (Landsat Collection 2 scaled integers)\n")
                f.write(f"Processing: None. Saved as Int16.\n")
            
    except Exception as e:
        print(f"  ⚠ Metadata warning: {str(e)[:30]}")

def main():
    """Main download function"""
    print("=" * 70)
    # ← MODIFIED: Updated title
    print("ANTARCTIC GLACIER IMAGE DOWNLOADER - MULTISPECTRAL DATA")
    print("Downloads full, unprocessed scientific data (Int16)")
    print("=" * 70)
    
    create_output_dirs()
    
    # Antarctic summer months - EXPANDED date range
    year_ranges = [
        ('2014-11-01', '2015-03-31'),
        ('2015-11-01', '2016-03-31'),
        ('2016-11-01', '2017-03-31'),
        ('2017-11-01', '2018-03-31'),
        ('2018-11-01', '2019-03-31'),
        ('2019-11-01', '2020-03-31'),
        ('2020-11-01', '2021-03-31'),
        ('2021-11-01', '2022-03-31'),
        ('2022-11-01', '2023-03-31'),
        ('2023-11-01', '2024-03-31'),
        ('2024-11-01', '2025-03-31'),
    ]
    
    downloaded_count = 0
    failed_count = 0
    
    # ← MODIFIED: Updated print statements
    print(f"\nTarget: {NUM_IMAGES} images (unprocessed multispectral)")
    print(f"Regions: {len(ANTARCTIC_REGIONS)}")
    print(f"Time periods: {len(year_ranges)}")
    print(f"Format: GeoTIFF Int16 (Raw scientific data)\n")
    
    for region_name, coords in ANTARCTIC_REGIONS.items():
        if downloaded_count >= NUM_IMAGES:
            break
            
        print(f"\n🧊 {region_name}")
        print("-" * 70)
        
        roi = ee.Geometry.Rectangle(coords)
        
        for start_date, end_date in year_ranges:
            if downloaded_count >= NUM_IMAGES:
                break
            
            collection, satellite_type = get_satellite_collection(start_date, end_date, roi)
            
            if collection is None:
                continue
            
            try:
                count = collection.size().getInfo()
                if count == 0:
                    continue
                
                print(f"  {start_date[:7]}: {count} images ({satellite_type.upper()})")
                
                images_to_download = min(5, count)
                images = collection.limit(images_to_download).toList(images_to_download)
                
                for i in range(images_to_download):
                    if downloaded_count >= NUM_IMAGES:
                        break
                    
                    image = ee.Image(images.get(i))
                    
                    # Get date
                    if satellite_type == 'sentinel2':
                        date_acquired = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    else:
                        date_acquired = image.get('DATE_ACQUIRED').getInfo()
                    
                    # ← MODIFIED: Updated filenames
                    sat_prefix = 'S2' if satellite_type == 'sentinel2' else 'L8'
                    filename = f"{OUTPUT_DIR}/multispectral_images/{region_name}_{date_acquired}_{sat_prefix}_MULTI.tif"
                    metadata_filename = f"{OUTPUT_DIR}/metadata/{region_name}_{date_acquired}_{sat_prefix}_MULTI_metadata.txt"
                    
                    # Skip if exists
                    if os.path.exists(filename):
                        downloaded_count += 1
                        print(f"    ⊙ {date_acquired} (exists)")
                        continue
                    
                    print(f"    ↓ {date_acquired}...", end=' ')
                    
                    if download_raw_image(image, roi, filename, satellite_type):
                        save_metadata(image, metadata_filename, region_name, satellite_type)
                        downloaded_count += 1
                        print(f"✓ ({downloaded_count}/{NUM_IMAGES})")
                    else:
                        failed_count += 1
                    
                    time.sleep(2)
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:50]}")
                continue
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"✓ Downloaded: {downloaded_count} images")
    print(f"✗ Failed: {failed_count}")
    # ← MODIFIED: Updated paths
    print(f"📁 Location: {OUTPUT_DIR}/multispectral_images/")
    print(f"📄 Metadata: {OUTPUT_DIR}/metadata/")
    print("\nWARNING: These images are raw Int16 data.")
    print("They will appear black or broken in standard image viewers.")
    print("This is CORRECT. They are ready for ML preprocessing.")
    print("=" * 70)

if __name__ == "__main__":
    main()