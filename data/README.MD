# Data Directory

This directory contains all datasets used in the Glacier Probe Model project, including raw satellite imagery, processed data, annotations, and reference materials. Total directory size ranges from 700MB to 1GB depending on the number of downloaded scenes and temporal range.

## Directory Structure

```
data/
├── annotations/
├── processed/
├── raw/
├── reference/
└── README.md
```

## Subdirectory Descriptions

### annotations/

Contains manually created and verified glacier boundary annotations used for model training and validation.

**Contents:**
- Hand-labeled glacier masks in GeoTIFF format
- Boundary shapefiles (.shp) with associated metadata
- Training/validation/test split indices
- Quality control logs and annotation guidelines

**Naming Convention:**
```
glacier_mask_YYYYMMDD_pathXXX_rowYYY.tif
glacier_boundary_YYYYMMDD_region_name.shp
```

**Format:** GeoTIFF masks (binary: 1=glacier, 0=non-glacier), ESRI Shapefiles

**Typical Size:** 50-100 MB

### processed/

Contains preprocessed satellite imagery ready for model input after atmospheric correction, cloud masking, and normalization.

**Contents:**
- Atmospherically corrected surface reflectance images
- Cloud-masked scenes with QA bands applied
- Co-registered multi-temporal image stacks
- Computed spectral indices (NDSI, NDWI, NDVI) as separate bands
- Terrain features derived from DEM (slope, aspect, hillshade)
- Normalized and standardized image patches (256x256 pixels)

**Naming Convention:**
```
processed_YYYYMMDD_LX_pathXXX_rowYYY.tif
indices_YYYYMMDD_LX_pathXXX_rowYYY.tif
terrain_features_region_name.tif
```

**Format:** Multi-band GeoTIFF (Float32 for reflectance, Int16 for indices)

**Typical Size:** 400-600 MB

### raw/

Contains unprocessed satellite imagery directly downloaded from sources before any preprocessing steps.

**Contents:**
- Landsat Collection 2 Level-2 scenes (Landsat 5, 7, 8, 9)
- Sentinel-2 Level-2A scenes
- Associated metadata files (.MTL, .xml)
- Digital Elevation Models (SRTM, ASTER GDEM)
- Quality Assessment (QA) bands for cloud detection

**Subdirectories:**
```
raw/
├── landsat_5_tm/
├── landsat_7_etm/
├── landsat_8_oli/
├── landsat_9_oli/
├── sentinel_2/
└── dem/
```

**Naming Convention:**
```
LC08_L2SP_XXX_YYY_YYYYMMDD_YYYYMMDD_02_T1/
S2A_MSIL2A_YYYYMMDD_TXXXXX_RXXX/
```

**Format:** GeoTIFF with companion metadata files

**Typical Size:** 200-400 MB

### reference/

Contains reference datasets and validation materials from established glacier inventories and scientific sources.

**Contents:**
- Randolph Glacier Inventory (RGI) shapefiles for validation
- GLIMS glacier outlines for cross-comparison
- Published glacier extent datasets from scientific literature
- Regional glacier databases (Himalaya, Alps, Andes, etc.)
- Glacier identification codes and metadata tables

**File Types:**
- Shapefiles (.shp, .shx, .dbf, .prj)
- GeoJSON files for web visualization
- CSV files with glacier metadata and statistics
- PDF documentation and data usage guidelines

**Typical Size:** 50-100 MB

## Data Acquisition

### Downloading Satellite Imagery

Use the provided download script to acquire satellite data:

```bash
python scripts/download_data.py --region himalaya --start-date 1990-01-01 --end-date 2023-12-31 --output data/raw
```

**Data Sources:**
- Landsat: USGS Earth Explorer or Google Earth Engine
- Sentinel-2: Copernicus Open Access Hub or Google Earth Engine
- DEM: USGS Earth Explorer (SRTM) or NASA Earthdata

### Reference Data Sources

- RGI: https://www.glims.org/RGI/
- GLIMS: https://www.glims.org/
- Regional datasets: Various scientific publications and institutional repositories

## Data Format Specifications

### Raw Satellite Imagery

**Landsat 8/9 Bands Used:**
- Band 2 (Blue): 0.45-0.51 μm
- Band 3 (Green): 0.53-0.59 μm
- Band 4 (Red): 0.64-0.67 μm
- Band 5 (NIR): 0.85-0.88 μm
- Band 6 (SWIR1): 1.57-1.65 μm
- Band 7 (SWIR2): 2.11-2.29 μm
- QA Pixel: Quality assessment band

**Sentinel-2 Bands Used:**
- Band 2 (Blue): 490 nm
- Band 3 (Green): 560 nm
- Band 4 (Red): 665 nm
- Band 8 (NIR): 842 nm
- Band 11 (SWIR1): 1610 nm
- Band 12 (SWIR2): 2190 nm

**Coordinate System:** UTM Zone (region-specific) / WGS84

**Resolution:** 30m (Landsat), 10-20m (Sentinel-2)

### Processed Data

**Image Patches:**
- Size: 256 x 256 pixels
- Format: Float32 normalized reflectance values (0-1)
- Channels: 6-8 bands (spectral + indices)

**Spectral Indices:**
- NDSI = (Green - SWIR) / (Green + SWIR)
- NDWI = (Green - NIR) / (Green + NIR)
- NDVI = (NIR - Red) / (NIR + Red)

### Annotation Format

**Binary Masks:**
- Values: 0 (non-glacier), 1 (glacier), 255 (no data/cloud)
- Data Type: UInt8
- Projection: Matches source imagery

**Shapefiles:**
- Geometry: Polygon
- Attributes: glacier_id, date, area_km2, confidence, source

## Storage Requirements

**Minimum Disk Space:** 700 MB
**Recommended Disk Space:** 1-2 GB (allows for multiple regions and time periods)
**Maximum Expected Size:** 5 GB (extensive multi-decadal analysis)

**Breakdown by Subdirectory:**
- raw: 200-400 MB
- processed: 400-600 MB
- annotations: 50-100 MB
- reference: 50-100 MB

## Data Management Best Practices

**Organization:**
- Keep raw data immutable after download
- Version processed data with timestamps
- Document preprocessing parameters in metadata files
- Maintain separate folders for different study regions

**Backup:**
- Raw satellite data can be re-downloaded if needed
- Back up annotations and processed data regularly
- Use version control for annotation files

**Cleaning:**
- Periodically remove intermediate processing files
- Archive old datasets not actively used
- Compress large GeoTIFF files with LZW compression

## Data Preprocessing Pipeline

1. Download raw satellite imagery to `raw/`
2. Apply atmospheric correction and cloud masking
3. Co-register multi-temporal scenes
4. Calculate spectral indices and terrain features
5. Save preprocessed data to `processed/`
6. Create training patches and save to `processed/patches/`
7. Load reference data into `reference/` for validation

## File Naming Conventions

**Raw Files:** Follow original source naming (e.g., Landsat scene identifiers)

**Processed Files:** `{type}_{date}_{sensor}_{path}_{row}.tif`
- type: processed, indices, terrain
- date: YYYYMMDD
- sensor: L5, L7, L8, L9, S2A, S2B
- path/row: Landsat WRS-2 coordinates or Sentinel-2 tile ID

**Annotations:** `{type}_{date}_{region}_{version}.{ext}`
- type: mask, boundary, polygon
- date: YYYYMMDD
- region: himalaya, alps, andes, etc.
- version: v1, v2, etc.
- ext: tif, shp, geojson

## Access and Permissions

This data directory may contain large files not suitable for Git version control. Consider using:
- Git LFS for files under 2 GB
- Cloud storage (Google Drive, AWS S3) for larger datasets
- DVC (Data Version Control) for data pipeline management

Add to `.gitignore`:
```
data/raw/
data/processed/
*.tif
*.zip
*.tar.gz
```

## Data Quality Notes

**Cloud Coverage:** Target scenes with less than 20 percent cloud cover over glacier regions

**Temporal Considerations:** Use end-of-summer imagery (August-September) for consistent snow line detection

**Spatial Coverage:** Ensure adequate overlap between scenes for mosaic creation

**Validation:** Cross-reference automated detections with RGI/GLIMS outlines

## Contact and Data Sharing

For questions about data sources, preprocessing procedures, or to request processed datasets for specific regions, please open an issue on the project repository or contact the maintainers.

**Data Attribution:** When using this data in publications, please cite the original data sources (USGS for Landsat, ESA for Sentinel-2, GLIMS/RGI for reference data) in addition to this project.