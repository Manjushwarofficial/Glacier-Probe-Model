# Glacier Probe Model

An AI-powered system for analyzing and mapping long-term glacier retreat patterns using satellite imagery. This project employs a hybrid approach combining traditional machine learning with deep learning techniques to detect, segment, and track glacier boundaries over time.

![Project Status](https://img.shields.io/badge/status-in%20dev-green)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

## Usage Example

![animation](https://github.com/user-attachments/assets/ae12656b-409e-4766-b549-9c2714a4bbb7)


## Overview

Glacier Probe Model processes multi-temporal satellite imagery from Landsat and Sentinel missions to quantify glacier retreat. The system uses spectral analysis (NDSI-based detection) combined with Random Forest classification for baseline detection, then refines boundaries using U-Net deep learning architecture for complex scenarios including debris-covered glaciers and shadowed regions.

## Key Capabilities

**Phase 1: ML-Based Detection**
- NDSI (Normalized Difference Snow Index) calculation from multispectral bands
- Random Forest classifier trained on spectral, texture (GLCM), and terrain features
- Automated cloud masking using QA bands
- Temporal image co-registration and atmospheric correction

**Phase 2: Deep Learning Enhancement**
- U-Net semantic segmentation for precise boundary delineation
- Transfer learning using ImageNet pre-trained encoders
- Temporal consistency enforcement across time series
- Debris-covered glacier detection using additional NIR/SWIR band combinations

**Analysis Pipeline**
- Multi-decadal change detection (1984-present)
- Automated area loss quantification and retreat rate calculation
- Statistical analysis of seasonal variations
- Interactive geospatial visualization with temporal sliders

## Technology Stack

### Core Libraries
- Python 3.8+
- PyTorch 2.0+ or TensorFlow 2.12+
- Scikit-learn 1.3+
- OpenCV 4.8+
- NumPy 1.24+, Pandas 2.0+

### Geospatial Processing
- Rasterio 1.3+ (GeoTIFF handling)
- GDAL 3.6+
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [Sentinelsat](https://sentinelsat.readthedocs.io/) (Sentinel-2 data access)
- GeoPandas 0.13+ (vector operations)
- Earthpy 0.9+ (remote sensing workflows)

### Machine Learning
- Scikit-image 0.21+ (GLCM texture features)
- XGBoost 2.0+ (gradient boosting alternative)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [timm](https://github.com/huggingface/pytorch-image-models) (pre-trained encoders)

### Visualization
- Matplotlib 3.7+, Plotly 5.17+
- Folium 0.14+ (interactive maps)
- Streamlit 1.28+ (web dashboard)

## Installation

### Prerequisites
Ensure GDAL is installed on your system before proceeding.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev python3-gdal
```

**macOS:**
```bash
brew install gdal
```

**Windows:**
Use [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or install via conda:
```bash
conda install -c conda-forge gdal
```

### Setup

1. Clone the repository
```bash
git clone https://github.com/Manjushwarofficial/Glacier-Probe-Model.git
cd Glacier-Probe-Model
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install Python dependencies
```bash
pip install -r requirements.txt
```

4. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

Follow the authentication flow in your browser and paste the token when prompted.

## Data Sources and Access

### Satellite Imagery

**Landsat Collection 2 (USGS)**
- Resolution: 30m (multispectral), 15m (panchromatic)
- Temporal coverage: 1984-present (Landsat 5/7/8/9)
- Access: [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/catalog/landsat) or [EarthExplorer](https://earthexplorer.usgs.gov/)
- Dataset IDs:
  - `LANDSAT/LC08/C02/T1_L2` (Landsat 8)
  - `LANDSAT/LC09/C02/T1_L2` (Landsat 9)

**Sentinel-2 (ESA Copernicus)**
- Resolution: 10m (visible), 20m (red-edge/SWIR)
- Temporal coverage: 2015-present
- Revisit time: 5 days
- Access: [Copernicus Open Access Hub](https://scihub.copernicus.eu/) or [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- Dataset ID: `COPERNICUS/S2_SR_HARMONIZED`

**MODIS Terra/Aqua**
- Resolution: 250m (bands 1-2), 500m (bands 3-7)
- Temporal coverage: 2000-present
- Access: [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/)

### Glacier Reference Data

**Randolph Glacier Inventory (RGI) v7.0**
- Global glacier outlines with area and location metadata
- Download: [NSIDC RGI Dataset](https://nsidc.org/data/nsidc-0770/versions/7)
- Format: Shapefiles (ESRI) organized by region

**GLIMS Glacier Database**
- Multi-temporal glacier outlines with analysis metadata
- Access: [GLIMS Web Interface](https://www.glims.org/maps/glims)
- API: Available for programmatic access

**Global Land Ice Measurements from Space (GLIMS)**
- Provides validation data and historical glacier extents
- Database: [GLIMS Database Search](https://www.glims.org/maps/search)

### Digital Elevation Models

**SRTM v3 (30m)**
- Coverage: 60°N to 56°S
- Access: [EarthExplorer](https://earthexplorer.usgs.gov/) or GEE: `USGS/SRTMGL1_003`

**ASTER GDEM v3 (30m)**
- Coverage: 83°N to 83°S
- Access: [NASA Earthdata](https://search.earthdata.nasa.gov/) or GEE: `NASA/ASTER_GED/AG100_003`

**Copernicus DEM (30m/90m)**
- Global coverage with improved accuracy
- Access: [Copernicus Data Space](https://dataspace.copernicus.eu/)

## Quick Start

### Basic Glacier Detection

```python
from glacier_probe import GlacierDetector, ChangeAnalyzer

# Initialize ML-based detector
detector = GlacierDetector(
    method='random_forest',
    ndsi_threshold=0.4,
    cloud_threshold=20
)

# Load Landsat 8 scene
image = detector.load_image('LC08_L2SP_path_row_date.tif')

# Detect glacier boundaries
glacier_mask = detector.detect(image)

# Save results
detector.save_mask(glacier_mask, 'output/glacier_2023.tif')
```

### Multi-Temporal Change Analysis

```python
# Analyze retreat between two dates
analyzer = ChangeAnalyzer()

images = [
    'data/glacier_1990.tif',
    'data/glacier_2000.tif',
    'data/glacier_2010.tif',
    'data/glacier_2023.tif'
]

results = analyzer.calculate_temporal_change(images)

print(f"Total area loss: {results['area_loss_km2']:.2f} km²")
print(f"Average retreat rate: {results['retreat_rate_m_per_year']:.2f} m/year")
```

### Running the Full Pipeline

```bash
# Step 1: Download and preprocess satellite data
python scripts/download_data.py --region himalaya --start-date 1990-01-01 --end-date 2023-12-31

# Step 2: ML-based baseline detection
python scripts/ml_detection.py --input data/processed --output results/ml_baseline

# Step 3: DL refinement (requires trained model)
python scripts/dl_segmentation.py --input data/processed --checkpoint models/unet_best.pth --output results/dl_refined

# Step 4: Generate analysis report
python scripts/generate_report.py --ml-results results/ml_baseline --dl-results results/dl_refined --output reports/
```

## Project Structure

```
Glacier-Probe-Model/
│
├── data/
│   ├── raw/                    # Downloaded satellite imagery (GeoTIFF)
│   ├── processed/              # Preprocessed and co-registered images
│   ├── annotations/            # Manual glacier boundary labels (optional)
│   └── reference/              # RGI shapefiles and DEM data
│
├── src/
│   ├── preprocessing/
│   │   ├── atmospheric_correction.py
│   │   ├── cloud_masking.py
│   │   └── coregistration.py
│   ├── features/
│   │   ├── spectral_indices.py    # NDSI, NDWI, NDVI calculation
│   │   ├── texture_features.py     # GLCM implementation
│   │   └── terrain_features.py     # Slope, aspect from DEM
│   ├── models/
│   │   ├── random_forest.py        # RF classifier
│   │   ├── unet.py                 # U-Net architecture
│   │   └── train.py                # Training scripts
│   ├── detection/
│   │   ├── threshold_detection.py  # NDSI thresholding
│   │   ├── ml_detection.py         # ML-based detection
│   │   └── dl_detection.py         # DL-based detection
│   └── visualization/
│       ├── temporal_plots.py
│       └── interactive_maps.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ndsi_baseline.ipynb
│   ├── 03_ml_training.ipynb
│   ├── 04_dl_refinement.ipynb
│   └── 05_temporal_analysis.ipynb
│
├── scripts/
│   ├── download_data.py            # GEE/Sentinelsat data download
│   ├── ml_detection.py
│   ├── dl_segmentation.py
│   └── generate_report.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── configs/
│   ├── ml_config.yaml              # ML hyperparameters
│   └── dl_config.yaml              # DL training config
│
├── models/                         # Saved model checkpoints
├── results/                        # Detection outputs
├── reports/                        # Generated analysis reports
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

### Phase 1: Machine Learning Baseline

**Step 1: Preprocessing**
- Apply atmospheric correction using LEDAPS (Landsat) or Sen2Cor (Sentinel-2)
- Cloud and cloud shadow masking using CFMask algorithm
- Co-registration using AROSICS library for sub-pixel accuracy
- Radiometric normalization across temporal series

**Step 2: Feature Engineering**
- Spectral indices:
  - NDSI = (Green - SWIR) / (Green + SWIR)
  - NDWI = (Green - NIR) / (Green + NIR)
  - NDVI = (NIR - Red) / (NIR + Red)
- Texture features using GLCM (contrast, homogeneity, energy, correlation)
- Terrain derivatives from DEM (slope, aspect, curvature)
- Band ratios (NIR/Red, SWIR1/SWIR2)

**Step 3: Random Forest Classification**
- Training data: RGI glacier outlines + manually verified samples
- Class balancing using SMOTE or class weights
- Hyperparameter tuning via 5-fold cross-validation
- Feature importance analysis
- Post-processing: Morphological opening/closing, minimum area threshold (0.01 km²)

**Expected Performance:**
- IoU: 0.82-0.88 on clean ice
- F1 Score: 0.85-0.91
- Challenges: Debris-covered glaciers, seasonal snow confusion

### Phase 2: Deep Learning Enhancement

**Step 1: Training Data Preparation**
- Generate 256×256 pixel patches from Landsat/Sentinel scenes
- Use Phase 1 predictions as pseudo-labels
- Manual refinement of 10-20% of patches for validation
- Data augmentation: rotation, flipping, color jittering
- Train/val/test split: 70/15/15

**Step 2: U-Net Training**
- Encoder: ResNet34/ResNet50 (ImageNet pre-trained)
- Loss function: Combined Binary Cross-Entropy + Dice Loss
- Optimizer: AdamW with learning rate 1e-4
- Learning rate scheduling: ReduceLROnPlateau
- Early stopping based on validation IoU
- Training epochs: 50-100

**Step 3: Temporal Consistency**
- Apply temporal smoothing using median filtering across 3-5 year windows
- Flag sudden area changes exceeding 20% as potential errors
- Cross-reference with meteorological data for validation

**Expected Performance:**
- IoU: 0.89-0.94 on clean ice
- IoU: 0.72-0.82 on debris-covered glaciers
- Boundary accuracy: ±15m (0.5 pixel for Landsat)

## Performance Metrics

**Spatial Accuracy:**
- Intersection over Union (IoU)
- F1 Score (Dice Coefficient)
- Pixel Accuracy
- Boundary F1 (within 30m buffer)

**Temporal Consistency:**
- Mean Absolute Error of year-to-year area change
- Temporal smoothness coefficient

**Validation:**
- Ground truth comparison using high-resolution imagery (Sentinel-2, Planet)
- Cross-validation with published glacier inventory updates

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-cloud-masking`)
3. Commit changes with clear messages
4. Ensure tests pass (`pytest tests/`)
5. Update documentation if adding new features
6. Submit a pull request with description of changes

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings (NumPy style) to all functions
- Include unit tests for new functionality
- Keep functions focused and under 50 lines when possible

## Citation

If this project contributes to your research, please cite:

```bibtex
@software{glacier_probe_model_2025,
  author = {Manjushwar},
  title = {Glacier Probe Model: Hybrid ML-DL System for Glacier Retreat Analysis},
  year = {2025},
  url = {https://github.com/Manjushwarofficial/Glacier-Probe-Model}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Google Earth Engine for providing free access to Landsat and Sentinel archives
- ESA Copernicus Programme for Sentinel-2 data
- USGS for Landsat Collection 2 data
- NSIDC for hosting the Randolph Glacier Inventory
- GLIMS community for glacier outline validation data

## Contact

Project Maintainer: Manjushwar

GitHub: [@Manjushwarofficial](https://github.com/Manjushwarofficial)

Project Link: [https://github.com/Manjushwarofficial/Glacier-Probe-Model](https://github.com/Manjushwarofficial/Glacier-Probe-Model)

## Development Roadmap

- [x] Repository initialization and documentation
- [ ] Data download pipeline implementation (GEE + Sentinelsat)
- [ ] Phase 1: NDSI baseline and Random Forest classifier
- [ ] Validation framework with RGI cross-comparison
- [ ] Phase 2: U-Net architecture and training pipeline
- [ ] Temporal analysis module with statistical testing
- [ ] Interactive Streamlit dashboard
- [ ] REST API for on-demand glacier analysis
- [ ] Multi-region comparative study (Himalaya, Alps, Andes)
- [ ] Real-time monitoring system with alert capabilities

## Known Issues

- GDAL version conflicts between rasterio and other libraries (pin GDAL==3.6.4)
- GEE Python API authentication expires after 7 days (requires re-authentication)
- Large temporal stacks may exceed memory on systems with <16GB RAM
- Debris-covered glacier detection remains challenging for ML baseline

---

**Status:** This project is under active development. Phase 1 implementation is in progress. Documentation and code will be updated regularly.
