# Glacier Probe Model

A machine learning and computer vision system for analyzing glacier retreat patterns using satellite imagery. This project employs traditional ML classifiers combined with advanced CV techniques to detect, segment, and track glacier boundaries over time.

![Project Status](https://img.shields.io/badge/status-in%20dev-green)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview

A computer vision project analyzing glacier disintegration over time to support climate awareness and action. Glacier Probe Model applies image segmentation, edge detection, morphological operations, and temporal pattern recognition to multi-decadal satellite imagery. The system uses traditional computer vision techniques combined with Random Forest classification for robust glacier boundary detection and change analysis.

## Key Capabilities

**Image Processing & Feature Extraction**
- NDSI (Normalized Difference Snow Index) calculation from multispectral bands
- Texture analysis using GLCM (Gray-Level Co-occurrence Matrix)
- Edge detection with Canny and Sobel operators
- Morphological operations for noise reduction and boundary refinement
- Automated cloud masking using QA bands

**Machine Learning Detection**
- Random Forest classifier trained on spectral, texture, and terrain features
- XGBoost ensemble for enhanced classification accuracy
- Feature importance analysis for model interpretability
- Cross-validation with spatial stratification

**Computer Vision Pipeline**
- Atmospheric correction and radiometric normalization
- Temporal image co-registration using SIFT/ORB features
- Contour detection and analysis for boundary extraction
- Multi-scale segmentation for handling complex glacier surfaces
- Watershed algorithm for separating adjacent glaciers

**Analysis Capabilities**
- Multi-decadal change detection (1984-present)
- Automated area loss quantification and retreat rate calculation
- Statistical analysis of seasonal variations
- Interactive geospatial visualization with temporal sliders

## Technology Stack

### Core Libraries
- Python 3.8+
- Scikit-learn 1.3+ (ML models and preprocessing)
- OpenCV 4.8+ (image processing and CV operations)
- NumPy 1.24+, Pandas 2.0+ (numerical computing)
- SciPy 1.11+ (signal processing and optimization)

### Geospatial Processing
- Rasterio 1.3+ (GeoTIFF handling)
- GDAL 3.6+ (geospatial transformations)
- Google Earth Engine Python API (satellite data access)
- Sentinelsat (Sentinel-2 data download)
- GeoPandas 0.13+ (vector operations)
- Earthpy 0.9+ (earth science workflows)

### Computer Vision & ML
- Scikit-image 0.21+ (edge detection, texture features, segmentation)
- XGBoost 2.0+ (gradient boosting classifier)
- Mahotas 1.4+ (advanced computer vision algorithms)
- Imutils 0.5+ (convenience functions for image processing)

### Visualization
- Matplotlib 3.7+, Seaborn 0.12+ (static plots)
- Plotly 5.17+ (interactive charts)
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
Use OSGeo4W or install via conda:
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
- Access: [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/landsat) | [EarthExplorer](https://earthexplorer.usgs.gov/)

**Sentinel-2 (ESA Copernicus)**
- Resolution: 10m (visible), 20m (red-edge/SWIR)
- Temporal coverage: 2015-present, 5-day revisit
- Access: [Copernicus Hub](https://scihub.copernicus.eu/) | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)

**MODIS Terra/Aqua**
- Resolution: 250m-500m
- Temporal coverage: 2000-present
- Access: [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/)

### Glacier Reference Data

**Randolph Glacier Inventory (RGI) v7.0**
- Global glacier outlines with metadata
- Download: [NSIDC RGI Dataset](https://nsidc.org/data/nsidc-0770/versions/7)

**GLIMS Glacier Database**
- Multi-temporal glacier outlines
- Access: [GLIMS Database](https://www.glims.org/maps/glims)

### Digital Elevation Models

**SRTM v3 (30m)**
- Coverage: 60°N to 56°S
- Access: [EarthExplorer](https://earthexplorer.usgs.gov/)

**ASTER GDEM v3 (30m)**
- Coverage: 83°N to 83°S
- Access: [NASA Earthdata](https://search.earthdata.nasa.gov/)

**Copernicus DEM (30m/90m)**
- Global coverage
- Access: [Copernicus Data Space](https://dataspace.copernicus.eu/)

## Quick Start

### Basic Glacier Detection

```python
from glacier_probe import GlacierDetector, ChangeAnalyzer

# Initialize ML-based detector
detector = GlacierDetector(
    method='random_forest',
    ndsi_threshold=0.4,
    cloud_threshold=20,
    min_glacier_area=0.01  # km²
)

# Load Landsat 8 scene
image = detector.load_image('LC08_L2SP_path_row_date.tif')

# Detect glacier boundaries with CV pipeline
glacier_mask = detector.detect(image, refine_edges=True)

# Save results
detector.save_mask(glacier_mask, 'output/glacier_2023.tif')
```

### Feature Extraction and Classification

```python
from glacier_probe.features import SpectralIndices, TextureFeatures, TerrainFeatures

# Calculate spectral indices
spectral = SpectralIndices(image)
ndsi = spectral.calculate_ndsi()
ndwi = spectral.calculate_ndwi()
ndvi = spectral.calculate_ndvi()

# Extract texture features using GLCM
texture = TextureFeatures(image['NIR'])
glcm_features = texture.calculate_glcm(
    distances=[1, 3, 5],
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
)

# Compute terrain derivatives
terrain = TerrainFeatures(dem_path='data/srtm_dem.tif')
slope = terrain.calculate_slope()
aspect = terrain.calculate_aspect()

# Combine features and classify
from glacier_probe.models import RandomForestGlacierClassifier

clf = RandomForestGlacierClassifier(n_estimators=200, max_depth=15)
features = np.hstack([ndsi, ndwi, glcm_features, slope])
glacier_prediction = clf.predict(features)
```

### Computer Vision Refinement

```python
from glacier_probe.cv_utils import EdgeRefinement, MorphologicalProcessor

# Edge detection and refinement
edge_refiner = EdgeRefinement()
edges = edge_refiner.detect_edges(
    glacier_mask,
    method='canny',
    sigma=1.5
)

# Morphological post-processing
morph = MorphologicalProcessor()
cleaned_mask = morph.apply_opening(glacier_mask, kernel_size=3)
cleaned_mask = morph.apply_closing(cleaned_mask, kernel_size=5)
cleaned_mask = morph.remove_small_objects(cleaned_mask, min_size=100)

# Extract and smooth contours
contours = morph.extract_contours(cleaned_mask)
smoothed_contours = morph.smooth_contours(contours, epsilon=2.0)
```

### Multi-Temporal Change Analysis

```python
# Analyze retreat between multiple dates
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
print(f"Frontal retreat distance: {results['front_retreat_m']:.2f} m")
```

### Running the Full Pipeline

```bash
# Step 1: Download and preprocess satellite data
python scripts/download_data.py --region himalaya --start-date 1990-01-01 --end-date 2023-12-31

# Step 2: Feature extraction and ML detection
python scripts/ml_detection.py --input data/processed --output results/ml_detection --use-texture

# Step 3: CV-based refinement
python scripts/cv_refinement.py --input results/ml_detection --output results/refined --smooth-boundaries

# Step 4: Generate analysis report
python scripts/generate_report.py --results results/refined --output reports/
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
│   │   ├── coregistration.py       # SIFT/ORB-based alignment
│   │   └── radiometric_norm.py
│   ├── features/
│   │   ├── spectral_indices.py     # NDSI, NDWI, NDVI calculation
│   │   ├── texture_features.py     # GLCM implementation
│   │   ├── terrain_features.py     # Slope, aspect from DEM
│   │   └── edge_features.py        # Canny, Sobel, Laplacian
│   ├── cv_utils/
│   │   ├── edge_detection.py       # Edge operators
│   │   ├── morphological_ops.py    # Opening, closing, erosion, dilation
│   │   ├── contour_analysis.py     # Contour extraction and smoothing
│   │   ├── watershed.py            # Watershed segmentation
│   │   └── superpixels.py          # SLIC/Felzenszwalb segmentation
│   ├── models/
│   │   ├── random_forest.py        # RF classifier
│   │   ├── xgboost_classifier.py   # XGBoost implementation
│   │   ├── ensemble.py             # Model ensembling
│   │   └── train.py                # Training scripts
│   ├── detection/
│   │   ├── threshold_detection.py  # NDSI thresholding
│   │   ├── ml_detection.py         # ML-based detection
│   │   └── hybrid_detection.py     # ML + CV hybrid approach
│   ├── change_analysis/
│   │   ├── temporal_metrics.py     # Area change, retreat rates
│   │   ├── statistical_tests.py    # Trend analysis
│   │   └── uncertainty.py          # Error propagation
│   └── visualization/
│       ├── temporal_plots.py
│       ├── interactive_maps.py
│       └── feature_maps.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ndsi_baseline.ipynb
│   ├── 03_cv_feature_extraction.ipynb
│   ├── 04_ml_training.ipynb
│   ├── 05_temporal_analysis.ipynb
│   └── 06_validation_metrics.ipynb
│
├── scripts/
│   ├── download_data.py            # GEE/Sentinelsat data download
│   ├── ml_detection.py
│   ├── cv_refinement.py
│   └── generate_report.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_cv_utils.py
│   └── test_models.py
│
├── configs/
│   ├── ml_config.yaml              # ML hyperparameters
│   └── cv_config.yaml              # CV pipeline parameters
│
├── models/                         # Saved model checkpoints
├── results/                        # Detection outputs
├── reports/                        # Generated analysis reports
├── requirements.txt
├── .gitignore
├── LICENSE.md
└── README.md
```

## Methodology

### Computer Vision Pipeline

**1. Preprocessing**
- Atmospheric correction using DOS (Dark Object Subtraction) or 6S radiative transfer
- Radiometric normalization across temporal images
- Cloud masking using QA bands and morphological operations
- Image co-registration using SIFT/ORB feature matching
- Histogram equalization for contrast enhancement

**2. Feature Extraction**

*Spectral Indices:*
- NDSI = (Green - SWIR) / (Green + SWIR)
- NDWI = (Green - NIR) / (Green + NIR)
- NDVI = (NIR - Red) / (NIR + Red)

*Texture Features (GLCM):*
- Contrast, dissimilarity, homogeneity
- Energy, correlation, ASM (Angular Second Moment)
- Entropy
- Multi-directional and multi-scale analysis

*Edge Features:*
- Canny edge detection with adaptive thresholding
- Sobel gradient magnitude and direction
- Laplacian of Gaussian (LoG) for blob detection

*Terrain Features:*
- Slope and aspect from DEM
- Curvature (profile and planform)
- Topographic Position Index (TPI)

**3. Segmentation Techniques**
- Otsu's thresholding for initial binary mask
- Watershed algorithm for separating adjacent glaciers
- SLIC superpixels for region-based analysis
- Active contours (snakes) for boundary refinement

**4. Morphological Operations**
- Opening: erosion followed by dilation (noise removal)
- Closing: dilation followed by erosion (gap filling)
- Morphological gradient for boundary enhancement
- Hit-or-miss transform for feature detection

**5. Boundary Refinement**
- Contour detection using cv2.findContours()
- Douglas-Peucker algorithm for contour simplification
- Convex hull computation for debris-covered regions
- Gaussian smoothing of boundary coordinates

### Machine Learning Classification

**Training Data Preparation**
- Stratified sampling from RGI-validated glaciers
- Balanced classes (glacier vs non-glacier)
- Feature normalization using StandardScaler
- Spatial cross-validation to avoid overfitting

**Random Forest Classifier**
- n_estimators: 200-500 trees
- max_depth: 15-20
- min_samples_split: 10-20
- Feature importance analysis
- Out-of-bag error estimation

**XGBoost Ensemble**
- Gradient boosting with learning_rate: 0.01-0.1
- max_depth: 6-10
- subsample: 0.8
- colsample_bytree: 0.8
- Early stopping based on validation loss

**Post-Processing**
- Minimum area filtering (e.g., 0.01 km²)
- Morphological opening/closing
- Connected component analysis
- Contour smoothing and simplification

**Expected Performance**
- IoU: 0.82-0.88
- F1 Score: 0.85-0.91
- Pixel Accuracy: 0.90-0.95
- Boundary F1 (30m buffer): 0.75-0.85

### Temporal Change Detection

**Co-registration and Alignment**
- SIFT/ORB feature matching between temporal images
- RANSAC for robust transformation estimation
- Resampling to common grid

**Change Metrics**
- Area change (km²) and percentage
- Frontal retreat distance (m)
- Average retreat rate (m/year)
- Terminus position tracking

**Statistical Analysis**
- Linear regression for trend analysis
- Mann-Kendall test for monotonic trends
- Change point detection using Pettitt's test
- Uncertainty quantification using error propagation

## Performance Metrics

- **Intersection over Union (IoU):** Overlap between prediction and ground truth
- **F1 Score (Dice Coefficient):** Harmonic mean of precision and recall
- **Pixel Accuracy:** Correctly classified pixels / Total pixels
- **Boundary F1:** F1 score within 30m buffer of true boundary
- **Temporal Consistency Score:** Agreement between consecutive time steps
- **Feature Importance:** Contribution of each feature to classification

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-edge-detection`)
3. Commit changes with clear messages
4. Ensure tests pass (`pytest tests/`)
5. Submit a pull request

Code should follow PEP 8 guidelines with NumPy-style docstrings.

## Citation

If this project contributes to your research, please cite:

```bibtex
@software{glacier_probe_model_2025,
  author = {Manjushwar},
  title = {Glacier Probe Model: ML and CV System for Glacier Retreat Analysis},
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
- OpenCV community for comprehensive computer vision tools
- Scikit-learn and scikit-image developers

## Development Roadmap

- [x] Repository initialization and documentation
- [ ] Data download pipeline implementation (GEE + Sentinelsat)
- [ ] NDSI baseline with adaptive thresholding
- [ ] GLCM texture feature extraction module
- [ ] Edge detection and morphological processing pipeline
- [ ] Random Forest classifier with feature engineering
- [ ] XGBoost ensemble implementation
- [ ] Temporal co-registration using SIFT/ORB
- [ ] Change detection and statistical analysis module
- [ ] Validation framework with RGI cross-comparison
- [ ] Interactive Streamlit dashboard
- [ ] Multi-region comparative study (Himalaya, Alps, Andes)


**Status:** This project is under active development. Core ML and CV modules are in progress. Documentation and code will be updated regularly.