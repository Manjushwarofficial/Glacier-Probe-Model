"""

GLACIER PROBE MODEL - PyQt6 GUI APPLICATION
=================================================
Complete GUI for glacier analysis with prediction and segmentation

Install dependencies:
    pip install PyQt6 matplotlib numpy pandas rasterio joblib scikit-image scipy Pillow
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QTabWidget, QProgressBar,
                             QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
                             QMessageBox, QSplitter, QStatusBar, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QIcon

# --- Imports for CLASSIFICATION Pipeline ---
import joblib
import rasterio
import warnings
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage import filters, feature, measure, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
warnings.filterwarnings('ignore')

# --- Imports for SEGMENTATION Pipeline ---
from PIL import Image
from scipy.ndimage import uniform_filter
# (joblib, os, Path, np, plt are already imported)


# ============================================================================
# EMBEDDED PREDICTION (CLASSIFICATION) PIPELINE
# ============================================================================

# --- All feature extraction functions from your prediction_pipeline.py ---

def extract_spectral_features_rgb(image_data):
    """Spectral/color features"""
    features = {}
    red = image_data[0].astype(float)
    green = image_data[1].astype(float)
    blue = image_data[2].astype(float)
    
    # Band statistics
    features['red_mean'] = np.mean(red)
    features['green_mean'] = np.mean(green)
    features['blue_mean'] = np.mean(blue)
    features['red_std'] = np.std(red)
    features['green_std'] = np.std(green)
    features['blue_std'] = np.std(blue)
    
    # Color ratios
    features['red_green_ratio'] = np.mean(red) / (np.mean(green) + 1e-10)
    features['blue_green_ratio'] = np.mean(blue) / (np.mean(green) + 1e-10)
    features['red_blue_ratio'] = np.mean(red) / (np.mean(blue) + 1e-10)
    
    # Brightness
    brightness = (red + green + blue) / 3
    features['brightness_mean'] = np.mean(brightness)
    features['brightness_std'] = np.std(brightness)
    features['brightness_max'] = np.max(brightness)
    
    # Saturation
    max_rgb = np.maximum(np.maximum(red, green), blue)
    min_rgb = np.minimum(np.minimum(red, green), blue)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-10)
    features['saturation_mean'] = np.mean(saturation)
    features['saturation_std'] = np.std(saturation)
    
    # Hue
    features['hue_mean'] = np.mean(np.arctan2(np.sqrt(3)*(green-blue), 2*red-green-blue))
    
    return features


def extract_edge_features_rgb(image_data):
    """Edge detection features"""
    features = {}
    gray = np.mean(image_data[:3], axis=0)
    
    # Sobel edges
    edges_sobel_h = filters.sobel_h(gray)
    edges_sobel_v = filters.sobel_v(gray)
    edges_sobel = np.sqrt(edges_sobel_h**2 + edges_sobel_v**2)
    
    features['edge_density'] = np.sum(edges_sobel > 0.1) / edges_sobel.size
    features['edge_mean'] = np.mean(edges_sobel)
    features['edge_std'] = np.std(edges_sobel)
    features['edge_max'] = np.max(edges_sobel)
    
    # Canny edges
    edges_canny = feature.canny(gray, sigma=2)
    features['canny_edge_count'] = np.sum(edges_canny)
    features['canny_edge_density'] = np.sum(edges_canny) / edges_canny.size
    
    # Terminus position
    edge_positions = np.where(edges_canny)
    if len(edge_positions[0]) > 0:
        features['terminus_y'] = np.max(edge_positions[0])
        features['terminus_x'] = np.mean(edge_positions[1][edge_positions[0] == features['terminus_y']])
        features['terminus_width'] = np.ptp(edge_positions[1])
        features['glacier_top_y'] = np.min(edge_positions[0])
        features['glacier_length'] = features['terminus_y'] - features['glacier_top_y']
    else:
        features['terminus_y'] = 0
        features['terminus_x'] = 0
        features['terminus_width'] = 0
        features['glacier_top_y'] = 0
        features['glacier_length'] = 0
    
    return features


def extract_texture_features_rgb(image_data):
    """Texture analysis features"""
    features = {}
    gray = np.mean(image_data[:3], axis=0)
    gray_norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-10) * 255).astype(np.uint8)
    
    # GLCM texture
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_norm, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    
    features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
    features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
    features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
    features['energy'] = np.mean(graycoprops(glcm, 'energy'))
    features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    features['ASM'] = np.mean(graycoprops(glcm, 'ASM'))
    
    # Local Binary Pattern
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_norm, n_points, radius, method='uniform')
    features['lbp_mean'] = np.mean(lbp)
    features['lbp_std'] = np.std(lbp)
    
    # Local standard deviation
    features['local_std_3x3'] = np.mean(ndimage.generic_filter(gray, np.std, size=3))
    features['local_std_5x5'] = np.mean(ndimage.generic_filter(gray, np.std, size=5))
    features['local_std_7x7'] = np.mean(ndimage.generic_filter(gray, np.std, size=7))
    
    # Entropy
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    try:
        entropy_img = entropy(gray_norm, disk(5))
        features['entropy_mean'] = np.mean(entropy_img)
        features['entropy_std'] = np.std(entropy_img)
    except:
        features['entropy_mean'] = 0
        features['entropy_std'] = 0
    
    return features


def extract_morphological_features_rgb(image_data):
    """Morphological/shape features"""
    features = {}
    gray = np.mean(image_data[:3], axis=0)
    
    # Binary mask
    try:
        thresh = threshold_otsu(gray)
        binary_mask = gray > thresh
    except:
        binary_mask = gray > 0.7
    
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)
    
    # Area features
    features['glacier_area'] = np.sum(binary_mask)
    features['glacier_percentage'] = np.sum(binary_mask) / binary_mask.size * 100
    
    # Region properties
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)
    
    if len(regions) > 0:
        largest = max(regions, key=lambda x: x.area)
        features['perimeter'] = largest.perimeter
        features['eccentricity'] = largest.eccentricity
        features['solidity'] = largest.solidity
        features['extent'] = largest.extent
        features['major_axis'] = largest.major_axis_length
        features['minor_axis'] = largest.minor_axis_length
        features['orientation'] = largest.orientation
        features['centroid_y'] = largest.centroid[0]
        features['centroid_x'] = largest.centroid[1]
        
        bbox = largest.bbox
        features['bbox_height'] = bbox[2] - bbox[0]
        features['bbox_width'] = bbox[3] - bbox[1]
        features['bbox_area'] = features['bbox_height'] * features['bbox_width']
        features['convex_area'] = largest.convex_area
        features['filled_area'] = largest.filled_area
    else:
        for key in ['perimeter', 'eccentricity', 'solidity', 'extent', 
                    'major_axis', 'minor_axis', 'orientation',
                    'centroid_y', 'centroid_x', 'bbox_height', 
                    'bbox_width', 'bbox_area', 'convex_area', 'filled_area']:
            features[key] = 0
    
    return features


def extract_statistical_features_rgb(image_data):
    """Statistical features per band"""
    features = {}
    band_names = ['red', 'green', 'blue']
    
    for i, name in enumerate(band_names):
        band = image_data[i].flatten()
        features[f'{name}_mean'] = np.mean(band)
        features[f'{name}_std'] = np.std(band)
        features[f'{name}_min'] = np.min(band)
        features[f'{name}_max'] = np.max(band)
        features[f'{name}_median'] = np.median(band)
        features[f'{name}_range'] = np.ptp(band)
        features[f'{name}_percentile_25'] = np.percentile(band, 25)
        features[f'{name}_percentile_75'] = np.percentile(band, 75)
        features[f'{name}_skewness'] = skew(band)
        features[f'{name}_kurtosis'] = kurtosis(band)
    
    return features


def extract_gradient_features_rgb(image_data):
    """Gradient features"""
    features = {}
    gray = np.mean(image_data[:3], axis=0)
    
    grad_y, grad_x = np.gradient(gray)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features['gradient_mean'] = np.mean(grad_magnitude)
    features['gradient_std'] = np.std(grad_magnitude)
    features['gradient_max'] = np.max(grad_magnitude)
    
    grad_direction = np.arctan2(grad_y, grad_x)
    features['gradient_direction_mean'] = np.mean(grad_direction)
    features['gradient_direction_std'] = np.std(grad_direction)
    
    return features


def extract_all_features(image_data):
    """
    Extract ALL 89 features from image
    """
    all_features = {}
    
    # Extract each feature group
    all_features.update(extract_spectral_features_rgb(image_data))
    all_features.update(extract_edge_features_rgb(image_data))
    all_features.update(extract_texture_features_rgb(image_data))
    all_features.update(extract_morphological_features_rgb(image_data))
    all_features.update(extract_statistical_features_rgb(image_data))
    all_features.update(extract_gradient_features_rgb(image_data))
    
    return all_features


class GlacierRetreatPredictor:
    """
    Complete pipeline for predicting glacier retreat from raw image
    (This class REPLACES the simple placeholder)
    """
    
    def __init__(self, model_path, scaler_path, selected_features_path):
        """
        Load trained model and preprocessing objects
        """
        print("Loading model and preprocessing objects...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.selected_features = joblib.load(selected_features_path) # This is your features_list.pkl
        
        print(f"✓ Model loaded: {type(self.model).__name__}")
        print(f"✓ Scaler loaded: {type(self.scaler).__name__}")
        print(f"✓ Selected features: {len(self.selected_features)}")
    
    
    def load_image(self, image_path):
        """
        Load satellite image from file
        """
        with rasterio.open(image_path) as src:
            image_data = src.read()
        
        # Ensure we have at least 3 bands
        if image_data.shape[0] < 3:
            raise ValueError(f"Image must have at least 3 bands (RGB), got {image_data.shape[0]}")
        
        return image_data
    
    
    def preprocess_image(self, image_data):
        """
        Optional preprocessing (normalization, etc.)
        """
        return image_data
    
    
    def extract_features(self, image_data):
        """
        Extract all features from image using the functions above
        """
        print("Extracting features...")
        features = extract_all_features(image_data)
        return features
    
    
    def prepare_features_for_model(self, features_dict):
        """
        Convert features dict to array and select only training features
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Select only the features used in training
        # Handle missing features (fill with 0)
        for feat in self.selected_features:
            if feat not in features_df.columns:
                features_df[feat] = 0
        
        # This line will now work, as features_df contains 'red_mean' etc.
        X = features_df[self.selected_features].values
        
        return X
    
    
    def predict(self, image_path):
        """
        Complete pipeline: Load image → Extract features → Predict
        """
        print(f"\n[1/5] Loading image...")
        image_data = self.load_image(image_path)
        print(f"  ✓ Image shape: {image_data.shape}")
        
        print("\n[2/5] Preprocessing...")
        image_data = self.preprocess_image(image_data)
        print(f"  ✓ Preprocessed")
        
        print("\n[3/5] Extracting features...")
        features = self.extract_features(image_data)
        print(f"  ✓ Extracted {len(features)} features")
        
        print("\n[4/5] Preparing features for model...")
        X = self.prepare_features_for_model(features)
        print(f"  ✓ Feature vector shape: {X.shape}")
        
        print("\n[5/5] Scaling and predicting...")
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        print(f"\n{'='*70}")
        print(f"PREDICTION RESULT")
        print(f"{'='*70}")
        print(f"Status: {'RETREATING' if prediction == 1 else 'STABLE'}")
        print(f"Confidence:")
        print(f"  - Stable:     {probability[0]:.1%}")
        print(f"  - Retreating: {probability[1]:.1%}")
        print(f"{'='*70}\n")
        
        # This dict matches what the GUI worker expects
        return {
            'prediction': prediction,
            'prediction_label': 'retreating' if prediction == 1 else 'stable',
            'probability_stable': probability[0],
            'probability_retreating': probability[1],
            'confidence': probability[prediction],
            'features': features,
            'image_data': image_data
        }


# ============================================================================
# EMBEDDED SEGMENTATION PIPELINE (NEWLY ADDED)
# ============================================================================

def extract_features_fast(image):
    """
    Extract 9 simple features per pixel:
    1-3: RGB values
    4-6: RGB means in 7x7 window
    7: Brightness
    8-9: Blue-Red ratio, Green-Red ratio (water indices)
    """
    if isinstance(image, str):
        image = np.array(Image.open(image))
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] > 3:
        # Multi-band image, take first 3
        image = image[:, :, :3]
    
    h, w = image.shape[:2]
    
    # Normalize to 0-1
    if image.dtype == np.uint8:
        img_norm = image.astype(float) / 255.0
    elif image.dtype == np.uint16:
        img_norm = image.astype(float) / 65535.0
    else:
        # Already float
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Create feature array
    features = np.zeros((h, w, 9))
    
    # RGB values
    features[:, :, 0] = img_norm[:, :, 0]  # Red
    features[:, :, 1] = img_norm[:, :, 1]  # Green
    features[:, :, 2] = img_norm[:, :, 2]  # Blue
    
    # Local means (smoothed RGB)
    features[:, :, 3] = uniform_filter(img_norm[:, :, 0], size=7)  # Red mean
    features[:, :, 4] = uniform_filter(img_norm[:, :, 1], size=7)  # Green mean
    features[:, :, 5] = uniform_filter(img_norm[:, :, 2], size=7)  # Blue mean
    
    # Brightness
    features[:, :, 6] = img_norm.mean(axis=2)
    
    # Color ratios (water detection)
    with np.errstate(divide='ignore', invalid='ignore'):
        features[:, :, 7] = np.where(img_norm[:, :, 0] > 0,
                                      img_norm[:, :, 2] / img_norm[:, :, 0], 0)  # B/R
        features[:, :, 8] = np.where(img_norm[:, :, 0] > 0,
                                     img_norm[:, :, 1] / img_norm[:, :, 0], 0)  # G/R
    
    return features

class GlacierSegmentationPredictor:
    """
    Complete pipeline for segmenting glacier images
    """
    
    def __init__(self, model_path, scaler_path):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to trained RandomForest model (.pkl)
            scaler_path: Path to fitted StandardScaler (.pkl)
        """
        print("="*70)
        print("GLACIER SEGMENTATION PREDICTOR")
        print("="*70)
        print(f"Loading model from: {model_path}")
        print(f"Loading scaler from: {scaler_path}")
        
        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"✓ Model loaded: {type(self.model).__name__}")
        print(f"✓ Scaler loaded: {type(self.scaler).__name__}")
        
        # Class mapping
        self.classes = {
            1: 'ice',
            2: 'water',
            3: 'other'
        }
        
        # Color map for visualization
        self.colors = np.array([
            [0, 0, 0],         # 0: black (unused)
            [0, 255, 255],     # 1: cyan (ice)
            [0, 0, 255],       # 2: blue (water)
            [128, 128, 128]    # 3: gray (other)
        ])
        
        print("="*70 + "\n")
        
    def load_image(self, image_path):
        """
        Load image from file (supports .tif, .jpg, .png)
        """
        try:
            # Try rasterio first (for .tif files)
            with rasterio.open(image_path) as src:
                image = src.read()
                # Convert from (bands, height, width) to (height, width, bands)
                if image.shape[0] <= 4:
                    image = image.transpose(1, 2, 0)
                # Take first 3 bands if more exist
                if image.shape[2] > 3:
                    image = image[:, :, :3]
        except:
            # Fallback to PIL
            image = np.array(Image.open(image_path))
        
        return image
        
    def segment(self, image_path):
        """
        Complete segmentation pipeline
        
        Args:
            image_path: Path to glacier image
            
        Returns:
            segmentation: Segmentation map (H x W) with class labels
            image: Original image
            features: Extracted features
        """
        print(f"\n{'='*70}")
        print(f"SEGMENTING IMAGE")
        print(f"{'='*70}")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Step 1: Load image
        print("\n[1/5] Loading image...")
        image = self.load_image(image_path)
        print(f"  ✓ Image shape: {image.shape}")
        
        # Step 2: Extract features
        print("\n[2/5] Extracting features...")
        features = extract_features_fast(image)
        h, w, n_features = features.shape
        print(f"  ✓ Feature shape: {features.shape}")
        print(f"  ✓ Total pixels: {h * w:,}")
        
        # Step 3: Reshape for prediction
        print("\n[3/5] Preparing features...")
        X = features.reshape(-1, n_features)
        print(f"  ✓ Feature matrix: {X.shape}")
        
        # Step 4: Scale features
        print("\n[4/5] Scaling features...")
        X_scaled = self.scaler.transform(X)
        print(f"  ✓ Features scaled")
        
        # Step 5: Predict
        print("\n[5/5] Predicting segmentation...")
        predictions = self.model.predict(X_scaled)
        
        # Reshape to image
        segmentation = predictions.reshape(h, w)
        
        # Calculate class statistics
        unique, counts = np.unique(segmentation, return_counts=True)
        stats_dict = {cls: count for cls, count in zip(unique, counts)}
        
        print(f"\n{'='*70}")
        print(f"SEGMENTATION COMPLETE")
        print(f"{'='*70}")
        print(f"Class Distribution:")
        for cls, count in stats_dict.items():
            cls_name = self.classes.get(cls, 'unknown')
            percentage = (count / segmentation.size) * 100
            print(f"  {cls_name.capitalize():10s}: {count:8,} pixels ({percentage:5.2f}%)")
        print(f"{'='*70}\n")
        
        return {
            'segmentation': segmentation,
            'image': image,
            'features': features,
            'stats': stats_dict
        }
    
    # Note: The 'visualize' and 'export_results' methods from the
    # original script are NOT needed here, as the GUI will
    # handle its own visualization and exporting.


# ============================================================================
# WORKER THREADS
# ============================================================================

class PredictionWorker(QThread):
    """Background worker for glacier (classification) prediction"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, image_path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit("Loading image and extracting features...")
            # This calls the FULL predict method
            result = self.predictor.predict(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")

# --- NEWLY ADDED ---
class SegmentationWorker(QThread):
    """Background worker for glacier segmentation"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, predictor, image_path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit("Loading image and segmenting...")
            # This calls the segmentation 'segment' method
            result = self.predictor.segment(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n\n{traceback.format_exc()}")
# --- END NEW ---

# ============================================================================
# MATPLOTLIB CANVAS WIDGET
# ============================================================================

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#FBFBFF') # Match background
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class GlacierAnalysisApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.predictor = None # For classification
        self.segmentation_predictor = None # For segmentation (NEW)
        
        self.current_image_path = None
        self.current_result = None
        
        self.current_seg_image_path = None # (NEW)
        self.current_seg_result = None # (NEW)

        self.initUI()
        self.setup_connections()
        self.apply_styling() # Apply styles after all widgets are created

    def initUI(self):
        """Initialize the user interface"""

        self.setWindowTitle("Glacier Probe Model")
        self.setGeometry(100, 100, 1400, 900)
        # self.setWindowIcon(QIcon("path/to/your/icon.png")) # Add an icon for a modern look

        # Create central widget and main layout
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget") # For styling
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ===== NEW: MAIN TITLE =====
        self.title_label = QLabel("Glacier Probe Model")
        self.title_label.setObjectName("MainTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.title_label)

        # ===== TOP: MODEL CONFIGURATION =====
        config_group = self.create_config_section()
        main_layout.addWidget(config_group)

        # ===== MIDDLE: TAB WIDGET =====
        self.tabs = QTabWidget()

        # Tab 1: Single Image Prediction (Classification)
        self.tab_single = self.create_single_prediction_tab()
        self.tabs.addTab(self.tab_single, "Single Image Prediction")
        
        # Tab 2: Image Segmentation (NEW)
        self.tab_segment = self.create_segmentation_tab()
        self.tabs.addTab(self.tab_segment, "Image Segmentation")

        # Tab 3: Batch Processing
        self.tab_batch = self.create_batch_processing_tab()
        self.tabs.addTab(self.tab_batch, "Batch Processing")

        # Tab 4: Results & Export
        self.tab_results = self.create_results_tab()
        self.tabs.addTab(self.tab_results, "Results & Export")

        main_layout.addWidget(self.tabs)

        # ===== BOTTOM: STATUS BAR =====
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please load model files.")

    def create_config_section(self):
        """Create model configuration section"""
        group = QGroupBox("Model Configuration")
        layout = QGridLayout()

        # --- Classification Model ---
        layout.addWidget(QLabel("--- Classification Model ---"), 0, 0, 1, 3)
        layout.addWidget(QLabel("Model File:"), 1, 0)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("glacier_retreat_model_svm_(rbf).pkl")
        layout.addWidget(self.model_path_edit, 1, 1)
        btn_model = QPushButton("Browse...")
        btn_model.clicked.connect(lambda: self.browse_file(self.model_path_edit, "Model (*.pkl)"))
        layout.addWidget(btn_model, 1, 2)

        layout.addWidget(QLabel("Scaler File:"), 2, 0)
        self.scaler_path_edit = QLineEdit()
        self.scaler_path_edit.setPlaceholderText("glacier_retreat_scaler.pkl")
        layout.addWidget(self.scaler_path_edit, 2, 1)
        btn_scaler = QPushButton("Browse...")
        btn_scaler.clicked.connect(lambda: self.browse_file(self.scaler_path_edit, "Scaler (*.pkl)"))
        layout.addWidget(btn_scaler, 2, 2)

        layout.addWidget(QLabel("Features File:"), 3, 0)
        self.features_path_edit = QLineEdit()
        self.features_path_edit.setPlaceholderText("glacier_retreat_features.pkl")
        layout.addWidget(self.features_path_edit, 3, 1)
        btn_features = QPushButton("Browse...")
        btn_features.clicked.connect(lambda: self.browse_file(self.features_path_edit, "Features (*.pkl)"))
        layout.addWidget(btn_features, 3, 2)

        # --- Segmentation Model (NEW) ---
        layout.addWidget(QLabel("--- Segmentation Model ---"), 4, 0, 1, 3)
        layout.addWidget(QLabel("Seg. Model File:"), 5, 0)
        self.seg_model_path_edit = QLineEdit()
        self.seg_model_path_edit.setPlaceholderText("glacier_segmentation_model.pkl")
        layout.addWidget(self.seg_model_path_edit, 5, 1)
        btn_seg_model = QPushButton("Browse...")
        btn_seg_model.clicked.connect(lambda: self.browse_file(self.seg_model_path_edit, "Model (*.pkl)"))
        layout.addWidget(btn_seg_model, 5, 2)

        layout.addWidget(QLabel("Seg. Scaler File:"), 6, 0)
        self.seg_scaler_path_edit = QLineEdit()
        self.seg_scaler_path_edit.setPlaceholderText("glacier_segmentation_scaler.pkl")
        layout.addWidget(self.seg_scaler_path_edit, 6, 1)
        btn_seg_scaler = QPushButton("Browse...")
        btn_seg_scaler.clicked.connect(lambda: self.browse_file(self.seg_scaler_path_edit, "Scaler (*.pkl)"))
        layout.addWidget(btn_seg_scaler, 6, 2)
        
        # --- Load Button & Status ---
        self.btn_load_model = QPushButton("Load Models")
        self.btn_load_model.setObjectName("PrimaryButton")
        self.btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load_model, 7, 0, 1, 3)

        self.model_status_label = QLabel("Model Status: Not Loaded")
        self.model_status_label.setObjectName("StatusLabel")
        self.model_status_label.setProperty("status", "neutral")
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.model_status_label, 8, 0, 1, 3)

        group.setLayout(layout)
        return group

    def create_single_prediction_tab(self):
        """Create single image prediction (classification) tab"""
        widget = QWidget()
        layout = QHBoxLayout()

        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setContentsMargins(0, 10, 10, 10) 

        self.btn_load_image = QPushButton("Load Glacier Image")
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_image.setEnabled(False)
        left_layout.addWidget(self.btn_load_image)

        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setWordWrap(True)
        left_layout.addWidget(self.image_info_label)

        self.btn_predict = QPushButton("Predict Retreat Status")
        self.btn_predict.setObjectName("PrimaryButton") 
        self.btn_predict.clicked.connect(self.predict_single)
        self.btn_predict.setEnabled(False)
        left_layout.addWidget(self.btn_predict)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        left_layout.addWidget(self.progress_bar)

        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        features_group = QGroupBox("Key Features")
        features_layout = QVBoxLayout()
        self.features_text = QTextEdit()
        self.features_text.setReadOnly(True)
        features_layout.addWidget(self.features_text)
        features_group.setLayout(features_layout)
        left_layout.addWidget(features_group)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # Right panel: Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 0, 10)

        self.canvas_single = MplCanvas(self, width=8, height=6)
        right_layout.addWidget(self.canvas_single)

        self.btn_export_single = QPushButton("Export Results")
        self.btn_export_single.clicked.connect(self.export_single_result)
        self.btn_export_single.setEnabled(False)
        right_layout.addWidget(self.btn_export_single)

        right_panel.setLayout(right_layout)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #DDE2E8; }")

        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget

    # --- NEWLY ADDED ---
    def create_segmentation_tab(self):
        """Create image segmentation tab"""
        widget = QWidget()
        layout = QHBoxLayout()

        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setContentsMargins(0, 10, 10, 10)

        # Load image button
        self.btn_load_seg_image = QPushButton("Load Image for Segmentation")
        self.btn_load_seg_image.clicked.connect(self.load_segmentation_image)
        self.btn_load_seg_image.setEnabled(False) # Disabled until model is loaded
        left_layout.addWidget(self.btn_load_seg_image)

        # Image info
        self.seg_image_info_label = QLabel("No image loaded")
        self.seg_image_info_label.setWordWrap(True)
        left_layout.addWidget(self.seg_image_info_label)

        # Predict button
        self.btn_run_segmentation = QPushButton("Run Segmentation")
        self.btn_run_segmentation.setObjectName("PrimaryButton")
        self.btn_run_segmentation.clicked.connect(self.run_segmentation)
        self.btn_run_segmentation.setEnabled(False) # Disabled until image is loaded
        left_layout.addWidget(self.btn_run_segmentation)

        # Progress bar
        self.seg_progress_bar = QProgressBar()
        self.seg_progress_bar.setVisible(False)
        self.seg_progress_bar.setTextVisible(False)
        left_layout.addWidget(self.seg_progress_bar)

        # Results display
        results_group = QGroupBox("Segmentation Statistics")
        results_layout = QVBoxLayout()
        self.seg_results_text = QTextEdit()
        self.seg_results_text.setReadOnly(True)
        results_layout.addWidget(self.seg_results_text)
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)
        
        left_layout.addStretch()
        left_panel.setLayout(left_layout)

        # Right panel: Visualization (3 plots)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setContentsMargins(10, 10, 0, 10)

        self.canvas_segmentation = MplCanvas(self, width=12, height=6)
        # Clear the default 111 axis so we can add 3
        self.canvas_segmentation.fig.clear() 
        right_layout.addWidget(self.canvas_segmentation)

        # Export button
        self.btn_export_segmentation = QPushButton("Export Segmentation Results")
        self.btn_export_segmentation.clicked.connect(self.export_segmentation_result)
        self.btn_export_segmentation.setEnabled(False)
        right_layout.addWidget(self.btn_export_segmentation)

        right_panel.setLayout(right_layout)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #DDE2E8; }")

        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget
    # --- END NEW ---

    def create_batch_processing_tab(self):
        """Create batch processing tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setContentsMargins(10, 10, 10, 10) # Add padding

        # Controls
        controls = QHBoxLayout()

        self.btn_select_folder = QPushButton("Select Image Folder")
        self.btn_select_folder.clicked.connect(self.select_batch_folder)
        self.btn_select_folder.setEnabled(False)
        controls.addWidget(self.btn_select_folder)

        self.batch_folder_label = QLabel("No folder selected")
        controls.addWidget(self.batch_folder_label, 1) # Give label more space

        self.btn_run_batch = QPushButton("Run Batch Prediction")
        self.btn_run_batch.setObjectName("PrimaryButton") # For special styling
        self.btn_run_batch.clicked.connect(self.run_batch_prediction)
        self.btn_run_batch.setEnabled(False)
        controls.addWidget(self.btn_run_batch)

        layout.addLayout(controls)

        # Progress
        self.batch_progress = QProgressBar()
        layout.addWidget(self.batch_progress)

        # Results table
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels(['Image', 'Prediction', 'Confidence', 'Status', 'Time'])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table)

        # Export batch
        self.btn_export_batch = QPushButton("Export Batch Results")
        self.btn_export_batch.clicked.connect(self.export_batch_results)
        self.btn_export_batch.setEnabled(False)
        layout.addWidget(self.btn_export_batch)

        widget.setLayout(layout)
        return widget

    def create_results_tab(self):
        """Create results and export tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setContentsMargins(10, 10, 10, 10) # Add padding

        # Summary statistics
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Visualization canvas

        self.canvas_results = MplCanvas(self, width=10, height=6)
        self.canvas_results.axes.clear() # Clear default axis
        layout.addWidget(self.canvas_results)

        # Export options
        export_layout = QHBoxLayout()
        export_layout.addStretch() # Push buttons to the right
        
        self.btn_export_csv = QPushButton("Export as CSV")
        self.btn_export_csv.clicked.connect(lambda: self.export_results('csv'))
        export_layout.addWidget(self.btn_export_csv)

        self.btn_export_pdf = QPushButton("Export Report (PDF)")
        self.btn_export_pdf.clicked.connect(lambda: self.export_results('pdf'))
        export_layout.addWidget(self.btn_export_pdf)

        layout.addLayout(export_layout)

        widget.setLayout(layout)
        return widget

    def apply_styling(self):
        """Apply custom styling to the application"""
        
        # --- Your Color Palette ---
        COLOR_BACKGROUND = "#FBFBFF"    # Main background
        COLOR_TEXT = "#040F16"          # Main text
        COLOR_TITLE_ACCENT = "#01BAEF"  # Title color (Light Blue)
        COLOR_BUTTON = "#0B4F6C"        # Primary button background (Dark Teal)
        COLOR_BUTTON_TEXT = "#FBFBFF"   # Primary button text
        
        # --- Complementary Colors ---
        COLOR_BORDER = "#DDE2E8"        # Light border for inputs/groups
        COLOR_INPUT_BG = "#FFFFFF"      # Input field background
        COLOR_DISABLED_BG = "#E0E0E0"
        COLOR_DISABLED_TEXT = "#A0A0A0"
        COLOR_BUTTON_HOVER = "#11678C"  # Hover for dark teal button
        COLOR_ACCENT_GREEN = "#4CAF50"  # For success
        COLOR_ACCENT_RED = "#D9534F"    # For error/retreating
        
        self.setStyleSheet(f"""
            QMainWindow, QWidget#CentralWidget {{
                background-color: {COLOR_BACKGROUND};
            }}
            
            /* --- Main Title --- */
            QLabel#MainTitle {{
                font-size: 28px;
                font-weight: bold;
                color: {COLOR_TITLE_ACCENT};
                padding: 5px 0 10px 0;
            }}
            
            QGroupBox {{
                font-size: 14px;
                font-weight: bold;
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {COLOR_BUTTON}; /* CHANGED */
            }}

            QLabel {{
                color: {COLOR_TEXT};
                font-size: 14px;
            }}
            
            /* --- StatusLabel custom property styling --- */
            QLabel#StatusLabel {{
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }}
            QLabel#StatusLabel[status="neutral"] {{
                color: {COLOR_TEXT};
            }}
            QLabel#StatusLabel[status="success"] {{
                color: {COLOR_ACCENT_GREEN};
            }}
            QLabel#StatusLabel[status="error"] {{
                color: {COLOR_ACCENT_RED};
            }}
            
            /* --- Default Button --- */
            QPushButton {{
                background-color: {COLOR_INPUT_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
            }}
            QPushButton:disabled {{
                background-color: {COLOR_DISABLED_BG};
                color: {COLOR_DISABLED_TEXT};
                border-color: {COLOR_DISABLED_BG};
            }}
            
            /* --- Primary Buttons (from objectName) --- */
            QPushButton#PrimaryButton {{
                background-color: {COLOR_BUTTON};
                color: {COLOR_BUTTON_TEXT};
                font-weight: bold;
                font-size: 15px;
                padding: 12px 15px;
            }}
            QPushButton#PrimaryButton:hover {{
                background-color: {COLOR_BUTTON_HOVER};
            }}
            QPushButton#PrimaryButton:disabled {{
                background-color: #5F7A89;
                color: {COLOR_BUTTON_TEXT};
                border-color: #5F7A89;
            }}
            
            /* --- Input Fields --- */
            QLineEdit, QTextEdit, QTableWidget {{
                background-color: {COLOR_INPUT_BG};
                color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }}
            QLineEdit:focus, QTextEdit:focus {{
                border: 1px solid {COLOR_TITLE_ACCENT}; /* Light Blue focus */
            }}
            
            /* --- Tabs --- */
            QTabWidget::pane {{
                border-top: 1px solid {COLOR_BORDER};
            }}
            QTabBar::tab {{
                background: {COLOR_BACKGROUND};
                color: #555;
                border: 1px solid {COLOR_BORDER};
                border-bottom: none;
                padding: 10px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 14px;
            }}
            QTabBar::tab:hover {{
                background: #f0f0f0;
            }}
            QTabBar::tab:selected {{
                background: {COLOR_BACKGROUND};
                color: {COLOR_TITLE_ACCENT}; 
                font-weight: bold;
                border-color: {COLOR_BORDER};
                border-bottom: 1px solid {COLOR_BACKGROUND}; /* Hides border */
                margin-bottom: -1px; /* Pulls tab pane up */
            }}
            
            /* --- Other Widgets --- */
            QStatusBar {{
                color: {COLOR_TEXT};
                font-size: 13px;
            }}
            QProgressBar {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_TITLE_ACCENT};
                border-radius: 5px;
            }}
        """)

    def setup_connections(self):
        """Setup signal-slot connections"""
        pass

    # ========================================================================
    # SLOT METHODS
    # ========================================================================

    def browse_file(self, line_edit, file_filter):
        """Browse for a file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def load_model(self):
        """Load the ML models and preprocessing objects"""
        
        # Get all 5 paths
        model_path = self.model_path_edit.text()
        scaler_path = self.scaler_path_edit.text()
        features_path = self.features_path_edit.text()

        seg_model_path = self.seg_model_path_edit.text()
        seg_scaler_path = self.seg_scaler_path_edit.text()

        # Check for at least one set of files
        class_files_present = all([model_path, scaler_path, features_path])
        seg_files_present = all([seg_model_path, seg_scaler_path])
        
        if not class_files_present and not seg_files_present:
            QMessageBox.warning(self, "Warning", "Please specify files for at least one model.")
            return

        status_messages = []
        
        # Try loading classification model
        if class_files_present:
            if all([os.path.exists(p) for p in [model_path, scaler_path, features_path]]):
                try:
                    self.predictor = GlacierRetreatPredictor(model_path, scaler_path, features_path)
                    self.btn_load_image.setEnabled(True)
                    self.btn_select_folder.setEnabled(True)
                    status_messages.append("✓ Classification Model Loaded")
                except Exception as e:
                    self.btn_load_image.setEnabled(False)
                    self.btn_select_folder.setEnabled(False)
                    status_messages.append(f"✗ Classification Model Failed: {e}")
            else:
                status_messages.append("✗ Classification Files: Not Found")
        else:
            status_messages.append("Classification Model: Skipped")

        # Try loading segmentation model
        if seg_files_present:
            if all([os.path.exists(p) for p in [seg_model_path, seg_scaler_path]]):
                try:
                    self.segmentation_predictor = GlacierSegmentationPredictor(seg_model_path, seg_scaler_path)
                    self.btn_load_seg_image.setEnabled(True) # Enable button on new tab
                    status_messages.append("✓ Segmentation Model Loaded")
                except Exception as e:
                    self.btn_load_seg_image.setEnabled(False)
                    status_messages.append(f"✗ Segmentation Model Failed: {e}")
            else:
                status_messages.append("✗ Segmentation Files: Not Found")
        else:
            status_messages.append("Segmentation Model: Skipped")

        # Update status label
        final_status = "\n".join(status_messages)
        self.model_status_label.setText(f"Model Status:\n{final_status}")
        
        if "✓" in final_status:
            self.model_status_label.setProperty("status", "success")
        elif "✗" in final_status:
            self.model_status_label.setProperty("status", "error")
        else:
            self.model_status_label.setProperty("status", "neutral")

        self.status_bar.showMessage("Model loading complete.", 3000)
        
        # Re-apply styling to update the property
        self.model_status_label.style().unpolish(self.model_status_label)
        self.model_status_label.style().polish(self.model_status_label)


    def load_image(self):
        """Load a glacier image for CLASSIFICATION"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Glacier Image", "",
            "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)"
        )

        if file_path:
            self.current_image_path = file_path
            self.image_info_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.btn_predict.setEnabled(True)
            self.status_bar.showMessage(f"Image loaded: {os.path.basename(file_path)}", 3000)

            # Display image
            self.display_image(file_path, self.canvas_single.axes)
            self.canvas_single.draw()


    def display_image(self, image_path, axes):
        """Display the loaded image on a given axes"""
        try:
            with rasterio.open(image_path) as src:
                image_data = src.read()

            if image_data.shape[0] >= 3:
                rgb = image_data[:3].transpose(1, 2, 0)
            else:
                rgb = image_data[0]

            # Normalize for display
            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
            rgb_norm = np.clip(rgb_norm, 0, 1) # Clip to valid range

            axes.clear()
            axes.imshow(rgb_norm)
            axes.set_title(os.path.basename(image_path), color="#040F16")
            axes.axis('off')
            axes.set_facecolor('#FBFBFF')

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not display image:\n{str(e)}")

    def predict_single(self):
        """Run prediction on single image (Classification)"""
        if not self.current_image_path:
            return
        
        if not self.predictor:
            QMessageBox.critical(self, "Error", "Classification Model is not loaded!")
            return

        self.btn_predict.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Running classification...")

        # Create worker thread
        self.worker = PredictionWorker(self.predictor, self.current_image_path)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)
        self.worker.progress.connect(self.status_bar.showMessage)
        self.worker.start()

    def on_prediction_finished(self, result):
        """Handle classification prediction completion"""
        self.current_result = result

        # Display results
        prediction = result['prediction_label'].upper()
        confidence = result['confidence']
        color = "#D9534F" if prediction == "RETREATING" else "#4CAF50"
        
        results_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 14px; color: #040F16;">
            <h2 style="color: {color};">Status: {prediction}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
            <hr style="border: none; border-top: 1px solid #DDE2E8;">
            <p><b>Probabilities:</b></p>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>Stable: {result['probability_stable']:.1%}</li>
                <li>Retreating: {result['probability_retreating']:.1%}</li>
            </ul>
        </div>
        """
        self.results_text.setHtml(results_html)

        # Display key features
        features = result['features']
        features_text = f"""
Brightness: {features.get('brightness_mean', 0):.2f}
Red Mean: {features.get('red_mean', 0):.2f}
Edge Density: {features.get('edge_density', 0):.4f}
Glacier %: {features.get('glacier_percentage', 0):.2f}%
Texture Contrast: {features.get('contrast', 0):.2f}
Gradient Mean: {features.get('gradient_mean', 0):.4f}
        """
        self.features_text.setText(features_text)

        # Visualize
        self.visualize_prediction(result)

        # Re-enable controls
        self.progress_bar.setVisible(False)
        self.btn_predict.setEnabled(True)
        self.btn_export_single.setEnabled(True)
        self.status_bar.showMessage(f"Prediction complete: {prediction}", 5000)

        # Show popup
        QMessageBox.information(self, "Prediction Complete",
                               f"Glacier Status: {prediction}\nConfidence: {confidence:.1%}")

    def on_prediction_error(self, error_msg):
        """Handle classification prediction error"""
        self.progress_bar.setVisible(False)
        self.btn_predict.setEnabled(True)
        self.status_bar.showMessage("Prediction failed", 3000)
        QMessageBox.critical(self, "Prediction Error", f"Error:\n{error_msg}")

    def visualize_prediction(self, result):
        """Visualize classification prediction result"""
        image_data = result['image_data']
        prediction = result['prediction_label']
        confidence = result['confidence']

        # Normalize image
        rgb = image_data[:3].transpose(1, 2, 0)
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
        rgb_norm = np.clip(rgb_norm, 0, 1)

        # Clear and plot
        self.canvas_single.axes.clear()
        self.canvas_single.axes.imshow(rgb_norm)
        self.canvas_single.axes.set_facecolor('#FBFBFF')

        # Add colored border
        color = '#D9534F' if prediction == 'retreating' else '#4CAF50'
        for spine in self.canvas_single.axes.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(5)

        title = f"Prediction: {prediction.upper()}\nConfidence: {confidence:.1%}"
        self.canvas_single.axes.set_title(title, fontsize=14, fontweight='bold', color=color)
        self.canvas_single.axes.axis('off')
        self.canvas_single.draw()

    def export_single_result(self):
        """Export single classification prediction result"""
        if not self.current_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "PNG Image (*.png);;CSV File (*.csv)"
        )

        if file_path:
            try:
                if file_path.endswith('.png'):
                    self.canvas_single.fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=self.canvas_single.fig.get_facecolor())
                    self.status_bar.showMessage(f"Saved: {file_path}", 3000)
                elif file_path.endswith('.csv'):
                    df = pd.DataFrame([{
                        'image': os.path.basename(self.current_image_path),
                        'prediction': self.current_result['prediction_label'],
                        'confidence': self.current_result['confidence'],
                        'prob_stable': self.current_result['probability_stable'],
                        'prob_retreating': self.current_result['probability_retreating']
                    }])
                    df.to_csv(file_path, index=False)
                    self.status_bar.showMessage(f"Saved: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save file:\n{str(e)}")

    # --- NEWLY ADDED SLOTS FOR SEGMENTATION ---
    def load_segmentation_image(self):
        """Load a glacier image for SEGMENTATION"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Segmentation", "",
            "Image Files (*.tif *.tiff *.jpg *.jpeg *.png)"
        )

        if file_path:
            self.current_seg_image_path = file_path
            self.seg_image_info_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.btn_run_segmentation.setEnabled(True)
            self.status_bar.showMessage(f"Image loaded: {os.path.basename(file_path)}", 3000)

            # Display image in the *first* panel of the segmentation canvas
            fig = self.canvas_segmentation.fig
            fig.clear()
            ax1 = fig.add_subplot(1, 3, 1)
            self.display_image(file_path, ax1)
            
            # Add placeholder plots
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title("Segmentation (Pending)")
            ax2.axis('off')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title("Overlay (Pending)")
            ax3.axis('off')
            
            fig.tight_layout()
            self.canvas_segmentation.draw()

    def run_segmentation(self):
        """Run segmentation on the loaded image"""
        if not self.current_seg_image_path:
            return
        
        if not self.segmentation_predictor:
            QMessageBox.critical(self, "Error", "Segmentation Model is not loaded!")
            return

        self.btn_run_segmentation.setEnabled(False)
        self.seg_progress_bar.setVisible(True)
        self.seg_progress_bar.setRange(0, 0)  # Indeterminate
        self.status_bar.showMessage("Running segmentation...")

        # Create worker thread
        self.seg_worker = SegmentationWorker(self.segmentation_predictor, self.current_seg_image_path)
        self.seg_worker.finished.connect(self.on_segmentation_finished)
        self.seg_worker.error.connect(self.on_segmentation_error)
        self.seg_worker.progress.connect(self.status_bar.showMessage)
        self.seg_worker.start()

    def on_segmentation_finished(self, result):
        """Handle segmentation completion"""
        self.current_seg_result = result
        
        # Display stats
        stats = result['stats']
        total_pixels = result['segmentation'].size
        stats_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 14px; color: #040F16;">
            <p><b>Total Pixels: {total_pixels:,}</b></p>
            <hr style="border: none; border-top: 1px solid #DDE2E8;">
            <p><b>Class Distribution:</b></p>
            <ul style="list-style-type: none; padding-left: 0;">
        """
        
        for cls_id, count in stats.items():
            cls_name = self.segmentation_predictor.classes.get(cls_id, 'unknown').capitalize()
            percentage = (count / total_pixels) * 100
            stats_html += f"<li><b>{cls_name}:</b> {count:8,} pixels ({percentage:5.2f}%)</li>"
            
        stats_html += "</ul></div>"
        self.seg_results_text.setHtml(stats_html)

        # Visualize
        self.display_segmentation_result(result)

        # Re-enable controls
        self.seg_progress_bar.setVisible(False)
        self.btn_run_segmentation.setEnabled(True)
        self.btn_export_segmentation.setEnabled(True)
        self.status_bar.showMessage("Segmentation complete", 5000)

        QMessageBox.information(self, "Segmentation Complete", "Segmentation finished successfully.")

    def on_segmentation_error(self, error_msg):
        """Handle segmentation error"""
        self.seg_progress_bar.setVisible(False)
        self.btn_run_segmentation.setEnabled(True)
        self.status_bar.showMessage("Segmentation failed", 3000)
        QMessageBox.critical(self, "Segmentation Error", f"Error:\n{error_msg}")

    def display_segmentation_result(self, result):
        """
        Visualize segmentation result on the 3-panel canvas
        (This re-implements the logic from your script's `visualize` function)
        """
        segmentation = result['segmentation']
        image = result['image']
        
        # Normalize image for display
        if image.dtype != np.uint8:
            img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            img_norm = image
        
        # Color the segmentation
        seg_colored = self.segmentation_predictor.colors[segmentation]
        
        # Create overlay
        overlay = (img_norm * 0.6 + seg_colored * 0.4).astype(np.uint8)
        
        # Create figure
        fig = self.canvas_segmentation.fig
        fig.clear()
        
        # Original image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_norm)
        ax1.set_title('Original Image', fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        # Segmentation
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(seg_colored.astype(np.uint8))
        ax2.set_title('Segmentation Map', fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='cyan', label='Ice/Glacier'),
            Patch(facecolor='blue', label='Water/Ocean'),
            Patch(facecolor='gray', label='Other (Rock/Shadow)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # Overlay
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(overlay)
        ax3.set_title('Overlay', fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        fig.tight_layout()
        self.canvas_segmentation.draw()
        
    def export_segmentation_result(self):
        """Export segmentation visualization and stats"""
        if not self.current_seg_result:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Segmentation Results", "", "PNG Image (*.png);;Numpy Array (*.npy)"
        )

        if file_path:
            try:
                base_name, ext = os.path.splitext(file_path)
                
                if ext == '.png':
                    # Save the 3-panel visualization
                    self.canvas_segmentation.fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=self.canvas_segmentation.fig.get_facecolor())
                    self.status_bar.showMessage(f"Saved visualization: {file_path}", 3000)
                
                elif ext == '.npy':
                    # Save the raw segmentation array
                    np.save(file_path, self.current_seg_result['segmentation'])
                    self.status_bar.showMessage(f"Saved segmentation array: {file_path}", 3000)
                
                # Also save a stats .txt file
                stats_path = f"{base_name}_stats.txt"
                with open(stats_path, 'w') as f:
                    f.write(self.seg_results_text.toPlainText())
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save file(s):\n{str(e)}")
    # --- END NEW ---


    def select_batch_folder(self):
        """Select folder for batch processing"""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.batch_folder = folder
            self.batch_folder_label.setText(f"Selected: {folder}")
            self.btn_run_batch.setEnabled(True)

    def run_batch_prediction(self):
        """Run batch prediction (placeholder)"""
        QMessageBox.information(self, "Batch Processing",
                               "Batch processing will process all images in the selected folder.\n\n"
                               "This feature is being implemented...")

    def export_batch_results(self):
        """Export batch results (placeholder)"""
        pass

    def export_results(self, format_type):
        """Export session results"""
        QMessageBox.information(self, "Export", f"Export to {format_type.upper()} coming soon...")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look - ESSENTIAL for QSS to work well

    window = GlacierAnalysisApp()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()