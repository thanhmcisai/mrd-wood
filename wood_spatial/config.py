"""
Wood Spatial — Central Configuration
======================================
Single source of truth for backbones, perturbations, datasets, paths, and display constants.

Path configuration:
    Set WOOD_BASE env var to override, or defaults to parent of this package.
"""
import os
from pathlib import Path

# ── Base directory ──────────────────────────────────────────────────────────
BASE = Path(os.environ.get('WOOD_BASE', Path(__file__).resolve().parent.parent))

# ── Derived paths ───────────────────────────────────────────────────────────
DATASET_ROOT = Path(os.environ.get('WOOD_DATASETS_DIR', BASE / 'dataset'))
V2_CACHE_DIR = BASE / 'results_v2' / 'feature_cache'   # reuse existing caches
# ``results/`` is the canonical local paper-output tree. Colab and external
# runs must set WOOD_RESULTS_DIR explicitly (the full-run driver does this from
# its JSON config), so local analysis cannot silently read legacy results_v4.
V4_DIR       = Path(os.environ.get('WOOD_RESULTS_DIR', BASE / 'results'))
V4_FEAT_CACHE   = V4_DIR / 'feature_cache'
V4_SPATIAL_CACHE = V4_DIR / 'spatial_cache'
V4_GRADCAM_CACHE = V4_DIR / 'gradcam_cache'
V4_CSV       = V4_DIR / 'csv'
V4_FIGURES   = V4_DIR / 'figures'

def ensure_dirs():
    for d in [V4_FEAT_CACHE, V4_SPATIAL_CACHE, V4_GRADCAM_CACHE, V4_CSV, V4_FIGURES]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# ── ImageNet normalization ──────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF'}

# ── Hyperparameters ─────────────────────────────────────────────────────────
IMG_SIZE        = 224
RANDOM_SEED     = 42
BATCH_SIZE      = 32
NUM_WORKERS     = int(os.environ.get('WOOD_NUM_WORKERS', '4'))
FP16            = True
KNN_K           = 5
BOOTSTRAP_B     = 1000
BOOTSTRAP_ALPHA = 0.05

# ── 7 Backbone Configurations ──────────────────────────────────────────────
BACKBONE_CONFIGS = {
    'resnet50': {
        'model_id': 'resnet50',
        'img_size': 224, 'batch': 32, 'extra': {},
    },
    'efficientnet_b3': {
        'model_id': 'efficientnet_b3',
        'img_size': 300, 'batch': 16, 'extra': {},
    },
    'convnext_tiny': {
        'model_id': 'convnext_tiny',
        'img_size': 224, 'batch': 32, 'extra': {},
    },
    'swin_tiny': {
        'model_id': 'swin_tiny_patch4_window7_224',
        'img_size': 224, 'batch': 32, 'extra': {},
    },
    'dinov2_b': {
        'model_id': 'vit_base_patch14_dinov2.lvd142m',
        'img_size': 224, 'batch': 16,
        'extra': {'dynamic_img_size': True}, 'no_global_pool': True,
    },
    'hrnet32': {
        'model_id': 'hrnet_w32',
        'img_size': 224, 'batch': 16, 'extra': {},
    },
    'mobilenetv3': {
        'model_id': 'mobilenetv3_large_100',
        'img_size': 224, 'batch': 32, 'extra': {},
    },
}

# Canonical ordering for all figures/tables
BB_ORDER = [
    'resnet50', 'efficientnet_b3', 'convnext_tiny',
    'swin_tiny', 'dinov2_b',
    'hrnet32', 'mobilenetv3',
]

BB_LABEL = {
    'resnet50':        'ResNet-50',
    'efficientnet_b3': 'EffNet-B3',
    'convnext_tiny':   'ConvNeXt-T',
    'swin_tiny':       'Swin-T',
    'dinov2_b':        'DINOv2-B',
    'hrnet32':         'HRNet-32',
    'mobilenetv3':     'MobileNetV3-L',
}

BB_COLOR = {
    'resnet50':        '#E24B4A',
    'efficientnet_b3': '#0D7055',
    'convnext_tiny':   '#E87A29',
    'swin_tiny':       '#B85A10',
    'dinov2_b':        '#7F77DD',
    'hrnet32':         '#0F6E56',
    'mobilenetv3':     '#1F77B4',
}

BB_MARKER = {
    'resnet50': 'o',   'efficientnet_b3': '<',
    'convnext_tiny': 'D', 'swin_tiny': 'h',
    'dinov2_b': '*',   'hrnet32': 'P',
    'mobilenetv3': 'X',
}

BB_ARCH = {
    'resnet50':        {'family': 'CNN',         'pretrain': 'IN1k',    'params_M': 25},
    'efficientnet_b3': {'family': 'CNN',         'pretrain': 'IN1k',    'params_M': 12},
    'convnext_tiny':   {'family': 'CNN',         'pretrain': 'IN1k',    'params_M': 28},
    'swin_tiny':       {'family': 'Transformer', 'pretrain': 'IN1k',    'params_M': 28},
    'dinov2_b':        {'family': 'Transformer', 'pretrain': 'LVD142M', 'params_M': 86},
    'hrnet32':         {'family': 'HRNet',       'pretrain': 'IN1k',    'params_M': 28},
    'mobilenetv3':     {'family': 'Lightweight',  'pretrain': 'IN1k',    'params_M': 5},
}

FAMILY_COLOR = {
    'CNN': '#E87A29', 'Transformer': '#534AB7',
    'HRNet': '#0F6E56', 'Lightweight': '#1F77B4',
}

# ── Perturbation Configurations ────────────────────────────────────────────
PERTURB_CONFIGS = {
    'gaussian_blur': {
        'type': 'blur',
        'values': [2, 4, 6, 8, 10, 12],
    },
    'defocus_blur': {
        'type': 'defocus',
        'values': [3, 5, 8, 11, 15],
    },
    'resize': {
        'type': 'resize',
        'values': [1.33, 1.67, 2.00],
    },
    'illumination': {
        'type': 'brightness',
        'values': [0.70, 0.85, 1.15, 1.30],
    },
    'jpeg': {
        'type': 'jpeg',
        'values': [10, 20, 30, 50, 70],
    },
    # Legacy red-channel caches are stored as color_shift_{offset}.
    'red_channel_shift': {
        'type': 'channel',
        'channel': 0,  # Red channel
        'values': [-45, -30, 30, 45],
    },
    'green_channel_shift': {
        'type': 'channel',
        'channel': 1,  # Green channel
        'values': [-45, -30, 30, 45],
    },
    'blue_channel_shift': {
        'type': 'channel',
        'channel': 2,  # Blue channel
        'values': [-45, -30, 30, 45],
    },
    'gaussian_noise': {
        'type': 'gaussian_noise',
        'values': [0.02, 0.05, 0.10],
    },
    'shot_noise': {
        'type': 'shot_noise',
        # Poisson level: lower values produce stronger signal-dependent noise.
        'values': [60, 30, 15],
    },
    'impulse_noise': {
        'type': 'impulse_noise',
        'values': [0.01, 0.03, 0.05],
    },
    'motion_blur': {
        'type': 'motion_blur',
        'values': [5, 9, 15],
    },
    'zoom_blur': {
        'type': 'zoom_blur',
        'values': [1.05, 1.10, 1.20],
    },
    'contrast': {
        'type': 'contrast',
        'values': [0.50, 0.75, 1.25, 1.50],
    },
    'pixelate': {
        'type': 'pixelate',
        'values': [0.50, 0.33, 0.25],
    },
    'scratch': {
        'type': 'scratch',
        'values': ['mild', 'moderate', 'severe'],
    },
    'rotation': {
        'type': 'rotation',
        'values': [45, 90, 135, 180],
    },
    'compound': {
        'type': 'compound',
        'values': ['mild', 'moderate', 'severe'],
    },
    'compound_optical': {
        'type': 'compound',
        'preset_prefix': 'optical',
        'values': ['mild', 'moderate', 'severe'],
    },
    'compound_digital': {
        'type': 'compound',
        'preset_prefix': 'digital',
        'values': ['mild', 'moderate', 'severe'],
    },
    'compound_field': {
        'type': 'compound',
        'preset_prefix': 'field',
        'values': ['mild', 'moderate', 'severe'],
    },
}

PT_LABEL = {
    'gaussian_blur': 'Gaussian Blur',
    'defocus_blur':  'Defocus Blur',
    'resize':        'Resize',
    'illumination':  'Illumination',
    'jpeg':          'JPEG Compression',
    'color_shift':   'Red-channel Shift',
    'red_channel_shift':   'Red-channel Shift',
    'green_channel_shift': 'Green-channel Shift',
    'blue_channel_shift':  'Blue-channel Shift',
    'gaussian_noise': 'Gaussian Noise',
    'shot_noise':    'Shot Noise',
    'impulse_noise': 'Impulse Noise',
    'motion_blur':   'Motion Blur',
    'zoom_blur':     'Zoom Blur',
    'contrast':      'Contrast',
    'pixelate':      'Pixelate',
    'scratch':       'Scratch Artifact',
    'rotation':      'Rotation',
    'compound':      'Compound',
    'compound_optical': 'Compound Optical',
    'compound_digital': 'Compound Digital',
    'compound_field':   'Compound Field',
}

PT_COLOR = {
    'gaussian_blur': '#E24B4A',
    'defocus_blur':  '#C43030',
    'resize':        '#534AB7',
    'illumination':  '#DAA520',
    'jpeg':          '#8B4513',
    'color_shift':   '#FF8888',
    'red_channel_shift':   '#FF8888',
    'green_channel_shift': '#40A857',
    'blue_channel_shift':  '#4C78A8',
    'gaussian_noise': '#7F7F7F',
    'shot_noise':    '#9A9A9A',
    'impulse_noise': '#5F5F5F',
    'motion_blur':   '#D55E00',
    'zoom_blur':     '#CC79A7',
    'contrast':      '#F0A202',
    'pixelate':      '#009E73',
    'scratch':       '#795548',
    'rotation':      '#0F6E56',
    'compound':      '#888888',
    'compound_optical': '#6E6E6E',
    'compound_digital': '#4E4E4E',
    'compound_field':   '#2E2E2E',
}

PT_GROUP = {
    'gaussian_blur': 'Blur',    'defocus_blur': 'Blur',
    'resize': 'Geometric',      'rotation': 'Geometric',
    'illumination': 'Photometric', 'jpeg': 'Photometric',
    'color_shift': 'Photometric',
    'red_channel_shift': 'Photometric',
    'green_channel_shift': 'Photometric',
    'blue_channel_shift': 'Photometric',
    'contrast': 'Photometric',
    'pixelate': 'Digital',
    'scratch': 'Surface artifact',
    'gaussian_noise': 'Noise',
    'shot_noise': 'Noise',
    'impulse_noise': 'Noise',
    'motion_blur': 'Blur',
    'zoom_blur': 'Blur',
    'compound': 'Compound',
    'compound_optical': 'Compound',
    'compound_digital': 'Compound',
    'compound_field': 'Compound',
}

# Compound perturbation definitions
COMPOUND_PRESETS = {
    'mild':     [('blur', 4),  ('brightness', 0.85), ('jpeg', 50)],
    'moderate': [('blur', 8),  ('brightness', 0.70), ('jpeg', 30)],
    'severe':   [('blur', 12), ('brightness', 0.70), ('jpeg', 10)],

    # Optical/acquisition failures: focus error, hand/camera motion, and lighting.
    'optical_mild':     [('defocus', 3),  ('motion_blur', 5),  ('brightness', 0.85)],
    'optical_moderate': [('defocus', 8),  ('motion_blur', 9),  ('brightness', 0.70)],
    'optical_severe':   [('defocus', 15), ('motion_blur', 15), ('brightness', 0.70)],

    # Digital pipeline failures: contrast mapping, pixel resampling, and compression.
    'digital_mild':     [('contrast', 0.75), ('pixelate', 0.50), ('jpeg', 70)],
    'digital_moderate': [('contrast', 0.50), ('pixelate', 0.33), ('jpeg', 30)],
    'digital_severe':   [('contrast', 1.50), ('pixelate', 0.25), ('jpeg', 10)],

    # Field-like mixed failures: sensor noise, color response, zoom blur, and compression.
    'field_mild':     [('gaussian_noise', 0.02), ('blue_channel_shift', 30),  ('zoom_blur', 1.05), ('jpeg', 70)],
    'field_moderate': [('gaussian_noise', 0.05), ('green_channel_shift', -30), ('zoom_blur', 1.10), ('jpeg', 30)],
    'field_severe':   [('gaussian_noise', 0.10), ('blue_channel_shift', -45),  ('zoom_blur', 1.20), ('jpeg', 10)],
}

# ── Dataset Metadata ───────────────────────────────────────────────────────
ALL_DATASETS = {
    'WRD25': {
        'root': str(DATASET_ROOT / 'WRD25'),
        'region': 'Vietnam', 'camera': 'smartphone_20x_lens',
        'n_classes': 25, 'n_images': 2167,
        'acquisition': 'OnePlus 3 smartphone through 20x magnifying glass; center 300x300 crop',
    },
    'DTSR14': {
        'root': str(DATASET_ROOT / 'DTSR14'),
        'region': 'Europe', 'camera': 'flatbed_scanner',
        'n_classes': 14, 'n_images': 8903,
        'acquisition': 'Scanned RGB wood-core images at 600/1200 dpi; local folder contains patches/crops',
    },
    'PCA11': {
        'root': str(DATASET_ROOT / 'PCA11'),
        'region': 'C.America', 'camera': 'digital_magnifying_glass',
        'n_classes': 11, 'n_images': 10795,
        'acquisition': 'Warehouse acquisition with digital magnifying glass, 640x480 pixels',
    },
    'BFS46': {
        'root': str(DATASET_ROOT / 'BFS46'),
        'region': 'Brazil', 'camera': 'zeiss_stereomicroscope',
        'n_classes': 46, 'n_images': 1901,
        'acquisition': 'Zeiss Discovery V12 stereomicroscope, 10x, 2080x1540 pixels, 150 dpi',
    },
    'FSDM41': {
        'root': str(DATASET_ROOT / 'FSDM41'),
        'region': 'Brazil', 'camera': 'sony_dsc_t20_macro_box',
        'n_classes': 41, 'n_images': 2942,
        'acquisition': 'Sony DSC-T20 macro images in a halogen-lit acquisition box, 3264x2448 pixels',
    },
    'GOIMAI': {
        'root': str(DATASET_ROOT / 'GOIMAI'),
        'region': 'Tropical', 'camera': 'smartphone_24x_lens',
        'n_classes': 37, 'n_images': 2121,
        'acquisition': 'Smartphone camera with attached 24x optical magnifying lens, 3000x4000 pixels',
    },
    'WOODAUTH': {
        'root': str(DATASET_ROOT / 'WOODAUTH'),
        'region': 'Europe', 'camera': 'nikon_d3300',
        'n_classes': 12, 'n_images': 8561,
        'acquisition': 'Nikon D3300 digital camera from 15-20 cm; images cropped to 400x400',
    },
    'BD11': {
        'root': str(DATASET_ROOT / 'BD11'),
        'region': 'Brazil', 'camera': 'portable_microscope_smartphone',
        'n_classes': 11, 'n_images': 440,
        'acquisition': 'Low-cost 640x480 portable microscope connected to a smartphone',
    },
    'VN26_x10': {
        'root': str(DATASET_ROOT / 'VN26' / 'x10'),
        'region': 'Vietnam', 'camera': 'smartphone_macro_lens',
        'n_classes': 26, 'n_images': 2600,
        'acquisition': 'OnePlus 3 smartphone with macro/magnifying lens; x10 split',
    },
    'VN26_x20': {
        'root': str(DATASET_ROOT / 'VN26' / 'x20'),
        'region': 'Vietnam', 'camera': 'smartphone_macro_lens',
        'n_classes': 26, 'n_images': 2600,
        'acquisition': 'OnePlus 3 smartphone with macro/magnifying lens; x20 split',
    },
    'VN26_x50': {
        'root': str(DATASET_ROOT / 'VN26' / 'x50'),
        'region': 'Vietnam', 'camera': 'smartphone_macro_lens',
        'n_classes': 26, 'n_images': 2600,
        'acquisition': 'OnePlus 3 smartphone with macro/magnifying lens; x50 split',
    },
}

# Dataset tier groupings
TIER_A = ['WRD25', 'DTSR14', 'PCA11']
TIER_B = ['BFS46', 'FSDM41', 'GOIMAI', 'WOODAUTH', 'BD11']
# Legacy name: older scripts use TIER_C for VN26 magnification subsets.
TIER_C = ['VN26_x10', 'VN26_x20', 'VN26_x50']
TIER_D = TIER_C
CROSS_SOURCE_PAIRS = [('BFS46', 'FSDM41'), ('DTSR14', 'WOODAUTH')]

DS_LABEL = {
    'WRD25': 'WRD25 (Phone+20x)', 'DTSR14': 'DTSR14 (Scanner)', 'PCA11': 'PCA11 (Magnifier)',
    'BFS46': 'BFS46 (Stereo)', 'FSDM41': 'FSDM41 (Macro box)',
    'GOIMAI': 'GOIMAI (Phone+24x)', 'WOODAUTH': 'WOODAUTH (DSLR)', 'BD11': 'BD11 (Phone microscope)',
    'VN26_x10': 'VN26 x10', 'VN26_x20': 'VN26 x20', 'VN26_x50': 'VN26 x50',
}

# ── Spatial Clustering ─────────────────────────────────────────────────────
N_CLUSTERS     = 3
GUIDED_RADIUS  = 16
GUIDED_EPS     = 1e-4
VESSEL_MIN_AREA = 100
MORPH_KERNEL_SIZE = 3

# Cluster names (ordered by brightness: darkest=0, brightest=2)
# Named generically — brightness ordering approximates tissue types
# but cluster-to-tissue correspondence is not manually validated
CLUSTER_NAMES = {0: 'C1 (Darkest)', 1: 'C2', 2: 'C3 (Lightest)'}

CLUSTER_COLOR = {
    0: '#8B0000',  # dark red
    1: '#228B22',  # green
    2: '#4169E1',  # blue
}

# Subset size for spatial analysis (per dataset)
SPATIAL_MAX_IMAGES = 200
SPATIAL_BATCH_SIZE = 1  # batch=1 for spatial to fit on 4GB GPU
