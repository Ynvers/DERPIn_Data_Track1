# DERPIn Data - Track 1: Climate Smart Agriculture

## Overview

Welcome to Track 1 of the DERPIn Data initiative, focusing on **Climate Smart Agriculture**. This project aims to leverage data science and innovative technologies to address critical challenges in agriculture while promoting climate resilience and sustainability.

## Pre-trained Models

### Delineate-Anything Model

We utilize the **Delineate-Anything** model, a resolution-agnostic deep learning framework for accurate agricultural field boundary detection from satellite imagery.

- **Model**: Delineate Anything
- **Provider**: Mykola Lavreniuk et al. (Open Source)
- **Repository**: [Delineate-Anything GitHub](https://github.com/Lavreniuk/Delineate-Anything)
- **Paper**: [arXiv:2504.02534](https://arxiv.org/abs/2504.02534)
- **License**: AGPL-3.0

#### Available Models:
| Model | IoU | Boundary F1 | FPS | Size | Download |
|-------|-----|-------------|-----|------|----------|
| **Delineate Anything S** | 0.632 | 0.383 | 16.8 | 17.6 MB | [Download](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything-S.pt?download=true) |
| **Delineate Anything** | 0.720 | 0.477 | 25.0 | 125 MB | [Download](https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt?download=true) |

This model helps in:
- 🌾 **Field Boundary Detection**: Resolution-agnostic field boundary delineation
- 🛰️ **Multi-Resolution Support**: Works across different satellite imagery resolutions
- 📍 **Geographic Flexibility**: Trained on 22M+ global field instances (FBIS-22M dataset)
- 🎯 **High Accuracy**: SOTA performance with IoU up to 0.720
- 🚀 **Real-time Processing**: Up to 25 FPS inference speed

### Usage

The model can be used directly with satellite RGB imagery to automatically detect and vectorize agricultural field boundaries.

## Installation and Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU recommended (CUDA support) for faster processing
- **RAM**: 8GB minimum, 16GB recommended for large images
- **Storage**: 2GB free space for models and outputs

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/Ynvers/DERPIn_Data_Track1.git
cd DERPIn_Data_Track1
```

2. **Create Virtual Environment**
```bash
# Create virtual environment
python -m venv derpin

# Activate environment
.\derpin\Scripts\activate  # Windows
# or
source derpin/bin/activate  # Linux/Mac
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Additional Dashboard Dependencies** (if using interactive dashboard)
```bash
pip install streamlit-cropper
```

## Sample Data

### Sentinel-2 TCI Images

We use **True Color Images (TCI)** from Sentinel-2 satellites, available as Cloud Optimized GeoTIFFs (COG) from AWS S3.

#### Download Sample Data

**Example 1: East Africa (2020-07-01)**
```bash
# Download TCI image for East Africa region
wget "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/Q/WD/2020/7/S2A_36QWD_20200701_0_L2A/TCI.tif" -O TCI_EastAfrica_20200701.tif
```

**Example 2: Asia (2023-08-15)**
```bash
# Download TCI image for Asian region
wget "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/T/WN/2023/8/S2A_54TWN_20230815_0_L2A/TCI.tif" -O TCI_Asia_20230815.tif
```

**Example 3: African Cropland Dataset (AFCD) - 2022**
```bash
# Download Annual Cropland Mapping for Africa (30m resolution)
wget "https://zenodo.org/api/records/14920706/files/AFCD_2022.tif/content" -O AFCD_2022.tif
```


#### Image Specifications

**Sentinel-2 TCI Images:**
- **Format**: Cloud Optimized GeoTIFF (COG)
- **Bands**: RGB (Red, Green, Blue)
- **Resolution**: 10m per pixel
- **Projection**: UTM (varies by location)
- **Data Type**: 8-bit unsigned integer
- **Size**: ~50-1000 MB per image


#### Processing these images
Once downloaded, these TCI files can be processed with the Delineate-Anything model to detect agricultural field boundaries.

## Interactive Dashboard

This project includes an interactive Streamlit dashboard for easy field delineation from satellite imagery.

### Quick Start

1. **Environment Setup**
```bash
# Activate the virtual environment
.\derpin\Scripts\activate  # Windows
# or
source derpin/bin/activate  # Linux/Mac

# Install required packages
pip install -r requirements.txt
```

2. **Launch the Dashboard**
```bash
# Run the main dashboard
cd Dashboard
streamlit run main.py
```

3. **Dashboard Features**
   - 🔄 **File Upload**: Support for GeoTIFF files up to 500MB
   - 🎯 **Region Selection**: Interactive region of interest selection using sliders
   - 🖼️ **Image Preview**: Real-time satellite image visualization
   - 🧠 **AI Processing**: Automatic field detection using DelineateAnything model
   - 📄 **Results Export**: Download detected field boundaries as GeoJSON

### Dashboard Workflow

1. **Upload Image**: Choose a GeoTIFF satellite image (Sentinel-2 TCI or AFCD)
2. **Select Region**: Use percentage sliders to define your area of interest
3. **Preview Selection**: View the selected region before processing
4. **Run Detection**: Click "Detect Fields" to start AI analysis
5. **Download Results**: Get GeoJSON file with detected field boundaries

### Dashboard Versions

**Interactive Dashboard (`Dashboard/main.py`)**
- Visual cropping tool using `streamlit-cropper`
- Click-and-drag region selection
- Real-time crop preview
- Best for precise region selection

### Configuration

The dashboard can be customized via `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500  # Maximum file size in MB
```
