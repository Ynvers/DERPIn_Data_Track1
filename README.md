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

**African Cropland Dataset (AFCD):**
- **Format**: GeoTIFF (AFCD_YYYY.tif, where YYYY is the year)
- **Resolution**: 30m per pixel
- **Bands**: Single band cropland mask
- **Data Values**: 
  - `1` = Cultivated area
  - `0` = Non-cultivated area
- **Coverage**: Annual cropland mapping product covering Africa
- **Source**: [Zenodo Record](https://zenodo.org/api/records/14920706)

#### Processing these images
Once downloaded, these TCI files can be processed with the Delineate-Anything model to detect agricultural field boundaries.
