"""
Utilitaires de traitement d'images satellite pour le dashboard DERPIN.

Ce module contient des fonctions pour charger, traiter et afficher des images
satellite Sentinel-2, facilitant leur intégration dans un dashboard interactif.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from tqdm import tqdm
import cv2


def download_tci_image(url, filename):
    """
    Télécharge une image TCI depuis une URL.
    
    Args:
        url (str): URL de l'image à télécharger
        filename (str): Nom du fichier local où sauvegarder l'image
        
    Returns:
        str: Chemin vers le fichier téléchargé
        
    Raises:
        Exception: En cas d'erreur de téléchargement
    """
    if Path(filename).exists():
        print(f"✅ {filename} déjà présent")
        return filename

    print(f"⬇️  Téléchargement de {filename}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as f, tqdm(
        desc=f"Téléchargement {filename}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"✅ {filename} téléchargé avec succès!")
    return filename


def load_geotiff(tif_path):
    """
    Charge un fichier GeoTIFF et retourne les métadonnées importantes.
    
    Args:
        tif_path (str): Chemin vers le fichier GeoTIFF
        
    Returns:
        tuple: (metadata, src) où metadata est un dict avec des infos clés 
               et src est l'objet rasterio ouvert (à fermer après usage)
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    if not Path(tif_path).exists():
        raise FileNotFoundError(f"Fichier introuvable: {tif_path}")

    src = rasterio.open(tif_path)
    
    metadata = {
        'crs': src.crs,
        'transform': src.transform,
        'bounds': src.bounds,
        'width': src.width,
        'height': src.height,
        'resolution': src.res,
        'bands': src.count
    }
    
    return metadata, src


def calculate_global_stretch(src, bands=None, percentiles=(0.5, 99.5), sample_size=8):
    """
    Calcule les paramètres d'étirement pour normaliser l'image entière.
    
    Args:
        src: Source rasterio ouverte
        bands (list): Indices des bandes à utiliser (1-indexed)
        percentiles (tuple): Percentiles bas et haut pour l'étirement
        sample_size (int): Facteur de sous-échantillonnage pour l'estimation
        
    Returns:
        tuple: (min, max) pour normaliser l'image
    """
    if bands is None:
        bands = [1, 2, 3] if src.count == 3 else [4, 3, 2]  # S2: [4,3,2] = R,G,B
    
    H, W = src.height, src.width
    preview = src.read(bands, out_shape=(len(bands), H//sample_size, W//sample_size))
    preview = preview.transpose(1, 2, 0).astype(np.float32)
    
    lo = float(np.percentile(preview, percentiles[0]))
    hi = float(np.percentile(preview, percentiles[1]))
    
    return lo, hi


def normalize_to_uint8(img, lo, hi):
    """
    Normalise une image à l'aide des valeurs min/max calculées.
    
    Args:
        img (numpy.ndarray): Image à normaliser
        lo (float): Valeur minimale pour l'étirement
        hi (float): Valeur maximale pour l'étirement
        
    Returns:
        numpy.ndarray: Image normalisée en uint8
    """
    x = img.astype(np.float32)
    x = np.clip((x - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
    return x.round().astype(np.uint8)


def load_and_display_tci(tif_path, figsize=(12, 8)):
    """
    Charge et affiche une image TCI pour visualisation.
    
    Args:
        tif_path (str): Chemin vers le fichier TCI
        figsize (tuple): Taille de la figure matplotlib
        
    Returns:
        tuple: (rgb_image, crs, transform) données de l'image et métadonnées
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    if not Path(tif_path).exists():
        print(f"❌ Fichier introuvable: {tif_path}")
        return None
    
    print(f"📖 Chargement de {tif_path}...")
    
    with rasterio.open(tif_path) as src:
        # Bandes RGB (ou sélection appropriée)
        bands = [1, 2, 3] if src.count == 3 else [4, 3, 2]
        rgb_data = src.read(bands)
        
        # Transposer pour matplotlib (H, W, C)
        rgb_image = np.transpose(rgb_data, (1, 2, 0))
        
        # Normaliser avec étirement global
        lo, hi = calculate_global_stretch(src, bands)
        rgb_image = normalize_to_uint8(rgb_image, lo, hi)
        
        print(f"✅ Image chargée: {rgb_image.shape}")
        print(f"   CRS: {src.crs}")
        print(f"   Résolution: {src.res}")
        print(f"   Bounds: {src.bounds}")
        
        # Affichage
        plt.figure(figsize=figsize)
        plt.imshow(rgb_image)
        plt.title(f"Image TCI Sentinel-2\n{Path(tif_path).name}")
        plt.axis('off')
        plt.show()
        
        return rgb_image, src.crs, src.transform


def get_tile_from_image(src, x, y, tile_size, apply_stretch=True):
    """
    Extrait une tuile d'une image rasterio avec étirement global.
    
    Args:
        src: Source rasterio ouverte
        x (int): Coordonnée x du coin supérieur gauche de la tuile
        y (int): Coordonnée y du coin supérieur gauche de la tuile
        tile_size (int): Taille de la tuile (carré)
        apply_stretch (bool): Appliquer l'étirement global
        
    Returns:
        numpy.ndarray: Tuile extraite, format RGB uint8
    """
    win = rasterio.windows.Window(
        x, y,
        min(tile_size, src.width - x),
        min(tile_size, src.height - y)
    )
    
    # Déterminer les bandes RGB
    bands = [1, 2, 3] if src.count == 3 else [4, 3, 2]
    
    # Lire la tuile
    tile = src.read(bands, window=win).transpose(1, 2, 0)
    
    if apply_stretch:
        # Appliquer l'étirement global
        lo, hi = calculate_global_stretch(src, bands)
        tile = normalize_to_uint8(tile, lo, hi)
    else:
        # Normalisation simple si nécessaire
        if tile.dtype == np.uint16:
            tile = (tile / 65535.0 * 255).astype(np.uint8)
        elif tile.max() > 255:
            tile = (tile / tile.max() * 255).astype(np.uint8)
    
    return tile


def convert_to_bgr(img):
    """
    Convertit une image RGB en BGR (pour OpenCV ou Ultralytics).
    
    Args:
        img (numpy.ndarray): Image RGB
        
    Returns:
        numpy.ndarray: Image BGR
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def convert_to_rgb(img):
    """
    Convertit une image BGR en RGB.
    
    Args:
        img (numpy.ndarray): Image BGR
        
    Returns:
        numpy.ndarray: Image RGB
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
