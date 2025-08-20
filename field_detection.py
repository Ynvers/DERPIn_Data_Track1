"""
Module de détection des champs avec DelineateAnything.

Ce module implémente les fonctions nécessaires pour détecter les délimitations de champs
à l'aide du modèle DelineateAnything, en prenant en compte le traitement par tuiles
et la génération de résultats géoréférencés.
"""

import os
import numpy as np
import cv2
import json
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
from ultralytics import YOLO
from pathlib import Path


class FieldDelineator:
    """Classe pour détecter les délimitations de champs dans les images satellites."""
    
    def __init__(self, model_path="DelineateAnything.pt"):
        """
        Initialise le détecteur de champs.
        
        Args:
            model_path (str): Chemin vers le fichier de modèle DelineateAnything
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Charge le modèle DelineateAnything.
        
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            print("🛰️ Chargement du modèle Delineate-Anything...")
            self.model = YOLO(self.model_path)
            print("✅ Modèle chargé avec succès!")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def process_region(self, tif_path, region=None, output_prefix="output",
                      tile_size=1024, overlap=128, conf_thr=0.35, iou_thr=0.5):
        """
        Traite une région d'une image satellite pour détecter les champs.
        
        Args:
            tif_path (str): Chemin vers le fichier GeoTIFF
            region (tuple): (x_min, y_min, width, height) définissant la région à traiter
                           None = traiter toute l'image
            output_prefix (str): Préfixe pour les fichiers de sortie
            tile_size (int): Taille des tuiles pour traitement par morceaux
            overlap (int): Chevauchement entre les tuiles
            conf_thr (float): Seuil de confiance pour le modèle
            iou_thr (float): Seuil IoU pour la suppression des non-maximums
            
        Returns:
            dict: Chemins des fichiers générés (overlay, geojson)
        """
        if self.model is None:
            print("❌ Modèle non chargé! Utilisez load_model() d'abord.")
            return None
            
        # Chemins des fichiers de sortie
        overlay_path = f"{output_prefix}_overlay.png"
        geojson_path = f"{output_prefix}_fields.geojson"
        
        # Variables pour collecter les résultats
        features = []  # Pour GeoJSON
        
        with rasterio.open(tif_path) as src:
            # Définir la région à traiter
            if region:
                x_min, y_min, width, height = region
                x_max = min(x_min + width, src.width)
                y_max = min(y_min + height, src.height)
            else:
                x_min, y_min = 0, 0
                x_max, y_max = src.width, src.height
                width, height = src.width, src.height
            
            # Récupérer les métadonnées importantes
            crs = src.crs
            transform = src.transform
            bands = [1,2,3] if src.count == 3 else [4,3,2]  # S2: [4,3,2] = R,G,B
            
            # Calculer l'étirement global pour toute l'image
            preview = src.read(bands, 
                window=rasterio.windows.Window(x_min, y_min, width, height),
                out_shape=(len(bands), height//8, width//8)).transpose(1,2,0).astype(np.float32)
            lo = float(np.percentile(preview, 0.5))
            hi = float(np.percentile(preview, 99.5))
            
            def to_uint8_global(img):
                x = img.astype(np.float32)
                x = np.clip((x - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
                return x.round().astype(np.uint8)
                
            # Lire la région complète pour l'overlay
            region_img = src.read(
                bands,
                window=rasterio.windows.Window(x_min, y_min, width, height)
            ).transpose(1, 2, 0)
            
            # Normaliser avec le même étirement pour toute la région
            overlay = to_uint8_global(region_img)
            
            # Traiter par tuiles avec chevauchement
            stride = tile_size - overlap
            print(f"🚀 Lancement de la détection (tuile par tuile)")
            
            # Parcourir les tuiles dans la région
            for y in range(y_min, y_max, stride):
                for x in range(x_min, x_max, stride):
                    # Définir la fenêtre pour cette tuile
                    win = rasterio.windows.Window(
                        x, y,
                        min(tile_size, x_max - x),
                        min(tile_size, y_max - y)
                    )
                    
                    # Extraire la tuile
                    tile = src.read(bands, window=win).transpose(1, 2, 0)
                    tile = to_uint8_global(tile)
                    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)  # Ultralytics attend BGR
                    
                    try:
                        # Prédiction avec le modèle
                        preds = self.model.predict(
                            source=tile,
                            conf=conf_thr,
                            iou=iou_thr,
                            imgsz=max(960, max(win.width, win.height)),  # Détail
                            retina_masks=True,                           # Masques haute résolution
                            agnostic_nms=True,                          
                            verbose=False
                        )
                    except Exception:
                        # Fallback en cas de mémoire insuffisante
                        preds = self.model.predict(
                            source=tile, conf=conf_thr, iou=iou_thr,
                            imgsz=640, retina_masks=True, agnostic_nms=True, verbose=False
                        )
                    
                    # Traitement des prédictions
                    for r in preds:
                        if getattr(r, "masks", None) is None:
                            continue
                        
                        # Cas 1: Ultralytics fournit les polygones directement
                        if getattr(r.masks, "xy", None):
                            for poly in r.masks.xy:
                                if poly is None or len(poly) < 3:
                                    continue
                                
                                # Traitement du polygone
                                poly_processed = self._process_polygon(
                                    poly, x, y, x_min, y_min, transform, overlay, features
                                )
                                
                        # Cas 2: Extraction des contours depuis le masque
                        elif getattr(r.masks, "data", None) is not None:
                            mdata = r.masks.data.cpu().numpy()
                            for m in mdata:
                                m = (m > 0.5).astype(np.uint8)*255
                                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in cnts:
                                    if len(cnt) < 3:
                                        continue
                                    
                                    # Traitement du contour comme polygone
                                    cnt = cnt.reshape(-1,2).astype(np.float32)
                                    self._process_polygon(
                                        cnt, x, y, x_min, y_min, transform, overlay, features
                                    )
                    
                    # Afficher la progression
                    print(f"✅ tuile ({x},{y}) traitée")
            
            # Calculer les coordonnées de l'overlay dans l'image originale
            rel_x, rel_y = x_min - x_min, y_min - y_min  # Relatif à la région extraite
            
        # Sauvegarder l'overlay
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_path, overlay_rgb)
        
        # Sauvegarder le GeoJSON
        geojson = {
            "type": "FeatureCollection", 
            "features": features,
            "crs": {"type": "name", "properties": {"name": str(crs)}}
        }
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)
        
        print("🎉 Traitement terminé !")
        print(f" - {overlay_path} : image avec les délimitations")
        print(f" - {geojson_path} : polygones géoréférencés")
        
        return {
            "overlay": overlay_path,
            "geojson": geojson_path
        }
    
    def _process_polygon(self, poly, tile_x, tile_y, region_x, region_y, transform, overlay, features):
        """
        Traite un polygone détecté: ajustement des coordonnées, dessin et conversion en GeoJSON.
        
        Args:
            poly: Points du polygone
            tile_x, tile_y: Coordonnées de la tuile dans l'image complète
            region_x, region_y: Coordonnées de la région dans l'image complète
            transform: Transform rasterio pour conversion en coordonnées géographiques
            overlay: Image overlay pour dessiner le polygone
            features: Liste des features GeoJSON à mettre à jour
            
        Returns:
            np.ndarray: Points du polygone ajustés
        """
        # Convertir en numpy array si ce n'est pas déjà le cas
        poly = np.array(poly, dtype=np.float32)
        
        # Ajuster les coordonnées à l'image complète
        poly[:, 0] += tile_x
        poly[:, 1] += tile_y
        
        # Ajuster les coordonnées à la région extraite (pour l'overlay)
        poly_overlay = poly.copy()
        poly_overlay[:, 0] -= region_x
        poly_overlay[:, 1] -= region_y
        
        # Dessiner sur l'overlay
        cv2.polylines(overlay, [poly_overlay.astype(np.int32)], True, (255,255,255), 2)
        
        # Convertir en coordonnées géographiques et ajouter au GeoJSON
        pts_geo = []
        a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        for cx, cy in poly:
            X = c + a*cx + b*cy  # Conversion en coordonnées du CRS
            Y = f + d*cx + e*cy
            pts_geo.append((X, Y))
            
        # Créer le polygone et vérifier sa validité
        if len(pts_geo) >= 3:
            P = Polygon(pts_geo)
            if P.is_valid and not P.is_empty and P.area > 0:
                features.append({
                    "type": "Feature",
                    "properties": {},
                    "geometry": mapping(P)
                })
                
        return poly
        
    def detect_single_tile(self, tile_img, conf_thr=0.35, iou_thr=0.5):
        """
        Détecte les champs sur une seule tuile d'image.
        Utile pour les applications interactives avec des tuiles sélectionnées.
        
        Args:
            tile_img (numpy.ndarray): Image tuile (format BGR pour YOLO)
            conf_thr (float): Seuil de confiance pour la détection
            iou_thr (float): Seuil IoU pour NMS
            
        Returns:
            list: Liste de polygones détectés (format array numpy)
        """
        if self.model is None:
            print("❌ Modèle non chargé! Utilisez load_model() d'abord.")
            return []
            
        polygons = []
        
        try:
            # Exécuter la prédiction
            preds = self.model.predict(
                source=tile_img,
                conf=conf_thr,
                iou=iou_thr,
                imgsz=max(960, max(tile_img.shape[0], tile_img.shape[1])),
                retina_masks=True,
                agnostic_nms=True,
                verbose=False
            )
            
            # Récupérer les polygones des prédictions
            for r in preds:
                if getattr(r, "masks", None) is None:
                    continue
                    
                # Cas 1: Polygones directement disponibles
                if getattr(r.masks, "xy", None):
                    for poly in r.masks.xy:
                        if poly is not None and len(poly) >= 3:
                            polygons.append(np.array(poly, dtype=np.float32))
                
                # Cas 2: Extraction des contours depuis les masques
                elif getattr(r.masks, "data", None) is not None:
                    mdata = r.masks.data.cpu().numpy()
                    for m in mdata:
                        m = (m > 0.5).astype(np.uint8)*255
                        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            if len(cnt) >= 3:
                                polygons.append(cnt.reshape(-1, 2).astype(np.float32))
                                
        except Exception as e:
            print(f"❌ Erreur lors de la détection: {e}")
            
        return polygons


def download_model(url=None):
    """
    Télécharge le modèle DelineateAnything si nécessaire.
    
    Args:
        url (str): URL optionnelle du modèle (sinon utilise HuggingFace)
        
    Returns:
        str: Chemin vers le fichier du modèle
    """
    model_path = "DelineateAnything.pt"
    
    if Path(model_path).exists():
        print(f"✅ Modèle {model_path} déjà présent")
        return model_path
    
    if url is None:
        # Par défaut, récupérer depuis HuggingFace
        try:
            import subprocess
            print("⬇️ Téléchargement du modèle depuis HuggingFace...")
            subprocess.run(["wget", "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt", 
                           "-O", model_path], check=True)
            print("✅ Modèle téléchargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur téléchargement: {e}")
            print("📝 Essayez de télécharger manuellement le modèle depuis:")
            print("https://huggingface.co/MykolaL/DelineateAnything/blob/main/DelineateAnything.pt")
    else:
        # Utiliser l'URL fournie
        try:
            import requests
            from tqdm import tqdm
            
            print(f"⬇️ Téléchargement du modèle depuis {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f, tqdm(
                desc="Téléchargement du modèle",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
            print("✅ Modèle téléchargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur téléchargement: {e}")
    
    return model_path if Path(model_path).exists() else None
