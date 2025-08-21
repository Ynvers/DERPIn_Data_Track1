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
import requests
from tqdm import tqdm

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
            if not os.path.exists(self.model_path):
                print(f"❌ Modèle non trouvé, téléchargement en cours...")
                self.model_path = download_model()  # Télécharger le modèle -_-

            if self.model_path is not None:
                print("🛰️ Chargement du modèle Delineate-Anything...")
                self.model = YOLO(self.model_path)
                print("✅ Modèle chargé avec succès!")
                return True
            else:
                print("❌ Échec du téléchargement du modèle.")
                return False
            
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
        
        # --- lecture du GeoTIFF en RGB uint8 pour servir de fond ---
        with rasterio.open(tif_path) as src:
            # rgb = src.read([1,2,3]).transpose(1,2,0)  # (H,W,3)
            # if rgb.dtype == np.uint16:
            #     rgb = (rgb.astype(np.float32)/65535.0*255).astype(np.uint8)
            # elif rgb.max() > 255:
            #     rgb = (rgb.astype(np.float32)/rgb.max()*255).astype(np.uint8)
            # calcule un étirement global (évite contrastes différents par tuile)
            bands = [1,2,3] if src.count == 3 else [4,3,2]  # S2: [4,3,2] = R,G,B
            crs = src.crs
            transform = src.transform
            H, W = src.height, src.width
            preview = src.read(bands, out_shape=(len(bands), H//8, W//8)).transpose(1,2,0).astype(np.float32)
            lo = float(np.percentile(preview, 0.5))
            hi = float(np.percentile(preview, 99.5))

            def to_uint8_global(img):
                x = img.astype(np.float32)
                x = np.clip((x - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
                return x.round().astype(np.uint8)

            # fond overlay en 8-bit (même étirement partout)
            rgb = to_uint8_global(src.read(bands).transpose(1,2,0))

        overlay = rgb.copy()
        features = []  # pour GeoJSON
        boundary_layer = np.zeros_like(overlay, dtype=np.uint8)

        def draw_outline(poly_pts, layer, color_fg=(255,255,255), color_bg=(0,0,0)):
            pts = poly_pts.astype(np.int32).reshape(-1,1,2)
            cv2.polylines(layer, [pts], True, color_bg, 6, lineType=cv2.LINE_AA)  # halo noir
            cv2.polylines(layer, [pts], True, color_fg, 3, lineType=cv2.LINE_AA)  # trait blanc


        print("🚀 Lancement de la détection (tuile par tuile)")
        with rasterio.open(tif_path) as src:
            for y in range(0, src.height, tile_size):
                for x in range(0, src.width, tile_size):
                    win = rasterio.windows.Window(
                        x, y,
                        min(tile_size, src.width  - x),
                        min(tile_size, src.height - y)
                    )
                    # image de la tuile
                    tile = src.read([1,2,3], window=win).transpose(1,2,0)
                    tile = to_uint8_global(tile)                    # même étirement global
                    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)  # Ultralytics attend BGR si ndarray
                    # if tile.dtype == np.uint16:
                    #     tile = (tile.astype(np.float32)/65535.0*255).astype(np.uint8)
                    # elif tile.max() > 255:
                    #     tile = (tile.astype(np.float32)/tile.max()*255).astype(np.uint8)
                    try:
                        preds = self.model.predict(
                            source=tile,
                            conf=conf_thr,
                            iou=iou_thr,
                            imgsz=max(960, max(win.width, win.height)),  # ↑ détail
                            retina_masks=True,                           # masques haute résolution
                            agnostic_nms=True,
                            verbose=False
                        )
                    except Exception:
                        # fallback si OOM
                        preds = self.model.predict(
                            source=tile, conf=conf_thr, iou=iou_thr,
                            imgsz=640, retina_masks=True, agnostic_nms=True, verbose=False
                        )

                    # try:
                    #     preds = model.predict(source=tile, conf=conf_thr, iou=iou_thr, verbose=False)
                    # except Exception as e:
                    #     print(f"❌ tuile ({x},{y}) — erreur: {e}")
                    #     continue

                    # pour chaque résultat de la tuile
                    for r in preds:
                        if getattr(r, "masks", None) is None:
                            continue

                        # cas 1: Ultralytics fournit les polygones directement
                        if getattr(r.masks, "xy", None):
                            for poly in r.masks.xy:
                                if poly is None or len(poly) < 3:
                                    continue
                                poly = np.array(poly, dtype=np.float32)
                                # offset vers l'image globale
                                poly[:, 0] += x
                                poly[:, 1] += y
                                # dessiner sur l'overlay (blanc, épaisseur 2)

                                cv2.polylines(overlay, [poly.astype(np.int32)], True, (255,255,255), 2)

                                # ajouter au GeoJSON (coords en CRS de l'image)
                                # transform: x_geo = a*col + b*row + c ; y_geo = d*col + e*row + f
                                # ici col = x, row = y
                                pts_geo = []
                                a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
                                for cx, cy in poly:
                                    X = c + a*cx + b*cy
                                    Y = f + d*cx + e*cy
                                    pts_geo.append((X, Y))
                                if len(pts_geo) >= 3:
                                    P = Polygon(pts_geo)
                                    if P.is_valid and not P.is_empty and P.area > 0:
                                        features.append({
                                            "type":"Feature",
                                            "properties":{},
                                            "geometry": mapping(P)
                                        })

                        # cas 2 (rare): si pas de .xy, on peut extraire les contours depuis le masque binaire
                        elif getattr(r.masks, "data", None) is not None:
                            # r.masks.data: (N, h, w) déjà remis à l’échelle de la tuile
                            mdata = r.masks.data.cpu().numpy()
                            for m in mdata:
                                m = (m > 0.5).astype(np.uint8)*255
                                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in cnts:
                                    if len(cnt) < 3:
                                        continue
                                    cnt = cnt.reshape(-1,2).astype(np.float32)
                                    cnt[:,0] += x
                                    cnt[:,1] += y
                                    cv2.polylines(overlay, [cnt.astype(np.int32)], True, (255,255,255), 2)

                                    pts_geo = []
                                    a,b,c,d,e,f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
                                    for cx, cy in cnt:
                                        X = c + a*cx + b*cy
                                        Y = f + d*cx + e*cy
                                        pts_geo.append((X, Y))
                                    if len(pts_geo) >= 3:
                                        P = Polygon(pts_geo)
                                        if P.is_valid and not P.is_empty and P.area > 0:
                                            features.append({
                                                "type":"Feature",
                                                "properties":{},
                                                "geometry": mapping(P)
                                            })

                    print(f"✅ tuile ({x},{y}) traitée")

            # sauvegardes
            cv2.imwrite("overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            geojson = {"type":"FeatureCollection","features":features,
                    "crs":{"type":"name","properties":{"name":str(crs)}}}
            with open("fields.geojson","w",encoding="utf-8") as f:
                json.dump(geojson, f)

            print("🎉 Fini !")
            print(" - overlay.png : image avec les délimitations")
            print(" - fields.geojson : polygones pour le dashboard")


def download_model(url=None):
    """
    Télécharge le modèle DelineateAnything si nécessaire.

    Args:
        url (str): URL optionnelle du modèle (sinon utilise HuggingFace)

    Returns:
        str: Chemin vers le fichier du modèle
    """
    model_path = "DelineateAnything.pt"

    # Vérifiez si le modèle existe déjà localement
    if Path(model_path).exists():
        print(f"✅ Le modèle {model_path} est déjà présent.")
        return model_path

    # Utiliser l'URL par défaut si aucune URL n'est fournie
    if url is None:
        url = "https://huggingface.co/MykolaL/DelineateAnything/resolve/main/DelineateAnything.pt"
        print(f"❌ URL non fournie, utilisation de l'URL par défaut : {url}")

    # Téléchargement du modèle
    try:
        print(f"⬇️ Téléchargement du modèle depuis {url}...")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Vérifie si la requête a échoué (par exemple, 404)

        # Vérifiez la taille du fichier pour afficher la progression
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

        print(f"✅ Modèle téléchargé avec succès! Le modèle est situé à : {model_path}")
        return model_path

    except requests.exceptions.RequestException as e:
        # Gérer les erreurs de téléchargement HTTP
        print(f"❌ Erreur lors du téléchargement du modèle : {e}")
        print(f"📝 Essayez de télécharger manuellement le modèle depuis : {url}")
        return None