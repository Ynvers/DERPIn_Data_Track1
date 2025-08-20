"""
Composants pour le dashboard de délimitation de champs.

Ce module contient des utilitaires pour créer un dashboard interactif
permettant aux utilisateurs de sélectionner des zones d'intérêt sur des images
satellite et de détecter les délimitations de champs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import json

# Import des modules locaux (commenter si pas encore créés)
from image_utils import load_geotiff, calculate_global_stretch, normalize_to_uint8, get_tile_from_image, convert_to_bgr
from field_detection import FieldDelineator, download_model


class DashboardManager:
    """
    Gestionnaire de dashboard pour la délimitation de champs.
    Cette classe fournit les fonctions nécessaires pour créer un dashboard
    interactif avec Streamlit, Dash ou toute autre bibliothèque de dashboard.
    """
    
    def __init__(self, model_path=None):
        """
        Initialise le gestionnaire de dashboard.
        
        Args:
            model_path (str): Chemin vers le modèle DelineateAnything 
                             (téléchargé automatiquement si None)
        """
        # Télécharger ou localiser le modèle
        if model_path is None:
            self.model_path = download_model()
        else:
            self.model_path = model_path
            
        # Initialiser le détecteur
        self.delineator = FieldDelineator(self.model_path)
        
        # État interne
        self.current_image_path = None
        self.metadata = None
        self.src = None
        self.stretch_params = None
        self.region_selection = None  # (x, y, width, height)
    
    def load_image(self, image_path):
        """
        Charge une image satellite dans le dashboard.
        
        Args:
            image_path (str): Chemin vers le fichier GeoTIFF
            
        Returns:
            dict: Métadonnées de l'image
        """
        try:
            self.current_image_path = image_path
            self.metadata, self.src = load_geotiff(image_path)
            
            # Calculer les paramètres d'étirement global
            bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
            lo, hi = calculate_global_stretch(self.src, bands)
            self.stretch_params = (lo, hi)
            
            print(f"✅ Image chargée: {Path(image_path).name}")
            print(f"   Dimensions: {self.metadata['width']}x{self.metadata['height']}")
            
            return self.metadata
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            return None
    
    def get_overview_image(self, max_size=1000):
        """
        Génère une image d'aperçu à afficher dans le dashboard.
        
        Args:
            max_size (int): Taille maximale de l'aperçu (pour contrôler la performance)
            
        Returns:
            numpy.ndarray: Image d'aperçu RGB
        """
        if self.src is None:
            print("❌ Aucune image chargée!")
            return None
            
        bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
        
        # Calculer la résolution d'aperçu pour respecter max_size
        scale = max(self.src.width / max_size, self.src.height / max_size, 1)
        out_shape = (len(bands), int(self.src.height / scale), int(self.src.width / scale))
        
        # Lire l'image à résolution réduite
        overview = self.src.read(bands, out_shape=out_shape)
        overview = overview.transpose(1, 2, 0)  # (H, W, C)
        
        # Appliquer l'étirement calculé précédemment
        lo, hi = self.stretch_params
        overview = normalize_to_uint8(overview, lo, hi)
        
        return overview
    
    def set_region_selection(self, x, y, width, height):
        """
        Définit la région sélectionnée par l'utilisateur.
        
        Args:
            x, y (int): Coordonnées du coin supérieur gauche
            width, height (int): Dimensions de la région
            
        Returns:
            tuple: La région sélectionnée (x, y, width, height)
        """
        if self.src is None:
            print("❌ Aucune image chargée!")
            return None
            
        # Valider et ajuster les coordonnées si nécessaire
        x = max(0, min(x, self.src.width - 1))
        y = max(0, min(y, self.src.height - 1))
        width = min(width, self.src.width - x)
        height = min(height, self.src.height - y)
        
        self.region_selection = (x, y, width, height)
        print(f"✅ Région sélectionnée: ({x}, {y}, {width}, {height})")
        
        return self.region_selection
    
    def get_selected_region_preview(self):
        """
        Renvoie un aperçu de la région sélectionnée.
        
        Returns:
            numpy.ndarray: Aperçu de la région sélectionnée
        """
        if self.src is None or self.region_selection is None:
            return None
            
        x, y, width, height = self.region_selection
        bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
        
        # Limiter la taille si nécessaire (pour performance)
        max_preview_size = 1024
        scale = max(width / max_preview_size, height / max_preview_size, 1)
        
        if scale > 1:
            out_shape = (len(bands), int(height / scale), int(width / scale))
            preview = self.src.read(
                bands,
                window=rasterio.windows.Window(x, y, width, height),
                out_shape=out_shape
            )
        else:
            preview = self.src.read(
                bands,
                window=rasterio.windows.Window(x, y, width, height)
            )
            
        preview = preview.transpose(1, 2, 0)
        preview = normalize_to_uint8(preview, *self.stretch_params)
        
        return preview
    
    def process_selected_region(self, output_prefix="output", tile_size=1024, overlap=128):
        """
        Traite la région sélectionnée avec le modèle DelineateAnything.
        
        Args:
            output_prefix (str): Préfixe pour les fichiers de sortie
            tile_size (int): Taille des tuiles
            overlap (int): Chevauchement entre tuiles
            
        Returns:
            dict: Chemins des fichiers générés
        """
        if self.src is None or self.region_selection is None:
            print("❌ Aucune image ou région sélectionnée!")
            return None
        
        # Fermer la source rasterio avant de la réutiliser
        if hasattr(self, 'src') and self.src is not None:
            self.src.close()
            
        result = self.delineator.process_region(
            self.current_image_path,
            region=self.region_selection,
            output_prefix=output_prefix,
            tile_size=tile_size,
            overlap=overlap
        )
        
        # Réouvrir la source après traitement
        self.metadata, self.src = load_geotiff(self.current_image_path)
        
        return result
    
    def process_image(self, image_data, output_prefix="output", tile_size=1024, overlap=128):
        """
        Traite une image chargée en mémoire (pour dashboards web sans accès fichier).
        Cette fonction est utile pour les applications web où les images sont téléchargées.
        
        Args:
            image_data (bytes): Données de l'image en bytes
            output_prefix (str): Préfixe pour les fichiers de sortie
            tile_size (int): Taille des tuiles
            overlap (int): Chevauchement entre tuiles
            
        Returns:
            dict: Résultats de détection
        """
        # Cette fonction serait implémentée en fonction du framework utilisé
        # Elle devrait:
        # 1. Sauvegarder l'image temporairement
        # 2. Appeler process_region sur toute l'image
        # 3. Retourner les résultats
        pass
    
    def image_to_base64(self, img):
        """
        Convertit une image numpy en base64 pour affichage web.
        
        Args:
            img (numpy.ndarray): Image à convertir
            
        Returns:
            str: Représentation base64 de l'image
        """
        buf = io.BytesIO()
        plt.imsave(buf, img, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'src') and self.src is not None:
            self.src.close()
            self.src = None


# Fonctions utiles pour différents frameworks de dashboard

def create_streamlit_app():
    """
    Crée une application Streamlit pour la délimitation de champs.
    Cette fonction est un exemple de la façon d'implémenter un dashboard
    avec Streamlit. Pour utiliser cette fonction, installez streamlit:
    `pip install streamlit`
    """
    try:
        import streamlit as st
        from streamlit_drawable_canvas import st_canvas
        
        st.set_page_config(page_title="Délimitation de Champs", layout="wide")
        st.title("Dashboard de Délimitation de Champs Agricoles")
        
        # Initialisation
        if 'dashboard' not in st.session_state:
            st.session_state.dashboard = DashboardManager()
            
        dashboard = st.session_state.dashboard
        
        # Sidebar pour les options
        st.sidebar.header("Options")
        
        # Upload de fichier
        uploaded_file = st.sidebar.file_uploader("Charger une image satellite (GeoTIFF)", type=["tif", "tiff"])
        
        if uploaded_file:
            # Sauvegarder le fichier temporairement
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            # Charger l'image
            metadata = dashboard.load_image(temp_path)
            
            if metadata:
                st.sidebar.success("✅ Image chargée avec succès!")
                
                # Afficher l'aperçu
                overview = dashboard.get_overview_image()
                if overview is not None:
                    st.image(overview, caption="Aperçu de l'image satellite", use_column_width=True)
                    
                    # Zone de sélection
                    st.header("Sélection de la région d'intérêt")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=dashboard.image_to_base64(overview),
                        height=600,
                        drawing_mode="rect",
                        key="canvas",
                    )
                    
                    # Traiter la sélection
                    if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                        rect = canvas_result.json_data["objects"][0]
                        # Convertir les coordonnées de la canvas en coordonnées d'image
                        scale_x = metadata['width'] / rect["canvasWidth"]
                        scale_y = metadata['height'] / rect["canvasHeight"]
                        x = int(rect["left"] * scale_x)
                        y = int(rect["top"] * scale_y)
                        width = int(rect["width"] * scale_x)
                        height = int(rect["height"] * scale_y)
                        
                        # Définir la sélection
                        dashboard.set_region_selection(x, y, width, height)
                        
                        # Afficher la région sélectionnée
                        selected_preview = dashboard.get_selected_region_preview()
                        if selected_preview is not None:
                            st.image(selected_preview, caption="Région sélectionnée", use_column_width=True)
                            
                            # Bouton pour lancer la détection
                            if st.button("Détecter les champs"):
                                with st.spinner("Détection en cours..."):
                                    result = dashboard.process_selected_region()
                                    
                                    if result:
                                        st.success("✅ Détection terminée!")
                                        
                                        # Afficher le résultat
                                        from PIL import Image
                                        overlay = np.array(Image.open(result["overlay"]))
                                        st.image(overlay, caption="Délimitation des champs", use_column_width=True)
                                        
                                        # Lien pour télécharger le GeoJSON
                                        with open(result["geojson"], "rb") as f:
                                            geojson_bytes = f.read()
                                            st.download_button(
                                                label="Télécharger le GeoJSON",
                                                data=geojson_bytes,
                                                file_name="fields.geojson",
                                                mime="application/json"
                                            )
        
        # Nettoyage lors de la fermeture
        import atexit
        atexit.register(dashboard.cleanup)
        
    except ImportError:
        print("Pour utiliser cette fonction, installez streamlit:")
        print("pip install streamlit streamlit-drawable-canvas")
        
        
def create_dash_app():
    """
    Crée une application Dash pour la délimitation de champs.
    Cette fonction est un exemple de la façon d'implémenter un dashboard
    avec Dash. Pour utiliser cette fonction, installez dash:
    `pip install dash dash-leaflet`
    """
    # Implémentation similaire à streamlit, mais avec Dash
    pass


# Si exécuté directement, lancer l'application Streamlit
if __name__ == "__main__":
    create_streamlit_app()
