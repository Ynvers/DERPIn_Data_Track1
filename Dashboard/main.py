import os
import sys
import streamlit as st
import rasterio
import numpy as np
from io import BytesIO
from PIL import Image
from streamlit_cropper import st_cropper
import tempfile
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from field_detection import FieldDelineator

def save_cropped_to_geotiff(cropped_img, original_profile, original_transform, crop_box):
    """
    Sauvegarde la région croppée en tant que fichier GeoTIFF temporaire.
    
    Args:
        cropped_img (PIL.Image): L'image croppée
        original_profile: Profil rasterio de l'image originale
        original_transform: Transform original de l'image
        crop_box: Coordonnées du crop {'left': x, 'top': y, 'width': w, 'height': h}
        
    Returns:
        str: Chemin vers le fichier temporaire GeoTIFF
    """
    # Convertir PIL en numpy array
    cropped_array = np.array(cropped_img)
    
    # Calculer la nouvelle transformation pour la zone croppée
    # Décalage de l'origine selon le crop
    left, top = crop_box['left'], crop_box['top']
    new_transform = rasterio.transform.Affine(
        original_transform.a, original_transform.b, 
        original_transform.c + original_transform.a * left + original_transform.b * top,
        original_transform.d, original_transform.e,
        original_transform.f + original_transform.d * left + original_transform.e * top
    )
    
    # Créer un profil pour la zone croppée
    crop_profile = original_profile.copy()
    crop_profile.update({
        'width': cropped_array.shape[1],
        'height': cropped_array.shape[0],
        'count': 3,
        'dtype': 'uint8',
        'transform': new_transform
    })
    
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmpfile:
        tmpfile_path = tmpfile.name
        
        with rasterio.open(tmpfile_path, 'w', **crop_profile) as dst:
            # Réorganiser (H,W,C) -> (C,H,W) pour rasterio
            dst.write(np.moveaxis(cropped_array, -1, 0))
        
        return tmpfile_path

@st.cache_data
def load_geotiff(file_bytes):
    with rasterio.open(BytesIO(file_bytes)) as src:
        profile = src.profile.copy()
        image = src.read()
        transform = src.transform
        crs = src.crs
        
        # Prendre les bandes appropriées selon le nombre disponible
        if image.shape[0] >= 4:  # Sentinel-2 avec bandes R,G,B aux positions 4,3,2
            rgb = np.stack([image[3], image[2], image[1]], axis=-1)  # R,G,B
        elif image.shape[0] >= 3:
            rgb = np.stack([image[0], image[1], image[2]], axis=-1)
        else:
            # Si moins de 3 bandes, dupliquer la première
            rgb = np.stack([image[0], image[0], image[0]], axis=-1)
        
        # Normalisation pour affichage (même logique que votre process_region)
        rgb = rgb.astype(np.float32)
        lo = float(np.percentile(rgb, 0.5))
        hi = float(np.percentile(rgb, 99.5))
        rgb = np.clip((rgb - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
        rgb = rgb.round().astype(np.uint8)
        
        pil_image = Image.fromarray(rgb)
        
    return pil_image, profile, transform, crs

def main():
    st.set_page_config(
        page_title="Field Delimitation for Climate-Smart Agriculture",
        layout="wide"
    )

    st.title("🌾 Field Delimitation for Climate-Smart Agriculture")
    st.markdown("""
    This application assists farmers in identifying and delineating their fields using 
    satellite imagery and machine learning techniques.

    ## Features
    - **🛰️ Field Visualization**: View satellite imagery of your fields
    - **🎯 Field Delimitation**: Automatically detect and delineate field boundaries
    - **⚡ Smart Processing**: Automatic tile management for optimal performance
    """)

    # Initialisation du modèle
    if 'delineator' not in st.session_state:
        with st.spinner("Loading DelineateAnything model..."):
            try:
                delineator = FieldDelineator(model_path="../DelineateAnything.pt")
                if delineator.model is not None:
                    st.session_state.delineator = delineator
                    st.success("🎉 Model loaded successfully!")
                else:
                    st.error("❌ Failed to load the model.")
                    return
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                return
    else:
        st.success("✅ Model is ready.")

    # Sidebar
    st.sidebar.title("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload GeoTIFF file", 
        type=['tiff', 'tif'],
        help="Upload satellite imagery in GeoTIFF format (Sentinel-2, Landsat, etc.)"
    )

    if uploaded_file:
        st.sidebar.success("📂 File uploaded successfully!")
        
        # Charger l'image
        try:
            pil_image, profile, transform, crs = load_geotiff(uploaded_file.getvalue())
        except Exception as e:
            st.error(f"❌ Error loading image: {e}")
            return

        # Afficher les informations de l'image
        with st.expander("📊 Image Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Dimensions:** {profile['width']} × {profile['height']} pixels")
                st.write(f"**Bands:** {profile['count']}")
                st.write(f"**Data Type:** {profile['dtype']}")
            with col2:
                st.write(f"**CRS:** {crs}")
                st.write(f"**Driver:** {profile['driver']}")
                if 'transform' in profile:
                    pixel_size = abs(profile['transform'].a)
                    st.write(f"**Resolution:** ~{pixel_size:.1f}m/pixel")

        # Interface de sélection de région
        st.subheader("🎯 Select Analysis Region")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Cropper pour sélectionner la région
            # Par défaut, st_cropper retourne l'image croppée directement
            try:
                # Essayer d'abord avec return_type (nouvelles versions)
                cropped_result = st_cropper(
                    pil_image,
                    realtime_update=True,
                    box_color="#FF6B6B",
                    aspect_ratio=None,
                    return_type="dict"
                )
                if isinstance(cropped_result, dict):
                    cropped_img = cropped_result.get('image', cropped_result.get('img', pil_image))
                    crop_box = cropped_result.get('box', cropped_result.get('bbox', None))
                else:
                    cropped_img = cropped_result
                    crop_box = None
            except:
                # Fallback pour les anciennes versions
                cropped_img = st_cropper(
                    pil_image,
                    realtime_update=True,
                    box_color="#FF6B6B",
                    aspect_ratio=None
                )
                crop_box = None
            
            # Si on n'a pas les coordonnées exactes, les estimer
            if crop_box is None:
                original_width, original_height = pil_image.size
                cropped_width, cropped_height = cropped_img.size
                
                crop_box = {
                    'left': (original_width - cropped_width) // 2,
                    'top': (original_height - cropped_height) // 2,
                    'width': cropped_width,
                    'height': cropped_height
                }
        
        with col2:
            st.write("**Selected Region Info:**")
            if crop_box:
                region_width = crop_box['width']
                region_height = crop_box['height']
                total_pixels = region_width * region_height
                
                st.metric("Width", f"{region_width} px")
                st.metric("Height", f"{region_height} px")
                st.metric("Total Pixels", f"{total_pixels:,}")
                
                # Recommandations de traitement basées sur votre logique
                if total_pixels > 1024 * 1024:  # > 1M pixels
                    st.warning("⚠️ Large region - will be processed in tiles")
                    tile_estimate = (region_width // 1024 + 1) * (region_height // 1024 + 1)
                    st.write(f"Estimated tiles: ~{tile_estimate}")
                else:
                    st.info("✅ Region will be processed efficiently")

        # Paramètres de traitement
        with st.expander("⚙️ Processing Parameters"):
            col1, col2 = st.columns(2)
            with col1:
                conf_threshold = st.slider(
                    "Confidence Threshold", 0.1, 0.9, 0.35, 0.05,
                    help="Lower = more detections, Higher = more precise detections"
                )
                iou_threshold = st.slider(
                    "IoU Threshold", 0.1, 0.9, 0.5, 0.05,
                    help="Threshold for removing overlapping detections"
                )
            with col2:
                # Note: tile_size sera géré automatiquement par votre process_region
                st.info("**Processing Info:**\n"
                       "- Tile size: Automatically optimized\n"
                       "- Overlap: Smart overlap handling\n"
                       "- Output: overlay.png & fields.geojson")

        # Bouton d'analyse
        if st.button("🚀 Analyze Selected Region", type="primary", use_container_width=True):
            if not crop_box:
                st.error("Please select a region first!")
                return
                
            # Estimation du temps de traitement
            estimated_time = max(5, (region_width * region_height) // 100000)
            
            with st.spinner(f"🔍 Analyzing region... (estimated {estimated_time}s)"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Étape 1: Préparer les données
                    status_text.text("📝 Preparing cropped region...")
                    progress_bar.progress(20)
                    
                    temp_tif_path = save_cropped_to_geotiff(
                        cropped_img, profile, transform, crop_box
                    )
                    
                    # Étape 2: Traitement avec FieldDelineator
                    status_text.text("🤖 Running AI detection...")
                    progress_bar.progress(40)
                    
                    delineator = st.session_state.delineator
                    
                    # Votre process_region gère automatiquement les paramètres optimaux
                    # Il sauvegarde directement overlay.png et fields.geojson
                    results = delineator.process_region(
                        tif_path=temp_tif_path,
                        region=None,  # Traite toute l'image croppée
                        conf_thr=conf_threshold,
                        iou_thr=iou_threshold
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("📊 Processing results...")
                    
                    # Vérifier que les fichiers ont été créés
                    overlay_exists = os.path.exists("overlay.png")
                    geojson_exists = os.path.exists("fields.geojson")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed!")
                    
                    if overlay_exists and geojson_exists:
                        st.success("🎉 Field detection completed successfully!")
                        
                        # Affichage des résultats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🖼️ Detected Field Boundaries")
                            overlay_img = Image.open("overlay.png")
                            st.image(overlay_img, caption="White lines show detected field boundaries", use_container_width=True)
                            
                        with col2:
                            st.subheader("📋 Detection Statistics")
                            
                            # Lire le GeoJSON pour les statistiques
                            with open("fields.geojson", 'r', encoding='utf-8') as f:
                                geojson_data = json.load(f)
                            
                            features = geojson_data.get('features', [])
                            num_fields = len(features)
                            
                            # Afficher les métriques principales
                            st.metric("🏞️ Fields Detected", num_fields)
                            
                            if num_fields > 0:
                                # Calcul de statistiques sur la complexité des polygones
                                complexities = []
                                for feature in features:
                                    if 'geometry' in feature and 'coordinates' in feature['geometry']:
                                        coords = feature['geometry']['coordinates'][0]
                                        complexities.append(len(coords))
                                
                                if complexities:
                                    st.metric("📐 Avg. Boundary Points", f"{np.mean(complexities):.1f}")
                                    st.metric("🔢 Most Complex Field", f"{max(complexities)} points")
                                    st.metric("🎯 CRS", str(geojson_data.get('crs', {}).get('properties', {}).get('name', 'Unknown')))
                                
                            else:
                                st.info("No fields detected. Try adjusting the confidence threshold.")
                        
                        # Section de téléchargement
                        st.subheader("📥 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Bouton de téléchargement pour l'image
                            with open("overlay.png", "rb") as f:
                                st.download_button(
                                    "🖼️ Download Overlay Image",
                                    f.read(),
                                    file_name="field_boundaries_overlay.png",
                                    mime="image/png"
                                )
                        
                        with col2:
                            # Bouton de téléchargement pour le GeoJSON
                            with open("fields.geojson", 'r', encoding='utf-8') as f:
                                st.download_button(
                                    "📍 Download GeoJSON",
                                    f.read(),
                                    file_name="field_boundaries.geojson",
                                    mime="application/json"
                                )
                    else:
                        st.error("❌ Analysis failed. Output files were not created.")
                        st.info("Try adjusting the confidence threshold or selecting a different region.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
                
                finally:
                    # Nettoyer le fichier temporaire
                    try:
                        if 'temp_tif_path' in locals() and os.path.exists(temp_tif_path):
                            os.unlink(temp_tif_path)
                    except:
                        pass
                    
                    # Nettoyer la barre de progression
                    progress_bar.empty()
                    status_text.empty()

    else:
        # Instructions d'utilisation quand aucun fichier n'est uploadé
        st.info("👆 Please upload a GeoTIFF file to start field detection.")
        
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            1. **Upload** a GeoTIFF satellite image (Sentinel-2, Landsat, etc.)
            2. **Select** the region you want to analyze by drawing a rectangle
            3. **Adjust** detection parameters if needed (optional)
            4. **Click** 'Analyze Selected Region' to run the AI detection
            5. **Download** the results (overlay image + GeoJSON boundaries)
            
            **Supported formats:** .tif, .tiff files with geographic coordinates
            **Best results:** Agricultural areas with visible field boundaries
            """)

if __name__ == "__main__":
    main()