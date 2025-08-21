import streamlit as st
import rasterio
import numpy as np
from predictions import predict_zones
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from streamlit_cropper import st_cropper
# import folium
# from streamlit_folium import folium_static
# from data_loder import DataLoader


def load_model():
    """Load the pre-trained model for field delineation."""
    model = YOLO("DelineateAnything.pt")
    return model

def main():
    st.set_page_config(
        page_title="Field Delimitation for Climate-Smart Argriculture",
        layout="wide"
    )

    # Title and description 
    st.title("Field Delimitation for Climate-Smart Agriculture")
    st.markdown("""
    This application is designed to assist farmers in identifying and delineating their fields using satellite imagery and machine learning techniques. 
    The goal is to provide a user-friendly interface for farmers to visualize their fields, understand their boundaries, and make informed decisions about crop management.

    ## Features
    - **Field Visualization**: View satellite imagery of your fields.
    - **Field Delimitation**: Automatically detect and delineate field boundaries.
    - **Data Analysis**: Analyze field characteristics and crop health.
    """)

    # if 'model' not in st.session_state:
    #     with st.spinner("Loading model..."):
    #         st.session_state.model = load_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    # File uploader for GeoTIFF
    uploaded_file = st.sidebar.file_uploader("Upload GeoTIFF file (up to 2GB)", type=['tiff', 'tif'])

    if uploaded_file :
        st.sidebar.success("File uploaded successfully!")
        
        file_bytes = BytesIO(uploaded_file.getvalue())
        with rasterio.open(file_bytes) as src:
            # Garder les métadonnées importantes
            profile = src.profile.copy()
            image = src.read()
            transform = src.transform
            crs = src.crs
            # Convertir en RGB pour affichage
            rgb = np.stack([image[0], image[1], image[2]], axis=-1)
            rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
            
            # Convertir en PIL Image pour le cropper
            pil_image = Image.fromarray(rgb)

            # Interface de sélection
            st.subheader("Select area to analyze")
            cropped_img = st_cropper(
                pil_image,
                realtime_update=True,
                box_color="red",
                aspect_ratio=None
            )

            #Button to run the model
            if st.button("Analyse Selected Area"):
                with st.spinner("Running model..."):
                    # Convertir l'image croppée en array numpy
                    cropped_array = np.array(cropped_img)
                    
                    # Créer un nouveau raster TIFF en mémoire avec les métadonnées appropriées
                    mem_file = BytesIO()
                    
                    # Mettre à jour le profile pour la nouvelle taille
                    crop_profile = profile.copy()
                    crop_profile.update({
                        'width': cropped_array.shape[1],
                        'height': cropped_array.shape[0],
                        'count': 3  # nombre de bandes
                    })
                    
                    # Créer un nouveau dataset rasterio en mémoire
                    with rasterio.io.MemoryFile() as memfile:
                        with memfile.open(**crop_profile) as dst:
                            # Réorganiser les bandes (H,W,C) -> (C,H,W)
                            cropped_raster = np.moveaxis(cropped_array, -1, 0)
                            dst.write(cropped_raster)
                            
                            # Lire les données avant de les passer à predict_zones
                            input_data = dst.read()
                            result = predict_zones(input_data)  # Passer les données numpy
                    
                    # Gestion du résultat
                    if isinstance(result, np.ndarray):
                        if result.ndim == 3:
                            if result.shape[0] == 3:
                                result = np.moveaxis(result, 0, -1)  # (C,H,W) -> (H,W,C)
                            elif result.shape[-1] == 3:
                                pass  # Déjà au bon format (H,W,C)
                            else:
                                st.error(f"Unexpected number of channels: {result.shape}")
                                return
                        else:
                            st.error(f"Unexpected dimensions: {result.ndim}")
                            return
                    else:
                        st.error(f"Unexpected result type: {type(result)}")
                        return
                    
                    st.image(result, caption="Delineated Fields", use_container_width=True)


if __name__ == "__main__":
    main()