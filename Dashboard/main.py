import streamlit as st
import rasterio
import numpy as np
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
        temp_tif = "temp.tif"
        with open(temp_tif, "wb") as f:
            f.write(uploaded_file.getvalue())

        with rasterio.open(temp_tif) as src:
            image = src.read()
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
                st.spinner("Running model...")
                # Convert cropped image to numpy array
                cropped_array = np.array(cropped_img)
                results = "hello"
                for result in results:
                    # Convert result to PIL Image
                    result_img = Image.fromarray(result.plot())
                    st.image(result_img, caption="Delineated Fields", use_column_width=True)
if __name__ == "__main__":
    main()