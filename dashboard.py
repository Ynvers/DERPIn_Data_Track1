"""
Components for the agricultural field delineation dashboard.

This module contains utilities to create an interactive dashboard
allowing users to select regions of interest on satellite images
and detect field boundaries.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
import json

# Import des modules locaux (commenter si pas encore créés)
import rasterio
from image_utils import load_geotiff, calculate_global_stretch, normalize_to_uint8, get_tile_from_image, convert_to_bgr
from field_detection import FieldDelineator, download_model


class DashboardManager:
    """
    Dashboard manager for field delineation.
    This class provides the necessary functions to create an interactive dashboard
    with Streamlit, Dash or any other dashboard library.
    """
    
    def __init__(self, model_path=None, output_dir="output"):
        """
        Initialize the dashboard manager.
        
        Args:
            model_path (str): Path to the DelineateAnything model 
                             (automatically downloaded if None)
            output_dir (str): Directory for output files and temporary uploads
        """
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Download or locate the model
        if model_path is None:
            self.model_path = download_model()
        else:
            self.model_path = model_path
            
        # Initialize the detector
        self.delineator = FieldDelineator(self.model_path)
        
        # Internal state
        self.current_image_path = None
        self.metadata = None
        self.src = None
        self.stretch_params = None
        self.region_selection = None  # (x, y, width, height)
    
    def load_image(self, image_path):
        """
        Load a satellite image into the dashboard.
        
        Args:
            image_path (str): Path to the GeoTIFF file
            
        Returns:
            dict: Image metadata
        """
        try:
            self.current_image_path = image_path
            self.metadata, self.src = load_geotiff(image_path)
            
            # Calculate global stretch parameters
            bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
            lo, hi = calculate_global_stretch(self.src, bands)
            self.stretch_params = (lo, hi)
            
            print(f"✅ Image loaded: {Path(image_path).name}")
            print(f"   Dimensions: {self.metadata['width']}x{self.metadata['height']}")
            
            return self.metadata
            
        except Exception as e:
            print(f"❌ Loading error: {e}")
            return None
    
    def get_overview_image(self, max_size=1000):
        """
        Generate a preview image to display in the dashboard.
        
        Args:
            max_size (int): Maximum preview size (to control performance)
            
        Returns:
            numpy.ndarray: RGB preview image
        """
        if self.src is None:
            print("❌ No image loaded!")
            return None
            
        bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
        
        # Calculate preview resolution to respect max_size
        scale = max(self.src.width / max_size, self.src.height / max_size, 1)
        out_shape = (len(bands), int(self.src.height / scale), int(self.src.width / scale))
        
        try:
            # Read image at reduced resolution
            overview = self.src.read(bands, out_shape=out_shape)
            overview = overview.transpose(1, 2, 0)  # (H, W, C)
            
            # Apply previously calculated stretch
            lo, hi = self.stretch_params
            overview = normalize_to_uint8(overview, lo, hi)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error reading preview: {e}")
            # Create an empty image in case of error
            overview = np.zeros((500, 500, 3), dtype=np.uint8)
            
        return overview
    
    def set_region_selection(self, x, y, width, height):
        """
        Define the region selected by the user.
        
        Args:
            x, y (int): Upper left corner coordinates
            width, height (int): Region dimensions
            
        Returns:
            tuple: The selected region (x, y, width, height)
        """
        if self.src is None:
            print("❌ No image loaded!")
            return None
            
        # Validate and adjust coordinates if necessary
        x = max(0, min(x, self.src.width - 1))
        y = max(0, min(y, self.src.height - 1))
        width = min(width, self.src.width - x)
        height = min(height, self.src.height - y)
        
        self.region_selection = (x, y, width, height)
        print(f"✅ Region selected: ({x}, {y}, {width}, {height})")
        
        return self.region_selection
    
    def get_selected_region_preview(self):
        """
        Return a preview of the selected region.
        
        Returns:
            numpy.ndarray: Preview of the selected region
        """
        import rasterio.windows
        
        if self.src is None or self.region_selection is None:
            return None
            
        x, y, width, height = self.region_selection
        bands = [1, 2, 3] if self.src.count == 3 else [4, 3, 2]
        
        # Limit size if necessary (for performance)
        max_preview_size = 1024
        scale = max(width / max_preview_size, height / max_preview_size, 1)
        
        try:
            # Create a rasterio window
            window = rasterio.windows.Window(x, y, width, height)
            
            if scale > 1:
                out_shape = (len(bands), int(height / scale), int(width / scale))
                preview = self.src.read(
                    bands,
                    window=window,
                    out_shape=out_shape
                )
            else:
                preview = self.src.read(
                    bands,
                    window=window
                )
                
            preview = preview.transpose(1, 2, 0)
            preview = normalize_to_uint8(preview, *self.stretch_params)
            
            return preview
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error reading selected region: {e}")
            return None
    
    def process_selected_region(self, output_prefix="output", tile_size=1024, overlap=128):
        """
        Process the selected region with the DelineateAnything model.
        
        Args:
            output_prefix (str): Prefix for output files (will be placed in output_dir)
            tile_size (int): Tile size
            overlap (int): Overlap between tiles
            
        Returns:
            dict: Paths to generated files
        """
        if self.src is None or self.region_selection is None:
            print("❌ No image or region selected!")
            return None
        
        # Create full output path in the output directory
        full_output_prefix = os.path.join(self.output_dir, output_prefix)
        
        # Close rasterio source before reusing it
        if hasattr(self, 'src') and self.src is not None:
            self.src.close()
            
        result = self.delineator.process_region(
            self.current_image_path,
            region=self.region_selection,
            output_prefix=full_output_prefix,
            tile_size=tile_size,
            overlap=overlap
        )
        
        # Reopen source after processing
        self.metadata, self.src = load_geotiff(self.current_image_path)
        
        return result
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        if hasattr(self, 'src') and self.src is not None:
            self.src.close()
            self.src = None
            
        # Clean up temporary files in output directory
        try:
            import glob
            temp_files = glob.glob(os.path.join(self.output_dir, "temp_*"))
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                    print(f"🗑️ Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {temp_file}: {e}")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


def create_streamlit_app():
    """
    Create a Streamlit application for field delineation.
    """
    try:
        import streamlit as st
        import os  # Add os import for local use
        
        st.set_page_config(page_title="Field Delineation", layout="wide")
        st.title("🛰️ Agricultural Field Delineation Dashboard")
        
        # State initialization
        if 'dashboard' not in st.session_state:
            st.session_state.dashboard = DashboardManager()
        if 'image_loaded' not in st.session_state:
            st.session_state.image_loaded = False
        if 'region_selected' not in st.session_state:
            st.session_state.region_selected = False
        if 'processing_done' not in st.session_state:
            st.session_state.processing_done = False
            
        dashboard = st.session_state.dashboard
        
        # Sidebar for options
        st.sidebar.header("📁 File Options")
        
        # Button to reset
        if st.sidebar.button("🔄 New Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # File upload 
        uploaded_file = st.sidebar.file_uploader(
            "Load satellite image (GeoTIFF)", 
            type=["tif", "tiff"],
            accept_multiple_files=False,
            help="Size limit: 500MB (configured in .streamlit/config.toml)"
        )
        
        # Load image only if a new file is uploaded
        if uploaded_file and not st.session_state.image_loaded:
            # Save file temporarily in output directory
            temp_path = os.path.join(dashboard.output_dir, f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            # Load image
            with st.spinner("🔄 Loading image..."):
                metadata = dashboard.load_image(temp_path)
                
            if metadata:
                st.session_state.image_loaded = True
                st.session_state.region_selected = False
                st.session_state.processing_done = False
                st.sidebar.success("✅ Image loaded successfully!")
                st.sidebar.write(f"**Dimensions:** {metadata['width']}x{metadata['height']}")
                st.sidebar.write(f"**CRS:** {metadata['crs']}")
            else:
                st.sidebar.error("❌ Error loading image")
        
        # Display main interface only if image is loaded
        if st.session_state.image_loaded:
            # Display image preview (once only)
            overview = dashboard.get_overview_image()
            if overview is not None:
                st.subheader("🖼️ Satellite Image Preview")
                st.image(overview, caption="Loaded satellite image", use_container_width=True)
                
                # Region selection section
                st.header("🎯 Region of Interest Selection")
                
                # Use sliders to select a region
                col1, col2 = st.columns(2)
                
                with col1:
                    x_start = st.slider("X Position (%)", 0, 100, 25, 1, key="x_slider")
                    width_percent = st.slider("Width (%)", 1, 100, 50, 1, key="width_slider")
                    
                with col2:
                    y_start = st.slider("Y Position (%)", 0, 100, 25, 1, key="y_slider")
                    height_percent = st.slider("Height (%)", 1, 100, 50, 1, key="height_slider")
                
                # Convert percentages to image coordinates
                metadata = dashboard.metadata
                x = int(metadata['width'] * x_start / 100)
                y = int(metadata['height'] * y_start / 100)
                width = int(metadata['width'] * width_percent / 100)
                height = int(metadata['height'] * height_percent / 100)
                
                # Display calculated coordinates
                st.info(f"📍 Selected region: X={x}, Y={y}, Width={width}, Height={height}")
                
                # Button to validate selection
                if st.button("🎯 Validate Region Selection", key="validate_selection"):
                    # Define selection in dashboard manager
                    dashboard.set_region_selection(x, y, width, height)
                    st.session_state.region_selected = True
                    st.session_state.processing_done = False
                    st.success("✅ Region selected successfully!")
                    st.rerun()  # Force refresh
                
                # Display region preview only if selected
                if st.session_state.region_selected:
                    st.subheader("🔍 Selected Region Preview")
                    try:
                        selected_preview = dashboard.get_selected_region_preview()
                        if selected_preview is not None:
                            from PIL import Image
                            if not isinstance(selected_preview, Image.Image):
                                selected_preview = Image.fromarray(selected_preview)
                            st.image(selected_preview, caption="Region to be processed", use_container_width=True)
                            
                            # Button to launch detection
                            if st.button("🚀 Detect Fields in This Region", key="detect_fields"):
                                with st.spinner("🧠 Detection in progress... This may take several minutes."):
                                    st.info("📍 The DelineateAnything model processes the region by tiles...")
                                    result = dashboard.process_selected_region(output_prefix="dashboard_output")
                                    
                                    if result:
                                        st.session_state.processing_done = True
                                        st.session_state.result_files = result
                                        st.success("✅ Detection completed!")
                                        st.balloons()  # Success animation
                                        st.rerun()  # Force refresh
                                    else:
                                        st.error("❌ Error during processing.")
                        else:
                            st.warning("⚠️ Unable to get preview of selected region.")
                    except Exception as e:
                        st.error(f"❌ Error displaying region: {e}")
                
                # Display results if available
                if st.session_state.processing_done and 'result_files' in st.session_state:
                    st.header("🎉 Detection Results")
                    result = st.session_state.result_files
                    
                    # Display image with delineations
                    try:
                        from PIL import Image
                        import os
                        if os.path.exists(result["overlay"]):
                            overlay = np.array(Image.open(result["overlay"]))
                            st.image(overlay, caption="🌾 Detected field boundaries", use_container_width=True)
                            
                            # Button to download GeoJSON
                            if os.path.exists(result["geojson"]):
                                with open(result["geojson"], "rb") as f:
                                    geojson_bytes = f.read()
                                    st.download_button(
                                        label="📄 Download GeoJSON File",
                                        data=geojson_bytes,
                                        file_name="fields_detection.geojson",
                                        mime="application/json"
                                    )
                        else:
                            st.error("❌ Result files not found")
                    except Exception as e:
                        st.error(f"❌ Error displaying results: {e}")
        else:
            st.info("👈 Please load a satellite image to get started")
            st.markdown("""
            ### Usage Instructions:
            
            1. **Load a satellite image** (.tif or .tiff file) via the sidebar
            2. **Select a region of interest** using position and size sliders
            3. **Validate your selection** and preview the region
            4. **Launch field detection** with the DelineateAnything model
            5. **Download results** in GeoJSON format
            """)
        
        # Cleanup on exit
        import atexit
        atexit.register(dashboard.cleanup)
        
    except ImportError:
        print("To use this function, install streamlit:")
        print("pip install streamlit")


# If executed directly, launch the Streamlit application
if __name__ == "__main__":
    create_streamlit_app()
