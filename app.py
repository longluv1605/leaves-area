import os
import streamlit as st
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import uuid
from area import load_model, prepare_image, get_mask, compute_leaves_area

@st.cache_resource
def load_cached_model(model_config: dict) -> torch.nn.Module:
    """Load and cache DeepLabv3+ model."""
    try: 
        return load_model(**model_config)
    except Exception as e:
        print('=========================================')
        print(e)
        raise FileNotFoundError("Cannot load model!")

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path (str): Path to config file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)

def setup_storage(upload_dir: str) -> None:
    """Create upload directory if it does not exist.

    Args:
        upload_dir (str): Path to upload directory.
    """
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """Save uploaded file with a unique filename.

    Args:
        uploaded_file: Streamlit uploaded file object.
        upload_dir (str): Directory to save the file.

    Returns:
        str: Path to saved file.

    Raises:
        ValueError: If file extension is invalid.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Only JPG, JPEG, or PNG files are allowed")
    
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def validate_image(file_path: str, max_size_mb: float, max_dimensions: tuple) -> Image.Image:
    """Validate image file size and dimensions.

    Args:
        file_path (str): Path to image file.
        max_size_mb (float): Maximum file size in MB.
        max_dimensions (tuple): Maximum width and height (width, height).

    Returns:
        PIL.Image.Image: Validated image object.

    Raises:
        ValueError: If file size or dimensions exceed limits, or image is corrupted.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")

    img = Image.open(file_path)
    img.verify()  # Check for corruption
    img = Image.open(file_path)  # Reopen after verify
    
    max_width, max_height = max_dimensions
    if img.size[0] > max_width or img.size[1] > max_height:
        raise ValueError(f"Image dimensions ({img.size[0]}x{img.size[1]}) exceed {max_width}x{max_height}")
    
    return img

def process_image_and_mask(model, image_path: str, device: str, original_size: tuple, coin_diameter_mm: float) -> tuple:
    """Process image to generate mask and calculate leaf area.

    Args:
        model: DeepLabv3+ model instance.
        image_path (str): Path to input image.
        device (str): Computation device ("cuda" or "cpu").
        original_size (tuple): Original image size (width, height).
        coin_diameter_mm (float): Diameter of reference coin in mm.

    Returns:
        tuple: (mask_image, area_value) where mask_image is PIL.Image and area_value is float.

    Raises:
        ValueError: If image processing or area calculation fails.
    """
    image = prepare_image(image_path)
    mask = get_mask(model, image, device)
    
    # Resize mask to original image size
    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize(original_size, resample=Image.NEAREST)
    mask = np.array(mask_resized)
    
    # Calculate leaf area
    area_value = compute_leaves_area(mask, coin_diameter_mm=coin_diameter_mm)
    
    # Apply nipy_spectral colormap
    cmap = plt.get_cmap("nipy_spectral")
    mask_normalized = mask / mask.max()
    mask_rgb = cmap(mask_normalized)[:, :, :3] * 255
    mask_rgb = mask_rgb.astype(np.uint8)
    mask_image = Image.fromarray(mask_rgb, mode="RGB")
    
    return mask_image, area_value

def display_results(area_value: float, original_image: Image.Image, mask_image: Image.Image) -> None:
    """Display leaf area and images in Streamlit.

    Args:
        area_value (float): Leaf area in mm².
        original_image (PIL.Image.Image): Original input image.
        mask_image (PIL.Image.Image): Segmentation mask image.
    """
    st.markdown(f"**Leaf Area: {area_value:.2f} mm²**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Input Image")
        st.image(original_image, caption="Original uploaded image", use_container_width=True)
    with col2:
        st.text("Segmentation Mask")
        st.image(mask_image, caption="Segmented mask with nipy_spectral colormap", use_container_width=True)

def main():
    """Main function to run the Streamlit app."""
    st.title("Leaf Area Calculator")
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    
    model_config = config["model"]
    device_config = config["device"]
    storage_config = config["storage"]
    segmentation_config = config["segmentation"]
    
    # Setup storage
    setup_storage(storage_config["upload_dir"])
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() and device_config == "cuda_if_available" else "cpu"
    
    # Load model
    try:
        model = load_cached_model(model_config)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file
            file_path = save_uploaded_file(uploaded_file, storage_config["upload_dir"])
            
            try:
                # Validate image
                img = validate_image(file_path, storage_config["max_file_size_mb"], (4000, 4000))
                
                # Process image and mask
                mask_image, area_value = process_image_and_mask(
                    model,
                    file_path,
                    device,
                    img.size,
                    segmentation_config["coin_diameter_mm"]
                )
                
                # Save mask
                mask_filename = f"mask_{os.path.basename(file_path)}.png"
                mask_path = os.path.join(storage_config["upload_dir"], mask_filename)
                mask_image.save(mask_path)
                
                # Display results
                display_results(area_value, img, mask_image)
                
            except ValueError as e:
                st.error(str(e))
            
            finally:
                # Clean up (optional, keep for debugging)
                if os.path.exists(file_path):
                    os.remove(file_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()