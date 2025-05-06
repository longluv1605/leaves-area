import os
import shutil
import torch
import yaml
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import area

import matplotlib.pyplot as plt
from matplotlib import cm
import time
import numpy as np

app = FastAPI()

# Load configuration
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract config values
MODEL_CONFIG = config["model"]
SERVER_CONFIG = config["server"]
STORAGE_CONFIG = config["storage"]
SEGMENTATION_CONFIG = config["segmentation"]

# Setup upload directory
UPLOAD_DIR = STORAGE_CONFIG["upload_dir"]
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Determine device
DEVICE = ("cuda" if torch.cuda.is_available()
          else "cpu") if MODEL_CONFIG["device"] == "cuda_if_available" else MODEL_CONFIG["device"]

# Load DeepLabv3+ model
DEEPLABV3PLUS = area.load_model(MODEL_CONFIG["checkpoint_path"])


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the index.html file at the root URL."""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Process uploaded image, generate mask, and calculate area.

    Args:
        file (UploadFile): Image file from client.

    Returns:
        JSONResponse: Mask URL, image URL, and area.

    Raises:
        HTTPException: If file is invalid or processing fails.
    """
    start = time.time()
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are allowed")

    # Check file size (in MB)
    file_size_bytes = 0
    chunk_size = 1024 * 1024  # Read in 1MB chunks
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        file_size_bytes += len(chunk)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    if file_size_mb > STORAGE_CONFIG["max_file_size_mb"]:
        raise HTTPException(status_code=400, detail="File size exceeds limit")
    
    await file.seek(0)

    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save image
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Validate image integrity and dimensions
        img = Image.open(file_path)
        img.verify()  # Check for corruption
        img = Image.open(file_path)  # Reopen after verify
        max_width, max_height = 4000, 4000
        if img.size[0] > max_width or img.size[1] > max_height:
            raise HTTPException(status_code=400, detail="Image dimensions exceed 4000x4000 pixels")
        # Process image
        image = area.prepare_image(file_path)
        mask = area.get_mask(DEEPLABV3PLUS, image, DEVICE)
        cmap = cm.get_cmap("nipy_spectral")

        # Normalize mask for colormap (values 0, 1, 2)
        mask_normalized = mask / 2.0  # Divide by max value (2)
        mask_rgb = cmap(mask_normalized)[:, :, :3]  # Get RGB channels
        mask_rgb = (mask_rgb * 255).astype(np.uint8)
        
        # Calculate area (in mm²)
        area_value = area.compute_leaves_area(
            mask, coin_diameter_mm=SEGMENTATION_CONFIG["coin_diameter_mm"])

        # Save mask
        mask_image = Image.fromarray(mask_rgb, mode="RGB")
        mask_filename = f"mask_{unique_filename}.png"
        mask_path = os.path.join(UPLOAD_DIR, mask_filename)
        mask_image.save(mask_path)
        
        end = time.time()
        print(f"Processing time: {end - start:.2f} seconds.")

        return JSONResponse(content={
            "mask_url": f"/uploads/{mask_filename}",
            "image_url": f"/Uploads/{unique_filename}",  # Add image URL
            "area": round(area_value, 2)  # Round to 2 decimal places
        })
    except Exception as e:
        # Do not remove file_path to keep input image
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/uploads/{filename}")
async def get_file(filename: str):
    """Serve file from uploads directory.

    Args:
        filename (str): Name of the file.

    Returns:
        FileResponse: Requested file.

    Raises:
        HTTPException: If file does not exist.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"]
    )