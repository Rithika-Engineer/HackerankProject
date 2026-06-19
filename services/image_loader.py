"""
Image loading utilities.
"""

import base64
import logging
from pathlib import Path
from PIL import Image
import io

logger = logging.getLogger(__name__)

def load_image(image_path: str, base_dir: str) -> dict:
    """Load a single image and convert to base64."""
    full_path = Path(base_dir) / image_path
    image_id = Path(image_path).stem
    
    if not full_path.exists():
        logger.error(f"Image not found: {full_path}")
        return {
            "image_id": image_id,
            "path": image_path,
            "base64_data": "",
            "mime_type": "image/jpeg",
            "valid": False
        }
    
    try:
        with Image.open(full_path) as img:
            # Determine mime type
            format_ext = img.format.lower() if img.format else "jpeg"
            mime_type = f"image/{format_ext}"
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format=img.format if img.format else "JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "image_id": image_id,
                "path": image_path,
                "base64_data": img_str,
                "mime_type": mime_type,
                "valid": True
            }
    except Exception as e:
        logger.error(f"Failed to load image {full_path}: {e}")
        return {
            "image_id": image_id,
            "path": image_path,
            "base64_data": "",
            "mime_type": "image/jpeg",
            "valid": False
        }

def load_images_for_claim(image_paths_str: str, base_dir: str) -> list[dict]:
    """Load all images for a claim."""
    if pd.isna(image_paths_str) or not image_paths_str:
        return []
        
    paths = [p.strip() for p in image_paths_str.split(";")]
    return [load_image(p, base_dir) for p in paths if p]

import pandas as pd
