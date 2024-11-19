import cv2
import numpy as np
from scipy import ndimage

def enhance_fingerprint(image):
    """
    Enhanced preprocessing pipeline for fingerprint images.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize intensity
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(normalized)
    
    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
    
    # Local adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Thinning/skeletonization
    skeleton = cv2.ximgproc.thinning(cleaned)
    
    return skeleton

def get_ridge_orientation(image, block_size=16, gaussian_kernel=(5,5)):
    """
    Calculate ridge orientation field.
    """
    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient components
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy
    
    # Apply Gaussian smoothing
    gxx = cv2.GaussianBlur(gxx, gaussian_kernel, 0)
    gyy = cv2.GaussianBlur(gyy, gaussian_kernel, 0)
    gxy = cv2.GaussianBlur(gxy, gaussian_kernel, 0)
    
    # Compute orientation
    orientation = np.zeros_like(image, dtype=np.float64)
    
    for i in range(0, image.shape[0] - block_size, block_size):
        for j in range(0, image.shape[1] - block_size, block_size):
            block_gxx = np.sum(gxx[i:i+block_size, j:j+block_size])
            block_gyy = np.sum(gyy[i:i+block_size, j:j+block_size])
            block_gxy = np.sum(gxy[i:i+block_size, j:j+block_size])
            
            theta = 0.5 * np.arctan2(2 * block_gxy, block_gxx - block_gyy)
            orientation[i:i+block_size, j:j+block_size] = theta
            
    return orientation