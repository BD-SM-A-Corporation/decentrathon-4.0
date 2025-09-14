import cv2
import numpy as np
from deskew import determine_skew

def estimate_noise_std_robust(image):
    """
    Estimates noise by finding the standard deviation of residuals after
    applying a median filter.

    Args:
        image (np.ndarray): The grayscale image.

    Returns:
        float: The estimated noise level.
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    median_filtered_image = cv2.medianBlur(image.astype(np.uint8), 5)

    noise_residuals = image - median_filtered_image

    noise_std = np.std(noise_residuals)

    return noise_std

def preprocess_image(image_path):
    """
    Preprocessing optimized for PaddleOCR:
    - grayscale
    - light denoising
    - deskew
    - adaptive thresholding
    """
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate noise
    noise_level = estimate_noise_std_robust(grayscale)

    # Use a safer dynamic rule: always within [5, 10]
    denoising_h = min(7, max(5, int(noise_level * 0.7)))
    print(f"Estimated noise level: {noise_level:.2f}. Using h={denoising_h} for denoising.")

    # 1. Light denoising
    denoised = cv2.fastNlMeansDenoising(
        grayscale, None, h=denoising_h,
        templateWindowSize=7, searchWindowSize=21
    )

    # 2. Deskew
    angle = determine_skew(denoised)
    if angle is not None:
        (h, w) = denoised.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            denoised, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    else:
        deskewed = denoised

    # 3. Adaptive thresholding
    final_image = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2  # чуть больше, чем 10 → белый фон лучше отделяется
    )

    # 4. Morphology (only if text is broken)
    # kernel = np.ones((1, 1), np.uint8)
    # final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)

    # Convert to 3-channel BGR (PaddleOCR expects this)
    # final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

    return deskewed
