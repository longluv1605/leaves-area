import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path('./src/baseline/config.yaml')
try:
    with CONFIG_PATH.open('r') as file:
        config = yaml.safe_load(file)
        if not config or 'CONFIG' not in config:
            raise ValueError("Invalid or empty configuration file")
        APP_CONFIG = config['CONFIG']
except FileNotFoundError:
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML file: {e}")

def get_color_thresholds(color_bgr: List[int], config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert BGR color to HSV and determine color thresholds.

    Args:
        color_bgr: Color in BGR format (Blue, Green, Red).
        config: Configuration dictionary with threshold parameters.

    Returns:
        Tuple of lower and upper thresholds in HSV space.

    Raises:
        ValueError: If color_bgr is invalid.
    """
    if len(color_bgr) != 3 or not all(0 <= x <= 255 for x in color_bgr):
        raise ValueError("color_bgr must be a list of 3 integers in range [0, 255]")

    color_array = np.uint8([[color_bgr]])
    hsv_color = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
    hue = hsv_color[0][0][0]

    lower = np.array([
        max(0, hue - config['hue_range_lower']),
        config['saturation_threshold'],
        config['value_threshold']
    ], dtype=np.uint8)
    upper = np.array([
        min(179, hue + config['hue_range_upper']),
        255,
        255
    ], dtype=np.uint8)

    return lower, upper

def create_mask(hsv_image: np.ndarray, color_bgr: List[int], config: Dict) -> np.ndarray:
    """Create a binary mask based on color thresholds.

    Args:
        hsv_image: Image in HSV color space.
        color_bgr: Target color in BGR format.
        config: Configuration dictionary.

    Returns:
        Binary mask of the segmented region.
    """
    lower, upper = get_color_thresholds(color_bgr, config)
    mask = cv2.inRange(hsv_image, lower, upper)
    return cv2.medianBlur(mask, config['median_blur_ksize'])

def find_contours(mask: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Find contours and hierarchy from a binary mask.

    Args:
        mask: Binary mask.

    Returns:
        Tuple of contours and hierarchy.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy

def detect_circles(image: np.ndarray, config: Dict) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Detect circles in an image using HoughCircles.

    Args:
        image: Input image in BGR format.
        config: Configuration dictionary.

    Returns:
        Tuple of detected circles, grayscale image, and edge image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, config['blur_kernel_size'], config['blur_sigma'])
    edges = cv2.Canny(gray, config['canny_lower'], config['canny_upper'])

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=config['min_distance_circles'],
        param1=config['hough_param1'],
        param2=config['hough_param2'],
        minRadius=config['min_radius_coin'],
        maxRadius=config['max_radius_coin']
    )
    return circles, gray, edges

def process_circles(circles: Optional[np.ndarray], result_image: np.ndarray, config: Dict) -> Tuple[np.ndarray, float]:
    """Process detected circles: draw all circles and select the first as the reference coin.

    Args:
        circles: Array of detected circles.
        result_image: Image to draw circles on.
        config: Configuration dictionary.

    Returns:
        Tuple of selected coin (circle) and scale factor (mm/pixel).

    Raises:
        ValueError: If no circles are detected.
    """
    if circles is None:
        raise ValueError("No circles detected")

    circles = np.uint16(np.around(circles))
    for coin in circles[0]:
        cv2.circle(
            result_image,
            (coin[0], coin[1]),
            coin[2],
            config['circle_color'],
            config['circle_thickness']
        )

    coin = circles[0][0]
    coin_diameter_px = coin[2] * 2
    scale = config['coin_diameter_mm'] / coin_diameter_px
    return coin, scale

def calculate_contour_area(
    contours: List[np.ndarray],
    hierarchy: np.ndarray,
    idx: int,
    processed: Set[int]
) -> float:
    """Calculate the actual area of a contour (excluding inner contours).

    Args:
        contours: List of contours.
        hierarchy: Contour hierarchy.
        idx: Index of the contour to process.
        processed: Set of processed contour indices.

    Returns:
        Actual area in pixels.
    """
    outer_area = cv2.contourArea(contours[idx])
    inner_area = 0
    child_idx = hierarchy[0][idx][2]

    while child_idx != -1:
        inner_area += cv2.contourArea(contours[child_idx])
        processed.add(child_idx)
        child_idx = hierarchy[0][child_idx][0]

    return max(0, outer_area - inner_area)

def draw_valid_contours(
    mask: np.ndarray,
    contours: List[np.ndarray],
    idx: int,
    hierarchy: np.ndarray
) -> None:
    """Draw valid contours on the mask, excluding inner regions.

    Args:
        mask: Mask to draw contours on.
        contours: List of contours.
        idx: Index of the contour to draw.
        hierarchy: Contour hierarchy.
    """
    cv2.drawContours(mask, [contours[idx]], -1, 255, -1)
    child_idx = hierarchy[0][idx][2]
    while child_idx != -1:
        cv2.drawContours(mask, [contours[child_idx]], -1, 0, -1)
        child_idx = hierarchy[0][child_idx][0]

def compute_area(
    image_path: str,
    config: Dict = APP_CONFIG,
    min_area: Optional[int] = None
) -> Dict[str, Union[float, List[np.ndarray], List[str]]]:
    """Calculate leaf area based on an image and a reference coin.

    Args:
        image_path: Path to the input image.
        config: Configuration dictionary. Defaults to APP_CONFIG.
        min_area: Minimum contour area. Defaults to config['min_contour_area'].

    Returns:
        Dictionary containing areas, images, and labels.

    Raises:
        FileNotFoundError: If the image file cannot be read.
        ValueError: If no contours are found or the image is invalid.
    """
    min_area = min_area if min_area is not None else config['min_contour_area']

    # Read and validate image
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image from: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    result = image.copy()

    # Segment leaf and find contours
    leaf_mask = create_mask(hsv, config['green_bgr'], config)
    contours, hierarchy = find_contours(leaf_mask)
    if not contours:
        raise ValueError("No contours found in the image")

    # Detect coin
    circles, gray, edges = detect_circles(image, config)
    final_mask = np.zeros_like(leaf_mask)

    # Prepare output
    images = [image, hsv, gray, edges, result, final_mask]
    labels = ['Original', 'HSV', 'Gray', 'Edges', 'Result', 'Mask']

    if circles is None:
        print("No coins detected in the image")
        return {
            'area_px': 0,
            'area_mm2': 0,
            'images': images,
            'labels': labels
        }

    # Process coin
    try:
        coin, scale = process_circles(circles, result, config)
    except ValueError as e:
        print(f"Error processing circles: {e}")
        return {
            'area_px': 0,
            'area_mm2': 0,
            'images': images,
            'labels': labels
        }

    # Calculate leaf area
    total_area_px = 0
    total_area_mm2 = 0
    processed = set()
    valid_contours = []

    for i in range(len(contours)):
        if i in processed or hierarchy[0][i][3] != -1:
            continue

        true_area_px = calculate_contour_area(contours, hierarchy, i, processed)
        if true_area_px <= 0 or true_area_px < min_area:
            continue

        total_area_px += true_area_px
        true_area_mm2 = true_area_px * (scale ** 2)
        total_area_mm2 += true_area_mm2
        valid_contours.append(contours[i])
        draw_valid_contours(final_mask, contours, i, hierarchy)

    # Draw results
    cv2.drawContours(result, valid_contours, -1, config['contour_color'], config['contour_thickness'])
    cv2.circle(
        result,
        (coin[0], coin[1]),
        coin[2],
        config['coin_color'],
        -1
    )

    print(f"Total area in pixels: {total_area_px:.2f}")
    print(f"Total area in mm²: {total_area_mm2:.2f}")

    return {
        'area_px': total_area_px,
        'area_mm2': total_area_mm2,
        'images': images,
        'labels': labels
    }