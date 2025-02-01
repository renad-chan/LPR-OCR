import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from matplotlib import pyplot as plt

def enhance_image(image):
    """
    Enhanced preprocessing with multiple enhancement techniques.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Create multiple enhanced versions
    enhanced_versions = []
    # Version 1: Basic enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced1 = clahe.apply(gray)
    enhanced_versions.append(enhanced1)
    # Version 2: High contrast
    enhanced2 = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    enhanced_versions.append(enhanced2)
    # Version 3: Adaptive thresholding
    enhanced3 = cv2.adaptiveThreshold(gray, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    enhanced_versions.append(enhanced3)
    # Version 4: Otsu's thresholding
    _, enhanced4 = cv2.threshold(gray, 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(enhanced4)
    return enhanced_versions

def detect_and_crop_plate(yolo_model, image_path):
    """
    Improved plate detection with multiple detection attempts.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image {image_path}!")
    # Try different scales
    scales = [1.0, 1.5, 2.0]
    best_detections = []
    for scale in scales:
        # Resize image
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        resized = cv2.resize(image, (width, height))
        # Run detection with different confidence thresholds
        for conf_threshold in [0.2, 0.15, 0.1]:
            results = yolo_model(resized, conf=conf_threshold, iou=0.5)
            for result in results[0].boxes:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = map(int, (result.xyxy[0] / scale))
                confidence = float(result.conf[0])
                # Add padding
                height = y2 - y1
                width = x2 - x1
                padding_y = int(height * 0.4)  # Increased padding
                padding_x = int(width * 0.4)
                y1_pad = max(0, y1 - padding_y)
                y2_pad = min(image.shape[0], y2 + padding_y)
                x1_pad = max(0, x1 - padding_x)
                x2_pad = min(image.shape[1], x2 + padding_x)
                plate_region = image[y1_pad:y2_pad, x1_pad:x2_pad]
                if plate_region.size > 0:
                    enhanced_versions = enhance_image(plate_region)
                    best_detections.append({
                        'original': plate_region,
                        'enhanced_versions': enhanced_versions,
                        'coords': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
    # Sort detections by confidence
    best_detections.sort(key=lambda x: x['confidence'], reverse=True)
    return best_detections, image

def process_plate_with_ocr(reader, plate_detection):
    """
    Improved OCR processing with multiple attempts and verification.
    """
    all_results = []
    # Process each enhanced version
    for idx, img in enumerate(plate_detection['enhanced_versions']):
        # Try different scales
        for scale in [1.0, 1.5, 2.0, 2.5]:
            scaled_img = cv2.resize(img, None, fx=scale, fy=scale, 
                                  interpolation=cv2.INTER_CUBIC)
            # Try different OCR configurations
            configs = [
                {'allowlist': '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
                {'allowlist': '0123456789'},
                {'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
            ]
            for config in configs:
                results = reader.readtext(
                    scaled_img,
                    paragraph=False,
                    height_ths=0.3,
                    width_ths=0.3,
                    **config
                )
                for bbox, text, conf in results:
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    if cleaned_text:
                        all_results.append((cleaned_text, conf))
    # Process results to construct the plate number
    if not all_results:
        return "", 0.0
    # Sort results by confidence
    all_results.sort(key=lambda x: x[1], reverse=True)
    # Collect all unique characters
    numbers = []
    letters = []
    seen_chars = set()
    for text, _ in all_results:
        for char in text:
            if char not in seen_chars:
                if char.isdigit():
                    numbers.append(char)
                else:
                    letters.append(char)
                seen_chars.add(char)
    # Construct final plate number
    # Ensure numbers come before letters
    final_numbers = ''.join(numbers[:4])  # First 4 numbers
    final_letters = ''.join(letters[:3])  # First 3 letters
    final_text = f"{final_numbers}{final_letters}"  # Numbers come before letters
    avg_conf = sum(conf for _, conf in all_results[:5]) / min(5, len(all_results))
    return final_text, avg_conf

def main(image_path, yolo_model_path, output_image_path):
    """
    Main function with improved detection and visualization.
    """
    print("Loading models...")
    yolo_model = YOLO(yolo_model_path)
    reader = easyocr.Reader(['en'], gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
    print("Models loaded successfully")
    try:
        print("Detecting plates...")
        plate_detections, original_image = detect_and_crop_plate(yolo_model, image_path)
        if not plate_detections:
            print("No license plates detected!")
            return
        print(f"Found {len(plate_detections)} potential plates")
        # Process each detection
        for i, detection in enumerate(plate_detections):
            x1, y1, x2, y2 = detection['coords']
            confidence = detection['confidence']
            print(f"\nProcessing plate {i+1}")
            print(f"Detection confidence: {confidence:.2f}")
            # Process plate
            final_text, ocr_confidence = process_plate_with_ocr(reader, detection)
            print(f"OCR result: {final_text}")
            print(f"OCR confidence: {ocr_confidence:.2f}")

            # Split numbers and letters
            numbers = ''.join([c for c in final_text if c.isdigit()])
            letters = ''.join([c for c in final_text if c.isalpha()])
            formatted_text = f"EN: {numbers}, {letters}"  # Format as EN: <numbers>, <letters>

            # Draw on original image
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add text with background
            text_size = cv2.getTextSize(formatted_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            # Draw white background for text
            cv2.rectangle(original_image,
                          (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0] + 10, y1),
                          (255, 255, 255), -1)
            # Draw formatted text
            cv2.putText(original_image, formatted_text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # Save debug images
            for j, enhanced in enumerate(detection['enhanced_versions']):
                cv2.imwrite(f"debug_plate_{i}_enhanced_{j}.jpg", enhanced)
        # Save final result
        cv2.imwrite(output_image_path, original_image)
        print(f"\nAnnotated image saved to: {output_image_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = "Path to your image"
    yolo_model_path = "Path to your model"
    output_image_path = "Path to your output image"
    main(image_path, yolo_model_path, output_image_path)
