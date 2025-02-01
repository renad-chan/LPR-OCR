LPR-OCR: License Plate Recognition and Optical Character Recognition

Overview

This project focuses on License Plate Detection and OCR (Optical Character Recognition) using YOLO for object detection and EasyOCR for text extraction. The system detects vehicle license plates in images, enhances them, and extracts the plate numbers with improved accuracy.

Features

License Plate Detection using YOLO: Detects and crops license plates from images.

Multi-Technique Image Enhancement: Applies CLAHE, adaptive thresholding, and other preprocessing techniques.

OCR Processing with EasyOCR: Extracts plate numbers using multiple OCR attempts with configurable settings.

Multiple Detection Attempts: Tries different scales and confidence thresholds for robust results.

Text Formatting and Visualization: Displays detected plate numbers on images with annotations.

Requirements

Ensure you have the following dependencies installed:

pip install opencv-python numpy easyocr ultralytics matplotlib

Usage

To run the license plate detection and OCR script, modify the paths in main() and execute the script:

python script.py

Parameters

image_path: Path to the input image containing the vehicle.

yolo_model_path: Path to the trained YOLO model weights.

output_image_path: Path to save the annotated output image.

How It Works

Load Models: YOLO for object detection and EasyOCR for text recognition.

Detect License Plate: YOLO detects possible license plates with different scales and confidence thresholds.

Enhance Image: Preprocesses the cropped license plate region with different enhancement techniques.

OCR Processing: Runs multiple OCR configurations to extract text with high confidence.

Result Formatting: Extracted text is formatted into structured plate numbers (numbers first, then letters).

Visualization: The detected plate is annotated on the original image and saved as an output file.

Output Example

After processing, the script outputs:

Annotated image with detected plate and recognized text.

Debug images of different plate enhancement techniques.

Extracted plate number with confidence scores.

Notes

Ensure the YOLO model is trained for license plate detection.

For best results, use high-quality images with clear license plates.

Adjust OCR settings and preprocessing techniques for improved recognition in different conditions.

License

This project is for research and educational purposes only. Ensure compliance with local regulations regarding license plate recognition and data privacy.

Author

Developed by Renad Alhano.

