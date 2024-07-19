# import easyocr
# import cv2
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from difflib import get_close_matches

# # Constants
# IOU_THRESHOLD = 0.5

# # Helper functions
# def calculate_iou(box_a, box_b):
#     """
#     Calculates the Intersection-over-Union (IOU) between two bounding boxes.

#     Args:
#         box_a (list): List of coordinates representing the first bounding box (e.g., [x1, y1, x2, y2]).
#         box_b (list): List of coordinates representing the second bounding box.

#     Returns:
#         float: The IOU value between the two bounding boxes.
#     """
#     x_a = max(box_a[0], box_b[0])
#     y_a = max(box_a[1], box_b[1])
#     x_b = min(box_a[2], box_b[2])
#     y_b = min(box_a[3], box_b[3])
#     inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
#     box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
#     box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
#     iou = inter_area / float(box_a_area + box_b_area - inter_area)
#     return iou

# def process_detections(detections, detected_texts, min_confidence=0.7, aspect_ratio_filter=True):
#     """
#     Processes carplate detections by filtering based on confidence and aspect ratio.

#     Args:
#         detections (list): List of dictionaries containing detection information (text, bbox, confidence).
#         detected_texts (dict): Dictionary to store unique detections and their bounding boxes.
#         min_confidence (float, optional): Minimum confidence threshold for filtering (default: 0.7).
#         aspect_ratio_filter (bool, optional): Flag to enable aspect ratio filtering (default: True).

#     Returns:
#         dict: Dictionary containing filtered detections and their bounding boxes.
#     """
#     new_detections = {}

#     for detection in detections:
#         text = detection['text']
#         bbox = detection['bbox']
#         confidence = detection.get('confidence', 1.0)

#         # Fix for bbox type (assuming it's a list of coordinates)
#         if isinstance(bbox[0], list):
#             bbox = [val[0] for val in bbox]  # Unpack coordinates from nested lists

#         # Filter by confidence and aspect ratio (optional)
#         if confidence < min_confidence:
#             continue
#         if aspect_ratio_filter:
#             w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
#             aspect_ratio = float(w) / h
#             if aspect_ratio < 0.5 or aspect_ratio > 3:
#                 continue

#         if text in detected_texts:
#             # Update existing detection
#             existing_detection = detected_texts[text]
#             existing_detection['bbox'] = bbox
#         else:
#             # Add new detection
#             new_detections[text] = {'bbox': bbox, 'confidence': confidence}

#     return new_detections

# def correct_text(text, dictionary):
#     """
#     Corrects OCR-detected text using a dictionary of common words/phrases.

#     Args:
#         text (str): The OCR-detected text.
#         dictionary (list): A list of common words/phrases for correction.

#     Returns:
#         str: The corrected text.
#     """
#     closest_matches = get_close_matches(text, dictionary, n=1, cutoff=0.7)
#     if closest_matches:
#         return closest_matches[0]
#     return text

# def main():
#     """
#     Main function to read an image, perform carplate detection and recognition, and display results.
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', required=True, help='Path to the image file')
#     args = parser.parse_args()

#     # Initialize EasyOCR reader for Thai and English
#     reader = easyocr.Reader(['th', 'en'])

#     # Common Thai words/phrases dictionary
#     thai_dictionary = [
#         "กรุงเทพมหานคร", "ประเทศไทย", "รถยนต์", "ทะเบียน", "จังหวัด", "สมุทรปราการ",
#         "ปทุมธานี", "พระนครศรีอยุธยา", "นนทบุรี", "นครปฐม", "สมุทรสาคร"
#     ]

#     # Read and process image
#     image_path = args.image
#     image = cv2.imread(image_path)
#     result = reader.readtext(image_path)

#     # Debug: Print OCR results
#     print("OCR Results:", result)

#     detections = [{'text': r[1], 'bbox': r[0], 'confidence': r[2]} for r in result]

#     detected_texts = process_detections(detections, {})

#     # Debug: Print filtered detections
#     print("Filtered Detections:", detected_texts)

#     # Draw bounding boxes on the image and display recognized text
#     for text, detection in detected_texts.items():
#         corrected_text = correct_text(text, thai_dictionary)
#         bbox = detection['bbox']
#         # Ensure bbox is in correct format for drawing
#         if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
#             bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
#         cv2.polylines(image, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
#         cv2.putText(image, corrected_text, (int(bbox[0][0]), int(bbox[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Save and display the result
#     output_path = 'output_image.jpg'
#     cv2.imwrite(output_path, image)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

#     print(f"Output image saved as {output_path}")

# if __name__ == '__main__':
#     main()

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
# font license plate: https://www.f0nt.com/release/saruns-thangluang/
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return invert

def read_thai_plate(image):
    custom_config = r'--oem 3 --psm 7 -l tha+eng'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()

def put_thai_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("tahoma.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

image_path = r'C:\Users\gistda\Desktop\save\SeniorProj\regplate\thaiplate.jpg'
img = cv2.imread(image_path)
original_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

detected_plate_text = "No plate detected"

for idx, contour in enumerate(contours):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
    
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(contour)
        license_img = original_img[y:y+h, x:x+w]
        
        processed_license = preprocess_for_ocr(license_img)
        processed_license = cv2.resize(processed_license, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        plate_text = read_thai_plate(processed_license)
        
        print(f"Contour {idx} - Raw OCR output:", plate_text)
        
        if plate_text:
            detected_plate_text = plate_text
            
            cv2.imshow(f"License Detected {idx}", license_img)
            cv2.imshow(f"Processed License {idx}", processed_license)
            
            cv2.drawContours(img, [contour], -1, (0, 255, 255), 3)
            
            img = put_thai_text(img, detected_plate_text, (x, y-30), 32, (0, 255, 0))

text_display = np.zeros((100, 600, 3), dtype=np.uint8)
text_display = put_thai_text(text_display, f"Detected plate: {detected_plate_text}", (10, 50), 32, (255, 255, 255))

cv2.imshow("Original Image with Detected Plate", img)
cv2.imshow("Detected Text", text_display)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Final detected license plate text:", detected_plate_text)