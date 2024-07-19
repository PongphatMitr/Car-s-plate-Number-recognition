import easyocr
import cv2
import argparse
import numpy as np
# Constants
IOU_THRESHOLD = 0.5
# Helper functions
def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def process_detections(detections, detected_texts):
    new_detections = {}
    for detection in detections:
        text = detection['text']
        bbox = detection['bbox']
        if text in detected_texts:
            # Update existing detection
            existing_detection = detected_texts[text]
            existing_detection['bbox'] = bbox
        else:
            # Add new detection
            new_detections[text] = {'bbox': bbox}
    
    return new_detections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Read and process image
    image_path = args.image
    image = cv2.imread(image_path)
    result = reader.readtext(image_path)
    
    detections = [{'text': r[1], 'bbox': r[0]} for r in result]

    detected_texts = {}
    detected_texts = process_detections(detections, detected_texts)
    
    # Draw bounding boxes on the image
    for text, detection in detected_texts.items():
        bbox = detection['bbox']
        cv2.polylines(image, [np.array(bbox, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, (int(bbox[0][0]), int(bbox[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
