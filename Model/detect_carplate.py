import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Perform morphological operations to enhance text
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph, image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()

    # Preprocess the image
    image_path = args.image
    preprocessed_image, original_image = preprocess_image(image_path)

    # Convert to PIL Image
    pil_image = Image.fromarray(preprocessed_image)

    # Perform OCR with Thai language and numbers
    custom_config = r'--oem 3 --psm 6 -l tha+eng'
    text = pytesseract.image_to_string(pil_image, config=custom_config)

    # Get bounding boxes
    boxes = pytesseract.image_to_boxes(pil_image, config=custom_config)

    # Draw bounding boxes
    draw = ImageDraw.Draw(pil_image)
    for box in boxes.splitlines():
        box = box.split(' ')
        draw.rectangle(((int(box[1]), int(box[2])), (int(box[3]), int(box[4]))), outline="red")

    print("Recognized Text:", text)

    # Convert back to OpenCV image
    result_image = np.array(pil_image)

    # Display the result
    cv2.imshow('Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
