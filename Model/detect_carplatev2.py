# -*- coding: utf-8 -*-
import easyocr
import cv2
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Helper functions
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return invert

def read_thai_plate(image):
    reader = easyocr.Reader(['th', 'en'], gpu=False)  # Use CPU for better accuracy
    result = reader.readtext(image, paragraph=True, detail=0)  # Use paragraph mode for better context
    return result

def put_thai_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(r"C:\Users\gistda\Desktop\save\Senior\Model\font\SarunsThangLuang\Sarun's ThangLuang.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to the image file')
    args = parser.parse_args()

    # Read and process image
    image_path = args.image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Unable to load image. Please check the path.")
        return

    original_img = img.copy()

    # Preprocess the image for better OCR results
    processed_img = preprocess_for_ocr(img)

    # Find contours to detect license plate area
    contours, _ = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    detected_plate_text = "No plate detected"

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:  # Check if the contour has four points
            x, y, w, h = cv2.boundingRect(contour)
            license_img = original_img[y:y+h, x:x+w]
            license_img = cv2.resize(license_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Read Thai text from the license plate
            results = read_thai_plate(license_img)
            if results:
                detected_plate_text = results[0]  # Only plate number if there's one line

            # Draw bounding box and put detected text
            cv2.drawContours(original_img, [contour], -1, (0, 255, 255), 3)
            original_img = put_thai_text(original_img, detected_plate_text, (x, y - 30), 32, (0, 255, 0))

            # Break after the first valid detection to avoid multiple bounding boxes
            break

    # Save the result
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, original_img)
    print(f"Output image saved as {output_path}")

if __name__ == '__main__':
    main()
