import cv2
import numpy as np
from skimage import measure
import pytesseract

class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(after_clean_plate_img)
                if characters_on_plate is not None and len(characters_on_plate) >= 6:
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    return after_clean_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []
        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)
        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)
        if len(plates) > 0:
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):
        characters_found = self.segment_chars(plate)
        if characters_found:
            return characters_found
        else:
            return None

    def segment_chars(self, plate_img):
        V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]
        thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)
        plate_img = cv2.resize(plate_img, (400, int(plate_img.shape[0] * 400 / plate_img.shape[1])))
        thresh = cv2.resize(thresh, (400, int(thresh.shape[0] * 400 / thresh.shape[1])))
        bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        labels = measure.label(thresh, background=0)
        charCandidates = np.zeros(thresh.shape, dtype='uint8')
        characters = []
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype='uint8')
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate_img.shape[0])
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.5 and heightRatio < 0.95
                if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)
        contours = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            contours = self.sort_cont(contours)
            addPixel = 4
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                y = max(0, y - addPixel)
                x = max(0, x - addPixel)
                temp = bgr_thresh[y:y + h + (addPixel * 2), x:x + w + (addPixel * 2)]
                characters.append(temp)
            return characters
        else:
            return None

    def sort_cont(self, character_contours):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
        (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes), key=lambda b: b[1][i], reverse=False))
        return character_contours

    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area
        ratioMin = 2.5
        ratioMax = 7
        rmin = 0
        rmax = 0
        if ratioMin <= 0:
            rmin = 0
        else:
            rmin = ratioMin
        if ratioMax <= 0:
            rmax = 7
        else:
            rmax = ratioMax
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        if (area < min or area > max) or (ratio < rmin or ratio > rmax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect
        if width > height:
            angle = -rect_angle
        else:
            angle = 90 + rect_angle
        if angle > 15:
            return False
        if height == 0 or width == 0:
            return False
        area = height * width
        if not self.ratioCheck(area, width, height):
            return False
        else:
            return True

    def draw_rectangle(self, input_img):
        self.plates = self.find_possible_plates(input_img)
        for (i, coord) in enumerate(self.corresponding_area):
            cv2.drawContours(input_img, [coord], -1, (0, 255, 0), 2)
            for character in self.char_on_plate[i]:
                cv2.imshow('Character', character)
                cv2.waitKey(0)
        cv2.imshow('Plates Found', input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def extract_text_from_plate(plate_img):
    """ Extract text from license plate image using Tesseract OCR """
    config = '--oem 1 --psm 8 -l tha'
    text = pytesseract.image_to_string(plate_img, config=config)
    return text

# Load your image
image_path = "path/to/your/image.jpg"
input_img = cv2.imread(image_path)

# Create a PlateFinder instance
plate_finder = PlateFinder(minPlateArea=4500, maxPlateArea=30000)

# Find and draw rectangles around plates
plates = plate_finder.find_possible_plates(input_img)
if plates is not None:
    for plate in plates:
        cv2.imshow('Plate', plate)
        cv2.waitKey(0)
        text = extract_text_from_plate(plate)
        print(f"Recognized Text: {text}")
        cv2.destroyAllWindows()
else:
    print("No plates found")
