import sys
import os
import sqlite3
import numpy as np
import cv2
import easyocr
from PIL import Image
import re
import logging
from datetime import datetime

# Initialize EasyOCR Reader (Set gpu=True if you have a compatible GPU)
reader = easyocr.Reader(['en'], gpu=False)

# Set up logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Function to connect to the SQLite database (using singleton pattern for reuse)
class Database:
    _conn = None

    @staticmethod
    def connect():
        if Database._conn is None:
            try:
                Database._conn = sqlite3.connect('vehicle_database.db')  # Connect to your SQLite database
                logger.info("Connecting to database...")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                sys.exit(1)
        return Database._conn

    @staticmethod
    def close():
        if Database._conn:
            Database._conn.close()
            Database._conn = None
            logger.info("Database connection closed.")

# Function to fetch vehicle details from the database
def fetch_vehicle_details_from_db(plate_number):
    try:
        conn = Database.connect()
        cursor = conn.cursor()

        # Query to fetch vehicle details based on the plate number
        cursor.execute("SELECT * FROM vehicles WHERE plate_number = ?", (plate_number,))
        result = cursor.fetchone()

        if result:
            vehicle_details = {
                'plate_number': result[0],
                'owner': result[1],
                'make': result[2],
                'model': result[3],
                'year': result[4],
                'status': result[5]
            }
            return vehicle_details
        else:
            return None
    except sqlite3.Error as e:
        logger.error(f"Error occurred while fetching data: {e}")
        return None

# Function to validate if the detected text is a valid vehicle number
def validate_number_plate(text):
    # Relaxed regex to match common vehicle plate formats (e.g., ABC1234, AB12CD3456)
    pattern = re.compile(r"^[A-Z]{2,3}[0-9]{1,2}[A-Z0-9]{1,4}$")
    return bool(pattern.match(text))

# Function to preprocess the image for better contrast and noise reduction
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjusting contrast and brightness dynamically based on image intensity
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 50    # Brightness control (0-100)
    img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Adaptive Thresholding to highlight edges and characters
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply dilation and erosion for noise removal and edge connection
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

# Function to detect contours and return the best candidates
def detect_contours(img):
    edges = preprocess_image(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours

# Function to check the ratio of width and height for valid plate candidates
def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio
    # Adjusted to be more flexible for detection
    if (area < 1000 or area > 100000) or (ratio < 2.5 or ratio > 7):
        return False
    return True

# Function to clean up the plate image (extract the main text region)
def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)
        max_cnt = num_contours[max_cntr_index]
        max_cntArea = contour_area[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate, None

        final_img = thresh[y:y + h, x:x + w]
        return final_img, [x, y, w, h]
    else:
        return plate, None

# Function to detect the number plate using EasyOCR
def number_plate_detection(img):
    contours = detect_contours(img)

    for cnt in contours:
        min_rect = cv2.minAreaRect(cnt)
        if ratio_and_rotation(min_rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y + h, x:x + w]

            # Convert the plate image to PIL format and use EasyOCR for text recognition
            plate_im = Image.fromarray(plate_img)
            result = reader.readtext(np.array(plate_im), detail=0, paragraph=True)

            if result:
                detected_text = result[0]
                logger.info(f"Detected text: {detected_text}")  # Debug print for detected text
                # Post-process the detected text to clean it up
                text = str("".join(re.split("[^a-zA-Z0-9]*", detected_text)))
                if validate_number_plate(text):
                    return text.strip()
    return None

# Function to check if the bounding box of the detected plate is valid (aspect ratio, rotation)
def ratio_and_rotation(rect):
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
    if not ratioCheck(area, width, height):
        return False
    else:
        return True

# Define the base directory where images are stored
base_dir = r"D:\project\Automatic-Number-plate-detection-for-Indian-vehicles-main - Copy\Dataset"

# Ensure the directory exists
if not os.path.exists(base_dir):
    logger.error(f"Dataset directory not found at: {base_dir}")
    sys.exit(1)

# Initialize camera capture
cap = cv2.VideoCapture(0)  # Using the default camera

if not cap.isOpened():
    logger.error("Error: Could not open webcam.")
    sys.exit(1)

# List to store vehicle number plates
array = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        logger.error("Failed to capture image.")
        break

    # Display the captured frame
    cv2.imshow('Capture Image - Press q to quit', frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit the loop
        logger.info("Exiting...")
        break

    if key == ord('c'):  # Capture the current image
        logger.info("Capturing image...")

        # Process the captured image (number plate detection)
        number_plate = number_plate_detection(frame)
        if number_plate:
            res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))
            res2 = res2.upper()
            logger.info(f"Detected Plate: {res2}")
            array.append(res2)

            # Draw the bounding box around the detected plate
            cv2.putText(frame, f"Plate: {res2}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Fetch vehicle details from the database
            vehicle_details = fetch_vehicle_details_from_db(res2)

            if vehicle_details:
                logger.info(f"Vehicle Details for Plate {res2}:")
                logger.info(f"Owner: {vehicle_details['owner']}")
                logger.info(f"Make: {vehicle_details['make']}")
                logger.info(f"Model: {vehicle_details['model']}")
                logger.info(f"Year: {vehicle_details['year']}")
                logger.info(f"Status: {vehicle_details['status']}")
            else:
                logger.info(f"No details found for Plate {res2}")
        else:
            logger.info("No plate detected in the current image.")

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
Database.close()
