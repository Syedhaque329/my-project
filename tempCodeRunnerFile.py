import sys
import glob
import os
import numpy as np
import cv2
from PIL import Image
import easyocr
import re

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU

# Function to preprocess the image for better contrast and noise reduction
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny Edge Detection to highlight edges
    edges = cv2.Canny(blurred, 100, 200)

    return edges

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
    if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
        return False
    return True

# Function to clean up the plate image (extract the main text region)
def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
            result = reader.readtext(np.array(plate_im))

            if result:
                text = result[0][1]
                # Post-process the detected text to clean it up
                text = str("".join(re.split("[^a-zA-Z0-9]*", text)))
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

# Function to validate if the detected text is a valid vehicle number
def validate_number_plate(text):
    # Regex to check for common vehicle number formats (e.g., ABC1234, AB12CD3456)
    pattern = re.compile(r"^[A-Z]{2,3}[0-9]{1,2}[A-Z]{0,2}[0-9]{1,4}$")
    return bool(pattern.match(text))

# Quick sort implementation for sorting vehicle numbers
def partition(arr, low, high):
    i = (low - 1)
    pivot = arr[high]

    for j in range(low, high):
        if arr[j] < pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return (i + 1)

def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)
    return arr

# Binary search implementation
def binarySearch(arr, l, r, x):
    if r >= l:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binarySearch(arr, l, mid - 1, x)
        else:
            return binarySearch(arr, mid + 1, r, x)
    else:
        return -1

# Define the base directory where images are stored
base_dir = r"D:\project\Automatic-Number-plate-detection-for-Indian-vehicles-main - Copy\Dataset"

# Ensure the directory exists
if not os.path.exists(base_dir):
    print(f"Dataset directory not found at: {base_dir}")
    sys.exit(1)

# Looking for .jpeg images directly in the base_dir
image_files = glob.glob(os.path.join(base_dir, "*.jpeg"))

# Check if there are any .jpeg images in the folder
if not image_files:
    print(f"No .jpeg images found in {base_dir}")
    sys.exit(1)

# List to store vehicle number plates
array = []
res2 = ""  # Initialize res2

# Loop through .jpeg images in the base_dir
for img_path in image_files:
    # Read the image
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    # Display the image to verify it's correct (you can comment this out after testing)
    img_resized = cv2.resize(img, (600, 600))  # Resize for display
    cv2.imshow("Image of car", img_resized)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # Process the image further (e.g., number plate detection)
    number_plate = number_plate_detection(img)
    if number_plate:
        res2 = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))
        res2 = res2.upper()
        print(res2)
        array.append(res2)

# Sorting vehicle numbers
array = quickSort(array, 0, len(array) - 1)
print("\n\n")
print("The Vehicle numbers registered are:-")
for i in array:
    print(i)
print("\n\n")

# Searching (Ensure res2 is defined)
search_number_plate = ""  # Add a variable to store the search result

# Searching in the same directory
search_files = glob.glob(os.path.join(base_dir, "*.jpeg"))

if not search_files:
    print(f"No .jpeg search images found in {base_dir}")
    sys.exit(1)

for img_path in search_files:
    # Read the image
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load search image: {img_path}")
        continue

    # Process the image for number plate detection
    number_plate = number_plate_detection(img)
    if number_plate:
        search_number_plate = str("".join(re.split("[^a-zA-Z0-9]*", number_plate)))

if search_number_plate:
    print("The car number to search is:- ", search_number_plate)
else:
    print("No number plate detected in the search image.")

# Perform binary search for the number plate
result = binarySearch(array, 0, len(array) - 1, search_number_plate)
if result != -1:
    print("\n\nThe Vehicle is allowed to visit.")
else:
    print("\n\nThe Vehicle is not allowed to visit.")
