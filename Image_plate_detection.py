import cv2
import numpy as np
import pytesseract

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Step 1: Load the image
img = cv2.imread("car2.jpg")
if img is None:
    print("Error: Could not read the image. Ensure 'car1.jpg' is in the same folder as this script.")
    exit()

# Step 2: Preprocess the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
gray = cv2.bilateralFilter(gray, 13, 15, 15)  # Smooth while keeping edges sharp
edge = cv2.Canny(gray, 100, 200)  # Perform edge detection

# Step 3: Find contours
contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area

# Step 4: Detect the single most likely number plate
detected_plate = None
plate_text = ""

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:  # Check for rectangular shape
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 2.0 <= aspect_ratio <= 6.0:  # Filter based on aspect ratio
            plate = img[y:y+h, x:x+w]

            # Enhance plate for OCR
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            plate_thresh = cv2.adaptiveThreshold(
                plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # OCR to recognize text
            temp_text = pytesseract.image_to_string(
                plate_thresh,
                config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip()

            if temp_text:  # If text is detected, choose the first valid plate
                detected_plate = plate
                plate_text = temp_text
                cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
                cv2.putText(img, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Stop after detecting the first valid plate

# Step 5: Display results
if plate_text:
    print("Detected Number Plate:", plate_text)
else:
    print("No number plate detected.")

cv2.imshow("Original Image", img)
cv2.imshow("Edge Detected", edge)

if detected_plate is not None:
    cv2.imshow("Detected Plate", detected_plate)

cv2.waitKey(0)
cv2.destroyAllWindows()
