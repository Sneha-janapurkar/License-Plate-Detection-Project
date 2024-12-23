import cv2
import numpy as np
import pytesseract

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_frame(frame):
    """Preprocess the frame for better plate detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 15, 15)  # Reduce noise while keeping edges sharp
    return gray

def detect_multiple_plates(frame):
    """Detect and recognize multiple license plates in the frame."""
    gray = preprocess_frame(frame)
    edge = cv2.Canny(gray, 150, 250)  # Adjusted thresholds for clearer edges
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    detected_plates = []
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Rectangle-like contour
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 6.0:  # Filtering for license plate shapes
                if 4000 < cv2.contourArea(contour) < 50000:  # Filter by area
                    # Crop and process the detected plate
                    plate = frame[y:y + h, x:x + w]
                    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    plate_thresh = cv2.adaptiveThreshold(
                        plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    
                    # Use OCR to recognize text
                    plate_text = pytesseract.image_to_string(
                        plate_thresh, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ).strip()
                    
                    if len(plate_text) > 2:  # Ensure valid plate detection
                        detected_plates.append(plate_text)
                        # Draw rectangle and display text on the frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, detected_plates

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Detecting license plates... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera.")
        break

    # Detect license plates in the frame
    frame, plates = detect_multiple_plates(frame)

    # Print detected plates in the terminal
    if plates:
        print("Detected Plates:", plates)

    # Display the live feed
    cv2.imshow("Live Feed - License Plate Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
