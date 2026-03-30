import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_dirt(cv_image):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Threshold based on dirt size
        if area > 100: 
            x, y, w, h = cv2.boundingRect(contour)
            detected_boxes.append((x, y, w, h))
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv_image, detected_boxes

def get_live_image():
    image_path = '/Users/mac/Desktop/AIML/orange peel.JPG'
    image = cv2.imread(image_path)
    return image

def manual_verification(detected_boxes):
    TP = len(detected_boxes)  
    FP = 0
    FN = 0
    return TP, FP, FN

# Store results instead of displaying in the loop
all_images = []
all_TP, all_FP, all_FN = 0, 0, 0

for _ in range(1):
    cv_image = get_live_image()
    processed_image, detected_boxes = detect_dirt(cv_image)
    # Store images for display later
    all_images.append(processed_image)  

    TP, FP, FN = manual_verification(detected_boxes)
    all_TP += TP
    all_FP += FP
    all_FN += FN

# Display images only once after the loop
for image in all_images:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Dirt')
    plt.axis('off')
    plt.show()

# Calculate and print overall metrics once
precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0
recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1_score:.4f}')