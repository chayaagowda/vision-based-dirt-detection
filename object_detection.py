import cv2
import numpy as np
import matplotlib.pyplot as plt
def detect_dirt(cv_image):

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []

    # Iterate through contours and filter based on area or other criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust this threshold based on your image and dirt size


            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)
            detected_boxes.append((x, y, w, h))
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)   
    return cv_image, detected_boxes


# Simulate live image capture (replace this with actual live capture code)
def get_live_image():


    # For demonstration, we'll use a static image
    image_path = '/Users/mac/Desktop/AIML/orange peel.JPG'
    image = cv2.imread(image_path)
    return image



# Manual verification function
def manual_verification(detected_boxes):
    TP, FP, FN = 0, 0, 0
    for box in detected_boxes:
        x, y, w, h = box
        plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))
        plt.show()



        # Ask the user if this detection is correct
        response = input("Is this detection correct? (y/n): ").strip().lower()
        if response == 'y':
            TP += 1
        else:
            FP += 1



    # Ask the user for any missed detections
    response = input("Were there any missed detections? (y/n): ").strip().lower()
    if response == 'y':
        FN += int(input("How many missed detections were there?: ").strip())
    return TP, FP, FN



# Capture and process live images
all_TP, all_FP, all_FN = 0, 0, 0
for _ in range(5):  # Simulate capturing 5 live images
    cv_image = get_live_image()
    processed_image, detected_boxes = detect_dirt(cv_image)


    # Display the image with detections
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Dirt')
    plt.axis('off')
    plt.show() 

    # Perform manual verification
    TP, FP, FN = manual_verification(detected_boxes)
    all_TP += TP
    all_FP += FP
    all_FN += FN


# Calculate and print overall metrics
precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0
recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1_score:.4f}')