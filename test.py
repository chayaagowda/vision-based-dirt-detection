import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Load the image using PIL
image_path = '/Users/mac/Desktop/AIML/orange peel.JPG'
image = Image.open(image_path).convert('RGB')
# Define a torchvision transform to preprocess the image
preprocess = transforms.Compose([
    # Resize to 256x256
    transforms.Resize((256, 256)),  
    # Convert PIL image to tensor
    transforms.ToTensor()          
])
# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
# Convert tensor to NumPy array for edge detection
np_image = (input_batch[0].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
# Edge detection using Canny algorithm
edges = cv2.Canny(np_image, 100, 200)
# Convert edge-detected image back to tensor
edges_tensor = transforms.ToTensor()(edges)
# Convert tensors back to images for display
input_image = transforms.ToPILImage()(input_tensor.squeeze(0))
edges_image = Image.fromarray(edges)
# Display the original image and the edge-detected image side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(input_image)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(edges_image, cmap='gray')
axes[1].set_title('Edge-detected Image')
axes[1].axis('off')
plt.tight_layout()
plt.show()
# Convert the original image to OpenCV format for dirt detection
cv_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
# Convert to grayscale
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Use adaptive thresholding to binarize the image
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Iterate through contours and filter based on area or other criteria
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Adjust this threshold based on your image and dirt size
        # Draw a bounding box around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Display the results using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Dirt')
plt.axis('off')
plt.show()