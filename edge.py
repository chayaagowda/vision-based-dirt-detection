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
input_batch = input_tensor.unsqueeze(0)  

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
