import cv2

# Load the RGB image
# Replace 'image.jpg' with the path to your image file
image = cv2.imread('FLASH.jpeg')

# Convert the RGB image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
# Adjust the thresholds as needed for your image
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)

# Display the edges
cv2.imshow('Edges', edges)

# Save the grayscale and edge-detected images
cv2.imwrite('gray_image.jpg', gray_image)
cv2.imwrite('edges.jpg', edges)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
