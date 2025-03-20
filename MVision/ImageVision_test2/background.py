import cv2
import numpy as np

# Load the image
image = cv2.imread("images/pieces.jpg")

# Reshape the image into a 2D array (each pixel becomes a row of RGB values)
pixels = image.reshape((-1, 3))

# Convert to float and apply K-Means clustering to find dominant colors
pixels = np.float32(pixels)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  # Number of clusters (adjust if needed)
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8 for comparison with the image
centers = np.uint8(centers)

# Identify the most frequent color (likely to be the background)
unique, counts = np.unique(labels, return_counts=True)
background_color = centers[unique[np.argmax(counts)]]

# Create a mask where pixels close to the background color are set to black
background_mask = np.all(np.abs(image - background_color) < 50, axis=-1)  # tolerance of 50

# Create a black background
black_background = np.zeros_like(image)

# Use the mask to copy only the non-background shapes onto the black background
black_background[~background_mask] = image[~background_mask]

# Show the result
small = cv2.resize(black_background, (black_background.shape[1] // 4, black_background.shape[0] // 4))  # Resize for display

#
cv2.imshow("Small Image", small)
cv2.waitKey(0)
cv2.destroyAllWindows()