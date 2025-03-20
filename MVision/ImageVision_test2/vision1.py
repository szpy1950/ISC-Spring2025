import cv2
import numpy as np


img = cv2.imread("images/pieces.jpg")  # Load original

# # DEBUG
# print(type(img))  # <class 'numpy.ndarray'>
# print(img.shape)  # (2026, 1800, 3)
# print(img.dtype)  # uint8

# img_modified = cv2.add(img, np.array([250, 0, 0], dtype=np.uint8))
# img_modified = cv2.subtract(img_modified, np.array([0, 100, 0], dtype=np.uint8))

def custom_kernel(size=10):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return kernel

# Apply custom kernel using filter2D
kernel = custom_kernel(10)  # A simple averaging kernel (3x3)
img_modified = cv2.filter2D(img, -1, kernel)


small = cv2.resize(img_modified, (img_modified.shape[1] // 4, img_modified.shape[0] // 4))  # Resize for display

#
cv2.imshow("Small Image", small)
cv2.waitKey(0)
cv2.destroyAllWindows()