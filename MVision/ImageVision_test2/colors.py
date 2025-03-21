import numpy as np
import cv2

image = cv2.imread("images/pieces.jpg")  # Use cv2.imread to load the image
n_clusters = 5

# to reduce complexity resize the image
data = cv2.resize(image, (100, 100)).reshape(-1, 3)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

cluster_sizes = np.bincount(labels.flatten())

palette = []
for cluster_idx in np.argsort(-cluster_sizes):
    palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
palette = np.hstack(palette)

sf = image.shape[1] / palette.shape[1]
out = np.vstack([image, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])

small = cv2.resize(out, (out.shape[1] // 4, out.shape[0] // 4))  # Resize for display

#
cv2.imshow("Small Image", small)
cv2.waitKey(0)
cv2.destroyAllWindows()