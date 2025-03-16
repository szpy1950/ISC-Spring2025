# Import required libraries
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os  # Added for path checking

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Debug: Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Make a list of calibration images
images = glob.glob('testing/chessboard2.jpg')

# Debug: Print number of images found
print(f"Number of images found: {len(images)}")
# Debug: Print the image paths
print(f"Image paths: {images}")

for idx, filename in enumerate(images):
    print(f"Processing image {idx + 1}/{len(images)}: {filename}")

    # Check if file exists and is readable
    if not os.path.isfile(filename):
        print(f"Error: File {filename} does not exist")
        continue

    # Read the image and convert it to a grayscale
    image = cv2.imread(filename)

    # Debug: Check if image was loaded properly
    if image is None:
        print(f"Error: Could not read image {filename}")
        continue

    print(f"Image shape: {image.shape}")

    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    print("Looking for chessboard corners...")
    found, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # Debug: Print if corners were found
    print(f"Chessboard corners found: {found}")

    # If found, add object points, image points (after refining them)
    if found == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(image, (8, 6), corners, found)
        print("Displaying image with corners. Press any key to continue...")
        cv2.imshow("Corners", image)
        cv2.waitKey(0)
    else:
        print("No chessboard corners found in this image")

print("Destroying all windows...")
cv2.destroyAllWindows()
print("Done.")


