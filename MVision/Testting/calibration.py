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

# Store image size for later calibration
img_size = None

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

    # Store the image size for calibration later
    if img_size is None:
        img_size = (image.shape[1], image.shape[0])  # Width, Height

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

# Check if we have enough points for calibration
if len(objpoints) > 0 and len(imgpoints) > 0 and img_size is not None:
    print("Performing camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if ret:
        print("Camera calibrated successfully!")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")

        # Save the calibration results for future use
        calibration_data = {
            'camera_matrix': mtx,
            'dist_coeffs': dist
        }
        np.save('camera_calibration.npy', calibration_data)
        print("Calibration data saved to 'camera_calibration.npy'")

        # If you want to test the calibration on a test image
        test_img_path = 'testing/chessboard2.jpg'
        if os.path.isfile(test_img_path):
            print(f"Testing calibration on {test_img_path}")
            test_img = cv2.imread(test_img_path)

            if test_img is not None:
                # Undistort the image
                undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)

                # Save the undistorted image
                output_path = 'undistorted_test_image.jpg'
                cv2.imwrite(output_path, undistorted)
                print(f"Undistorted image saved to: {output_path}")

                # Plot the original and undistorted images using matplotlib
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
                plt.title('Undistorted Image')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig('comparison_plot.png')
                print("Comparison plot saved to: comparison_plot.png")
                plt.show()  # This will display the plot before terminating

                # Display original and undistorted images using OpenCV
                cv2.imshow("Original", test_img)
                cv2.imshow("Undistorted", undistorted)
                print("Press any key to close the windows and terminate the program...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Could not read test image {test_img_path}")
        else:
            print(f"Test image {test_img_path} not found")
    else:
        print("Camera calibration failed")
else:
    print("Not enough data for calibration")

print("Done.")