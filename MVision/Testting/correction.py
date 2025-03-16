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

                # PERSPECTIVE CORRECTION START - ADD THIS PART
                # Finding chessboard corners again on the undistorted image
                undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                found_undistorted, corners_undistorted = cv2.findChessboardCorners(undistorted_gray, (8, 6), None)

                if found_undistorted:
                    # Refine the corner detection
                    corners_undistorted = cv2.cornerSubPix(undistorted_gray, corners_undistorted, (11, 11), (-1, -1),
                                                           criteria)

                    # Get the four outer corners of the chessboard
                    board_width = 8
                    board_height = 6

                    # Extract the four outer corners of the chessboard
                    top_left = corners_undistorted[0][0]
                    top_right = corners_undistorted[board_width - 1][0]
                    bottom_left = corners_undistorted[board_width * (board_height - 1)][0]
                    bottom_right = corners_undistorted[-1][0]

                    print(
                        f"Using corners for perspective correction: TL={top_left}, TR={top_right}, BL={bottom_left}, BR={bottom_right}")

                    # Define the source points (corners detected in the image)
                    src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

                    # Define the destination points (where we want the corners to be in the output image)
                    # Calculate the width and height based on the detected corners
                    width = int(max(
                        np.linalg.norm(top_right - top_left),
                        np.linalg.norm(bottom_right - bottom_left)
                    ))
                    height = int(max(
                        np.linalg.norm(bottom_left - top_left),
                        np.linalg.norm(bottom_right - top_right)
                    ))

                    print(f"Target dimensions for perspective correction: width={width}, height={height}")

                    # Create destination points for a perfect rectangle
                    dst_points = np.array([
                        [0, 0],  # top-left
                        [width - 1, 0],  # top-right
                        [width - 1, height - 1],  # bottom-right
                        [0, height - 1]  # bottom-left
                    ], dtype=np.float32)

                    # Calculate the perspective transform matrix
                    M = cv2.getPerspectiveTransform(src_points, dst_points)

                    # Apply the perspective transformation
                    warped = cv2.warpPerspective(undistorted, M, (width, height))

                    # Save the warped image
                    perspective_path = 'perspective_corrected_chessboard.jpg'
                    cv2.imwrite(perspective_path, warped)
                    print(f"Perspective-corrected image saved to: {perspective_path}")

                    # Update the plot to include the perspective correction
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
                    plt.title('Original Image')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
                    plt.title('Undistorted Image')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                    plt.title('Perspective Corrected')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig('full_correction_comparison.png')
                    print("Full correction comparison plot saved to: full_correction_comparison.png")
                    plt.show()  # This will display the plot before terminating

                    # Display with OpenCV
                    cv2.imshow("Original", test_img)
                    cv2.imshow("Undistorted", undistorted)
                    cv2.imshow("Perspective Corrected", warped)
                else:
                    print("Could not find chessboard corners in the undistorted image for perspective correction")
                    # Fall back to the original comparison
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
                # PERSPECTIVE CORRECTION END

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