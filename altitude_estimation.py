
'''
This is the main and complete function to estimate the Relative Altitude using SIFT features on UDWA dataset
'''
# https://github.com/Aprus-system/UDWA/blob/main/README.md  UDWA datset
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Start the timer
start_time = time.time()

num_start = 0
num_end = 770
num_images = num_end - num_start

# ---- Update these lines to reflect the correct file path in your directory ----
data_df = pd.read_csv('UDWA/place01/place01.csv') 
directory_path = "UDWA/place01/images"
images_directory = "UDWA/place01/images/"

# Extract and subset altitude data from annotations
altitude = data_df['Altitude']
altitude = altitude.iloc[num_start:num_end]


# Generate a list of full paths to each image file within the specified directory
images_path = [os.path.join(directory_path, im) for im in oos.listdir(images_directory)]

# Select a subset of the image paths based on the specified start and end indices
selected_images_path = images_path[num_start:num_end]

# Create pairs of consecutive images
image_pairs = [(selected_images_path[i], selected_images_path[i + 1]) for i in range(num_images - 1)]
scales = []

ratio_thresh = 0.5
ii = 1

# for pair in image_pairs:
for pair in image_pairs:

    # Print the names of the images in the pair
    print(f"\n --- {ii} ---")
    print(f"Processing image pair: {pair[0]} and {pair[1]}")

    # Read the image pair
    im1 = cv2.imread(pair[0])
    im2 = cv2.imread(pair[1])

    # Convert to grayscale images
    II1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    II2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Resize the images
    I1 = cv2.resize(II1, (1920, 1080))
    I2 = cv2.resize(II2, (1920, 1080))

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Find the key-points and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(I1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(I2, None)

    print('Number of keypoints : ', len(keypoints1), len(keypoints2))

    # Match features
    bf = cv2.BFMatcher()
    sift_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    print('Number of matched keypoints :', len(sift_matches))

    good = []
    good_matches1 = []
    good_matches2 = []

    for m, n in sift_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches1.append(keypoints1[m.queryIdx])
            good_matches2.append(keypoints2[m.trainIdx])
            good.append(m)

    good_matches1 = np.float32([kp.pt for kp in good_matches1])
    good_matches2 = np.float32([kp.pt for kp in good_matches2])

    print('Number of good matched keypoints :', len(good_matches1), len(good_matches2))

    # Homography Matrix
    H, mask = cv2.findHomography(good_matches1, good_matches2, cv2.RANSAC, 5)
    mask = mask.ravel().tolist()

    # RANSAC inliers
    inliers1 = []
    inliers2 = []

    for i in range(good_matches1.shape[0]):
        if mask[i] == 1:
            inliers1.append(good_matches1[i])
            inliers2.append(good_matches2[i])

    inliers1 = np.array(inliers1)
    inliers2 = np.array(inliers2)

    print('Number of inliers :', len(inliers1), len(inliers2))

    inlier_matches = [m for i, m in enumerate(good) if i < len(mask) and mask[i]]

    # Calculate average scale ratio
    ratios = [keypoints2[m.trainIdx].size / keypoints1[m.queryIdx].size for m in inlier_matches]
    avg_ratio = sum(ratios) / len(ratios)

    scales.append(avg_ratio)

    ii = ii + 1

# Standard Scaling
scaler = StandardScaler()
scales = np.array(scales)

scales = scales.reshape(-1, 1)
scales_scaled = scaler.fit_transform(scales)

# Estimate the altitude by summing the relative altitude changes between consecutive image pairs
Z_scaled = np.zeros(num_images)
for i in range(1, num_images):
    Z_scaled[i] = Z_scaled[i - 1] + scales_scaled[i - 1]


# Standard Normalization
scaler0 = StandardScaler()
scaler1 = StandardScaler()

altitude = np.array(altitude)
altitude = altitude.reshape(-1, 1)
altitude_scaled = scaler0.fit_transform(altitude)

# Z_scaled = -1 * Z_scaled
estimated_altitude = np.array(-1 * Z_scaled)
estimated_altitude = estimated_altitude.reshape(-1, 1)
estimated_altitude_scaled = scaler1.fit_transform(estimated_altitude)


# Convert to the their original
altitude0_original = scaler0.inverse_transform(altitude_scaled)
altitude1_original = scaler0.inverse_transform(estimated_altitude_scaled)


# Convert to numpy arrays
altitude0_original = np.array(altitude0_original)
altitude0_original = altitude0_original.ravel()

altitude1_original = np.array(altitude1_original)
altitude1_original = altitude1_original.ravel()


idx = range(num_images)  # Creates a sequence from 0 to num_images - 1
plt.plot(idx, altitude0_original, label="Real")
plt.plot(idx, altitude1_original, label="Estimated")
plt.xlabel('Image Number', fontsize=12)
plt.ylabel('Altitude (meter)', fontsize=12)
plt.grid()
plt.legend(fontsize=12)
plt.show()
