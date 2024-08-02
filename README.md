# UAV-Altitude-Estimation
This repository contains the code and data used for estimating UAV altitude by analyzing pairs of consecutive images using SIFT (Scale-Invariant Feature Transform) features. 
The code processes a series of images, extracts keypoints, matches them, and calculates the relative altitude changes between consecutive image pairs.

Description:
This code is focused on estimating the altitude of a UAV using visual information from images. 
The process involves: 
	-Reading images and corresponding altitude data from the UDWA dataset.
	-Extracting SIFT features from pairs of consecutive images.
	-Matching the features to compute the homography matrix.
	-Estimating the altitude changes between images by analyzing the scale of the matched keypoints.
	-Visualizing the estimated altitude against the actual altitude.

Dataset:
The images and altitude data used in this project are part of the UDWA (UAV Dataset with Altitude), which can be found in the https://github.com/Aprus-system/UDWA/blob/main/README.md

Structure (Data):

"place01.csv" Contains the altitude data corresponding to the images.

"place01/images" Contains the image files used for the analysis.

The code in this repository are provided for academic and research purposes.
