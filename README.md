# UAV-Altitude-Estimation using SIFT Features
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

The images and additional files can be accessed via the following Google Drive link:
[Download the files] (https://drive.google.com/drive/folders/1hw6ufSxx1Pk-nUjWErdJXeAEVVwKIZKK?usp=sharing)

"place01.csv" contains the altitude data corresponding to the images.

"place01/images" contains the image files used for the analysis, which have been compressed into a ZIP archive for easier storage and transfer.

The code in this repository are provided for academic and research purposes.
