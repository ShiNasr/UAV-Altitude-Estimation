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



"place01.csv" contains the altitude data corresponding to the images.

"palce01/images" contains the image files of the place01 subcategory in the UDWA dataset used for the analysis.

### Access the Dataset

The images and additional files (jason files) can be accessed via the following link:
The jason files have been converted into a single CSV (place0.csv) file for easier access and are shared in this repository
[Download the files](https://zenodo.org/records/5813232#.YdKDG2BByUk)

The code in this repository are provided for academic and research purposes.
