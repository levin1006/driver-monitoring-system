//! Test code for development environment set-up
// 2017.08.25 First implementation
///           Integrating OpenCV and GVAPI
// 2017.08.29 
///           Implementation of learning sequence
// 2017.08.30 
///           Features extraction with monochrome subsampling
///		      Distance map creation induced by recognization
// 2017.08.31
///           Implemention of object detection sequence usign bounding box
// 2017.09.01
///           Rebuilding code structure
///           - Extract header file for parameters
///           - Function packaging and separation
// 2017.09.04
///           Improving file read system: batch reading using cv::glob
// 2017.09.05
///           Improving training sequence: full training using cat0, cat1 until no changes
///           Implementation of neuron id map
///           Implementation of neuron vector map
// 2017.09.07
///			  Code conversion for using gradient feature
///           Image enhancement using linear trasnfer function
// 2017.09.08
///           Implementation of gradient feature for training
// 2017.09.11
///           Recognition test using gradient feature
///           Implementation of training validation
// 2017.09.12
///           Applying manual ROS
// 2017.09.14 Re-naming project as FR_EyeBlink
///           Data packaging
///           Function modulization
///           Displaying output text on both command window and text file
///           Caculating elapsed time using stop watch function
// 2017.09.15 Imrpoving input/output
///           Implementation of video output
///           Improving handling input dataset
// 2017.09.18 
///           Debugging output video to use play trackbar
///			  Implementation of training sequence for multi category
// 2017.09.19
///			  Packaging ROS recognition / firing point clustering function
///           Modifying detection algorithm to manage multi category
// 2017.09.21
///           Applying new dataset - MHE1709
///			  Modifying training/validation code for adjusting input dataset adaptively
//!           Modifying edge detection strategy (edge detection before subsampling --> subsampling before edge detection)
// 2017.09.22 Implementation of tracking algorithm: 1day
///			  Unifying outputs for bounding box, distance and firing points
///		      Modifying clustering algorithm from categorized/geometrical clustering to only geometrical clustering for categorizing after clustering
///			  Modifying cluster representative method from averaging min/max points to averaging all of the points
///           Implementation of tracking target creation and update
// 2017.09.25 Implementation of tracking algorithm: 2day
///           Modifying access method for tracking data as a liked list
///			  First draft of tracking algoritm, except for splitting, merging case
// 2017.09.26 Reducing calculation time
///			  Subsampling on preprocessing phase
///			  Packaging tracking algorithm
///           Module optimization
///			  Use face ROS and eye ROS
// 2017.09.27 Adding new dataset
///			  Training/testset of Sinwook and Habit
// 2017.10.13 Second phase after draft presentation
///			  Implementation of iris detection using isophote curvature (1)
// 2017.10.16 
///			  Implementation of iris detection using isophote curvature (2)
// 2017.10.17 Face detection
///			  Unifying test and validation operation
///			  Starting implementing face training module
// 2017.10.18 Face detection
///			  Implementation of face training module
///			  Mean-Std.dev. normalization of face image
///			  Calculate top-bottom haar like feature using integral image
// 2017.10.19 Face detection
///			  Haar-like feature extraction and vectorizing features
///			  Training and detection using Haar-like feature
// 2017.10.20 Face detection
///			  Draft of rough face detection in real time
// 2017.10.23 Face detection
///           Scalable face detection for ini		tialization
// 2017.10.24 Face detection
///           Scalable face detection for initialization: fix indexing point for every scales
///           Scalable face detection: coarse to fine approach to fit face region
// 2017.10.25 Face detection
///			  Improving scalable C2F face detection
///			  Modifying seed extraction method(last idx of min dist -> average of position)
// 2017.10.26 Face detection
///			  Improving C2F method by using multi-seed appraoch
///			  Reducing redundatnt calculation by using access map
///			  Improving scalable window by using reference size on initial step and tracking step
// 2017.10.30 Face detection
///			  Modifying parameter reference process
// 2017.10.31 Face detection
///			  Implementation of face validation
// 2017.11.01 Face detection
///			  Add more rectangle features
// 2017.11.02 Face detection
///			  Modifying Rectangle featuer window to 8x8 from 7x8
// 2017.11.07 Face detection
///			  Removing operating mode and make user chooses the knf file source
// 2017.11.08 Face detection
///			  Adding face training samples
///			  Applying mean normalization on roi window
// 2017.11.09 Face detection
///			  Implementing checking neuron id sequence
// 2017.11.10 Face detection
///			  Implementation of Adaboost Haar like cascade classifier
// 2017.11.13 Code Arrangement
///			  Code arrangement for eye detection
// 2017.11.14 Eye detection
///			  Iris detection using iris mask
// 2017.11.15 Eye detection
///			  Closed state detection using iris profiling
///			  Improving stop watch to respresent microsecond unit
// 2017.11.16 Eye detection
///			  Re-arranging required inputs of each of modules
///			  Closed eye detection using feature points from eye tail and lid
// 2017.11.17 Eye detection
///           Eye filtering from MSERs
///			  System state visualization by fonts
// 2017.11.20 Eye detection
///			  Iris profiling for closed state detection
// 2017.11.21 Eye detection
///			  Sclera detection for eye blink, gaze detection
// 2017.11.28 Eye detection
///			  Eye corner detection using HOG descriptor and NeuroMem
// 2017.11.29 Eye blink detection
///			  Arranging code
///           Release demo software draft
// 2017.12.08 Iris detection
///			  Implementation of CHT operator for iris detection
// 2017.12.11 Batch mode
///			  Implementation of batch operation mode for multi datasets
// 2017.12.12 Eye region analysis
///			  Implementation of IPF, VPF and HPF
// 2017.12.14 Eye region projection
///			  Detecting eye lid using gradient of HPFh
// 2017.12.15 Eye lid detection
///			  Feature points detection near iris
// 2017.12.18 Eye lid detection
///			  Eye lid curve fitting using RANSAC
// 2017.12.19 Eye blink detection
///			  Eye detection using upper eye lid model
///			  Eye blink detection using HOG
// 2017.12.20 Eye gaze detection
///			  Improving eye lid feature point detection using brightness term
///			  Eye gaze detection from upper eye lid curve
// 2018.01.04 Iris detection code modification: integer computation
// 2018.01.18 Debugging datast batch operation
///			  Debugging frame index handling
// 2018.01.23 Reducing calculation
///			  Reducing RANSAC iteration
// 2018.01.29 Install DLIB for facial landmark detection
// 2018.01.31 Head pose estimation tutorial
// 2018.02.02 Head pose estimation from facial landmark
// 2018.02.05 Relative head pose calculation from initial head pose
///			  Tracking initial head pose
// 2018.02.06 Creating portable project
// 2018.03.22 Team project upload
// 2018.03.23 Algorithm separation from test simulation environment
// 2018.03.26 File separation for algorithm and simulation environment


#include "DMS_Simulation_Bench.h"

void main()
{ 

	// Run
	DMS_Simulation_Bench();


}


