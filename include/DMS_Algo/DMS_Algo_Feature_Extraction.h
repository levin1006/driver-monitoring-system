#pragma once

#include "opencv2/highgui.hpp"
#include "DMS_Algo_HOG_OpenCV.h"

int FeatureExtraction_HOG_Descriptor( cv::Mat grayImage, HOGDescriptor &HOG, unsigned char *featVec );
