#pragma once

#include "DMS_Algo_typedef.h"

void Eye_Detection_Preprocessing( sEyeDetector &eyeDetector, cv::Mat faceImage, unsigned char fFaceDetect );

void Iris_Detection( sEyeData &eyeData, sParam_eyeDetector param );

void Eye_Blink_Detection( sEyeData &eyeData, sHOG closedEye_HOG, sParam_eyeDetector param );

void Eye_Lid_Detection( sEyeData &eyeData, sParam_eyeDetector param );

void Eye_Gaze_Direction_Detection( sEyeData &eyeData, sParam_eyeDetector param );