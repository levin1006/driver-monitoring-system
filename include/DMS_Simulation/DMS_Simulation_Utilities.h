#pragma once

#define _STOPWATCH_STOP		0
#define _STOPWATCH_START	1

#include "opencv2\highgui.hpp"


void stopWatch( unsigned char fRun, unsigned char dispFPS = 0 );

std::chrono::steady_clock::time_point getTime_init();

float getTime_elapsed( std::chrono::steady_clock::time_point &ref );


void separate_dir_and_file( cv::String fullPath, cv::String &dir, cv::String &fileName );

cv::String get_fileName_from_fullPath( cv::String fullPath );

cv::String get_fileName_removing_extension( cv::String fileNameWithExt );


cv::Mat rotationMatrixToEulerAngles( cv::Mat &R );