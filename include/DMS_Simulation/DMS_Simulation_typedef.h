#pragma once

#include <chrono>
#include "opencv2/highgui.hpp"

typedef struct sDataset
{
	std::string path;
	std::string name;
} sPathName;


typedef struct sOutput
{
	cv::Mat image_output;
	cv::VideoWriter *video_output;

	unsigned char fOutput_frameFPS;
	unsigned char fOutput_faceBox;
	unsigned char fOutput_eyeROI;
	unsigned char fOutput_iris;
	unsigned char fOutput_eyelid;
	unsigned char fOutput_eyeGaze;
	unsigned char fOutput_drowsyDetection;

} sOutput;


typedef struct sParam_inputFrame
{
	int inputSource;
	int dataset_firstFrame;
	int dataset_lastFrame;

	int output_video_FPS;

} sParam_inputFrame;

typedef struct sInputFrame
{
	sParam_inputFrame param;

	int datasetIdx;
	int frameIdx;
	int dataset_lastFrameIdx;


	std::vector<sDataset> datasetList;
	std::vector<cv::String> dataset;

	cv::VideoCapture input_CAM; // open the default camera

	cv::Mat srcImage;
	cv::Mat grayImage;

} sInputFrame;


typedef struct sTestEnvironment
{

	// System flow control
	sInputFrame inputFrame;


	std::chrono::steady_clock::time_point timeCapture;
	float timeElapsed;
	float FPS;


	// Output
	sOutput output;

} sTestEnvironment;

