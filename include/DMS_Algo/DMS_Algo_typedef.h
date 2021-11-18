#pragma once

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "DMS_Algo_CvHaarCascade.h"
#include "DMS_Algo_HOG_OpenCV.h"


typedef struct sParam_faceDetector
{
	float scaleFactor;
	int minNeighbors;
	int flags;
	float minFaceRatio, maxFaceRatio;

	float maxScaleMultiplier;

} sParam_faceDetector;


typedef struct sFaceDetector
{
	sParam_faceDetector param_init;
	sParam_faceDetector param;

	cv::Rect faceROI;
	cv::Mat faceRoiImage;

	CvHaarClassifierCascade *classifier;

	cv::Rect face;

	unsigned char fFaceDetect;

} sFaceDetector;


typedef struct sHOG
{
	cv::Size winSize;
	cv::Size winStride;
	cv::Size cellSize;
	cv::Size blockSize;
	cv::Size blockStride;
	int nbins;
	int vecLen;
	unsigned char signedGradient;

	HOGDescriptor HOG;

} sHOG;


typedef struct sParam_eyeDetector
{
	cv::Rect2f eyeRoiVsFace;
	cv::Size eyeRoiResize;

	float irisMinRadius;
	float irisMaxRadius;


	/// eye lid feature point detection
	int minFeaturePoints;
	int nSamples;
	int sampleInterval;
	int edgeWeight;
	int darknessWeight;
	int maxBrightness;
	int maxEdge;
	float threshold_eyelidResp;


	/// Eye lid curve fitting
	int eyelid_RANSAC_residual_threshold;
	int eyelid_RANSAC_nIteration;


	/// Eye gaze direction detection
	float threshold_eyelidConvexity;
	int threshold_irisLatOffset;

	/// Eye blink detection
	cv::Size closedEyeRegion;
	int NM_closedEye_MAXIF;
	int NM_closedEye_MINIF;


} sParam_eyeDetector;


typedef struct sEyeData
{

	/// Iris detection
	cv::Rect face2eyeROI;
	cv::Mat eyeROI;
	cv::Mat eyeROI_intg;
	cv::Mat irisRespImage;

	cv::Point2i iris;
	int irisResp;

	/// iris CHT
	cv::Mat eyeGx;
	cv::Mat eyeGy;
	cv::Mat eyeEdge;
	cv::Mat eyeEdge_intg;



	/// Eyelid feature point
	unsigned char fEyeLidDetect;
	std::vector<cv::Point> eyelid_feat;
	std::vector<cv::Point> eyelid_inlier;
	cv::Mat eyelid_model;
	


	/// Eye gaze direction detection
	cv::Point eyeTopCenter;
	unsigned char eyeGazeDirection;


	/// Eye blink detection
	unsigned char fEyeState;
} sEyeData;


typedef struct sEyeDetector
{
	sParam_eyeDetector param;
	/// Data feeder
	cv::Mat faceImage;

	float eyeResizeScale;

	sEyeData eyeData[2];

	sHOG closedEye_HOG;
	

} sEyeDetector;


typedef struct sGlobalObject
{
	cv::Rect faceROI;
	cv::Rect face;

	cv::Rect eyeROI[2];
	cv::Point2i iris[2];
	cv::Point eyeOcclusion[2][2];

} sGlobalObject;


typedef struct sDMS
{

	// Face detection
	sFaceDetector faceDetector;


	// Eye detection
	sEyeDetector eyeDetector;

	// Object management
	sGlobalObject objects;

	// Image
	cv::Mat inputFrame;
	cv::Size imgSize;

	// State
	unsigned char detectState;


} sDMS;
