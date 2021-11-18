#include "DMS_Algo_Global_Parameter.h"
#include "DMS_Algo_typedef.h"


char *Get_knowledge_path( int opMode )
{
	char *path = "";

	if( opMode == _KNF_FACE )
	{
		//path = "additional/haarcascade_frontalface_alt.xml";
		path = "additional/haarcascade_frontalface_alt2.xml";
		
	}
	else if( opMode == _KNF_EYE_CLOSED )
	{
		path = "additional/20171219_ClosedEye_v1.knf";
	}
	else
	{
		printf("[ERROR] Cannot find training file");
		system("pause");
	}


	return path;
}


void Set_Parameter_FaceDetection( sParam_faceDetector &param, unsigned char fInit )
{ 
	if( fInit == TRUE )
	{
		param.scaleFactor = 1.1;
		param.minNeighbors = 3;
		//param.flags = cv::CASCADE_FIND_BIGGEST_OBJECT;
		param.minFaceRatio = 0.2;
		param.maxFaceRatio = 1;
	}
	else
	{
		param.scaleFactor = 1.1;
		param.minNeighbors = 3;
		//param.flags = cv::CASCADE_FIND_BIGGEST_OBJECT;
		param.minFaceRatio = 0.7;
		param.maxFaceRatio = 1.5;

		param.maxScaleMultiplier = 1.2;
	}
}


void Set_Parameter_EyeDetection( sParam_eyeDetector &param )
{
	param.eyeRoiVsFace.x	  = 3 / 16.0;
	param.eyeRoiVsFace.y	  = 4.5 / 16.0;
	param.eyeRoiVsFace.width  = 4 / 16.0;
	param.eyeRoiVsFace.height = 4 / 16.0;

	param.eyeRoiResize.width  = 40;
	param.eyeRoiResize.height = param.eyeRoiResize.width
							  / (float)param.eyeRoiVsFace.width
							  * (float)param.eyeRoiVsFace.height
							  + 0.5;


	/// Iris detection
	param.irisMinRadius = ( 1.0 / 10.0 ) * param.eyeRoiResize.width + 0.5;
	param.irisMaxRadius = ( 1.3 / 10.0 ) * param.eyeRoiResize.width + 0.5;


	/// Eye lid feature point detection
	param.minFeaturePoints = 5;
	param.nSamples = 31;		/// must be odd
	param.sampleInterval = 1;	/// irisMinRadius / 2;
	param.maxEdge = 150;
	param.maxBrightness = 80;
	float edgeWeight_normalized = 0.3;
	param.edgeWeight = edgeWeight_normalized / (float)param.maxEdge * 1000.0;
	param.darknessWeight = ( 1 - edgeWeight_normalized ) / (float)param.maxBrightness * 1000.0;

	param.threshold_eyelidResp = 0;


	/// Eye lid curve fitting
	param.eyelid_RANSAC_residual_threshold = 1.5;
	param.eyelid_RANSAC_nIteration = 15;


	/// Eye gaze direction detection
	param.threshold_eyelidConvexity = 0;
	param.threshold_irisLatOffset = param.irisMinRadius * 0.5;


	/// Eye blink detection
	param.closedEyeRegion.width  = 32;
	param.closedEyeRegion.height = 32;
	param.NM_closedEye_MAXIF = 5000;
	param.NM_closedEye_MINIF = 500;
}

void Set_Parameter_Closed_Eye_HOG( sHOG &closedEye_HOG )
{

	closedEye_HOG.winSize		= cv::Size( 32, 32 );
	closedEye_HOG.winStride		= cv::Size( 1, 1 );
	closedEye_HOG.cellSize		= cv::Size( 8, 8 );
	closedEye_HOG.blockSize		= cv::Size( 16, 16 );
	closedEye_HOG.blockStride	= cv::Size( 8, 8 );
	closedEye_HOG.nbins			= 7;
	closedEye_HOG.vecLen = ( ( closedEye_HOG.winSize.width - closedEye_HOG.blockSize.width ) / closedEye_HOG.blockStride.width + 1 )
						   * ( closedEye_HOG.blockSize.width / closedEye_HOG.cellSize.width )
						   * ( ( closedEye_HOG.winSize.height - closedEye_HOG.blockSize.height ) / closedEye_HOG.blockStride.height + 1 )
						   * ( closedEye_HOG.blockSize.height / closedEye_HOG.cellSize.height )
						   * closedEye_HOG.nbins;
	closedEye_HOG.signedGradient = FALSE;

	closedEye_HOG.HOG.winSize		= closedEye_HOG.winSize;
	closedEye_HOG.HOG.cellSize		= closedEye_HOG.cellSize;
	closedEye_HOG.HOG.blockSize		= closedEye_HOG.blockSize;
	closedEye_HOG.HOG.blockStride	= closedEye_HOG.blockStride;
	closedEye_HOG.HOG.nbins			= closedEye_HOG.nbins;
	closedEye_HOG.HOG.signedGradient = FALSE;
}