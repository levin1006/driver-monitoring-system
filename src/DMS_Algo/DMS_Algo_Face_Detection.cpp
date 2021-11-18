
#include "DMS_Algo_Global_Parameter.h"
#include "DMS_Algo_typedef.h"



static char buf_char[1024];

void Face_Detection_Preprocessing( sFaceDetector &faceDetector, cv::Mat grayImage )
{
	int width  = grayImage.cols;
	int height = grayImage.rows;

	//faceDetector.fFaceDetect = FALSE;

	/// Set ROI
	if( faceDetector.fFaceDetect == FALSE )
	{
		faceDetector.faceROI.x		 = 0;
		faceDetector.faceROI.y		 = 0;
		faceDetector.faceROI.width  = width;
		faceDetector.faceROI.height = height;

	}
	else
	{
		int delta = ( faceDetector.face.width 
				  * ( faceDetector.param.scaleFactor 
					* faceDetector.param.maxScaleMultiplier
					- 1 ) ) / 2;
		
		faceDetector.face = faceDetector.face + faceDetector.faceROI.tl();
		faceDetector.faceROI.x		= MAX( 0, faceDetector.face.x - delta );
		faceDetector.faceROI.y		= MAX( 0, faceDetector.face.y - delta );
		faceDetector.faceROI.width  = MIN( width - 1, faceDetector.faceROI.x + faceDetector.face.width + 2 * delta )
									- faceDetector.faceROI.x;
		faceDetector.faceROI.height = MIN( height - 1, faceDetector.faceROI.y + faceDetector.face.height + 2 * delta )
									- faceDetector.faceROI.y;
	}

	faceDetector.faceRoiImage = grayImage( faceDetector.faceROI );
}

unsigned char Face_Detection( sFaceDetector &faceDetector )
{
	sParam_faceDetector param;
	std::vector<cv::Rect> faces;
	cv::Size imgSize = faceDetector.faceRoiImage.size();

	//equalizeHist( DMS->grayImage, DMS->grayImage );

	if( faceDetector.fFaceDetect == FALSE )
		param = faceDetector.param_init;
	else
		param = faceDetector.param;

	std::vector<int> fakeLevels;
	std::vector<double> fakeWeights;
	std::vector<CvAvgComp> vecAvgComp;
	

	detectMultiScaleOldFormat( faceDetector.faceRoiImage, faceDetector.classifier, faces, 
		fakeLevels, fakeWeights, vecAvgComp, 
		param.scaleFactor, param.minNeighbors, param.flags, 
		cv::Size( imgSize.width  * param.minFaceRatio,
				imgSize.height * param.minFaceRatio ), 
		cv::Size( imgSize.width  * param.maxFaceRatio,
				imgSize.height * param.maxFaceRatio ) );


	if( faces.size() != 0 )
	{
		faceDetector.fFaceDetect = TRUE;

		faceDetector.face.x		 = faces[0].x;
		faceDetector.face.y		 = faces[0].y;
		faceDetector.face.width  = faces[0].width;
		faceDetector.face.height = faces[0].height;
	}
	else
	{
		faceDetector.fFaceDetect = FALSE;

		faceDetector.face.x		 = 0;
		faceDetector.face.y		 = 0;
		faceDetector.face.width  = 0;
		faceDetector.face.height = 0;
	}

	return faceDetector.fFaceDetect;
}

