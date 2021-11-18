
#include "DMS_Algo_Typedef.h"
#include "DMS_Algo_Global_Parameter.h"

#include "DMS_Algo_Face_Detection.h"
#include "DMS_Algo_Eye_Detection.h"
#include "DMS_Algo_Parameters.h"
#include "NM_Wrapper.h"


static void Transformation_Object_Coordinate_Local_to_Global( sDMS *DMS );

static void DMS_Reset( sDMS *DMS );


void DMS_Algorithm( sDMS *DMS, unsigned char *srcImage )
{
	// Reset DMS data structure
	DMS_Reset( DMS );


	// Image buffering
	DMS->inputFrame = cv::Mat::zeros( DMS->imgSize, CV_8U );
	memcpy( DMS->inputFrame.data, srcImage, DMS->imgSize.area() );

	


	// Face detection
	printf( "[PROC] Face detection - preprocessing\n" );
	Face_Detection_Preprocessing( DMS->faceDetector, DMS->inputFrame );

	printf( "[PROC] Face detection\n" );
	DMS->detectState = DMS->detectState & Face_Detection( DMS->faceDetector );
	



	// Eye detection
	if( DMS->faceDetector.fFaceDetect )
	{
		printf( "[PROC] Eye detection - preprocessing\n" );
		Eye_Detection_Preprocessing( DMS->eyeDetector, DMS->faceDetector.faceRoiImage( DMS->faceDetector.face ), DMS->faceDetector.fFaceDetect );

		

		for( int eye = 0 ; eye < 2 ; eye++ )
		{
			printf( "[PROC] Iris detection\n" );
			Iris_Detection( DMS->eyeDetector.eyeData[eye], DMS->eyeDetector.param );

			printf( "[PROC] Eye blink detection\n" );
			Eye_Blink_Detection( DMS->eyeDetector.eyeData[eye], DMS->eyeDetector.closedEye_HOG, DMS->eyeDetector.param );

			printf( "[PROC] Eye lid detection\n" );
			Eye_Lid_Detection( DMS->eyeDetector.eyeData[eye], DMS->eyeDetector.param );

			printf( "[PROC] Gaze direction detection\n" );
			Eye_Gaze_Direction_Detection( DMS->eyeDetector.eyeData[eye], DMS->eyeDetector.param );
		}
	}



	// Coordinate transformation: Local to Global
	Transformation_Object_Coordinate_Local_to_Global( DMS );


}



sDMS *DMS_Algorithm_Initialization()
{
	sDMS *DMS = new sDMS;


	// Set parameters
	Set_Parameter_FaceDetection( DMS->faceDetector.param_init, TRUE );
	Set_Parameter_FaceDetection( DMS->faceDetector.param, FALSE );
	Set_Parameter_EyeDetection( DMS->eyeDetector.param );
	Set_Parameter_Closed_Eye_HOG( DMS->eyeDetector.closedEye_HOG );



	// NeuroMem Knowledge
	NM_Initialization();

	NM_LoadKnowledge( Get_knowledge_path( _KNF_EYE_CLOSED ) );
	

	// Load Haar like cascade classifier
	//DMS->faceDetector.classifier.load( Get_knowledge_path( _KNF_FACE ) );
	DMS->faceDetector.classifier = load_haarcascade_frontalface_alt2();



	// Face detection initial state
	DMS->faceDetector.fFaceDetect = FALSE;

	
	// input image related
	DMS->imgSize = cv::Size( _INPUT_IMAGE_WIDTH, _INPUT_IMAGE_HEIGHT );


	return DMS;
}


void Transformation_Object_Coordinate_Local_to_Global( sDMS *DMS )
{
	/// Face ROI
	DMS->objects.faceROI.x	    = DMS->faceDetector.faceROI.x;
	DMS->objects.faceROI.y	    = DMS->faceDetector.faceROI.y;
	DMS->objects.faceROI.width  = DMS->faceDetector.faceROI.width;
	DMS->objects.faceROI.height = DMS->faceDetector.faceROI.height;

	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		/// Face
		DMS->objects.face = DMS->faceDetector.face + DMS->objects.faceROI.tl();


		for( int eye = 0 ; eye < 2 ; eye++ )
		{
			/// Eye ROI
			DMS->objects.eyeROI[eye] = DMS->eyeDetector.eyeData[eye].face2eyeROI + DMS->objects.face.tl();


			/// Iris
			DMS->objects.iris[eye] = DMS->eyeDetector.eyeData[eye].iris * DMS->eyeDetector.eyeResizeScale
				+ DMS->objects.eyeROI[eye].tl();

		}

	}




}


void DMS_Reset( sDMS *DMS )
{
	DMS->inputFrame.release();

	DMS->faceDetector.faceRoiImage.release();

	for( int eye = 0 ; eye < 2 ; eye++ )
	{
		DMS->eyeDetector.eyeData[eye].eyeROI.release();
		DMS->eyeDetector.eyeData[eye].eyeROI_intg.release();
		DMS->eyeDetector.eyeData[eye].irisRespImage.release();

		DMS->eyeDetector.eyeData[eye].eyelid_feat.clear();
		DMS->eyeDetector.eyeData[eye].eyelid_inlier.clear();

	}

}
