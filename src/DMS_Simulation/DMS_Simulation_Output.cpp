
#include <stdio.h>
#include <direct.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


#include "GVAPI.h"
#include "NM_Wrapper.h"

#include "DMS_Algo_typedef.h"
#include "DMS_Algo_Global_Parameter.h"

#include "DMS_Simulation_typedef.h"
#include "DMS_Simulation_Global_Parameter.h"
#include "DMS_Simulation_User_Parameter.h"
#include "DMS_Simulation_Dataset_Definition.h"
#include "DMS_Simulation_Utilities.h"

#include "definition_macro.h"

#include <Windows.h>

#define _OUTPUT_IMAGE_RESIZE_RATIO 2
#define _OUTPUT_GAZE_TEXT 0


FILE *logFile;
static char buf_char_msg[1024];
static char buf_char_path[256];


// Temporal for application
static cv::Mat eye_blink_graph;
static int eye_blink_score = 0;


static char output_path[_MAX_PATH];

SYSTEMTIME sysTime;


void Output_Video_Write( cv::VideoWriter *srcVideo, cv::Mat srcImage );



void Set_output_path( std::string folderName )
{
	
	_mkdir( "output" );
	sprintf_s( output_path, "output/%s", _OUTPUT_FOLDER_NAME );
	_mkdir( output_path );
	sprintf_s( output_path, "%s/%s", output_path, folderName.c_str() );
	_mkdir( output_path );
}

void disp( char *message )
{
#if _OUTPUT_DISP_LOG
	printf( message );
#endif
#if _OUTPUT_SAVE_LOG
	sprintf_s( buf_char_path, "%s/log.txt", output_path );
	fopen_s( &logFile, buf_char_path, "a" );
	fprintf( logFile, message );
	fclose( logFile );
#endif
}


static unsigned char onFuncBox[6];

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	sTestEnvironment *testEnv = (sTestEnvironment *)userdata;

	if  ( event == cv::EVENT_LBUTTONDOWN )
	{
		if( cv::Rect( 50, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
			testEnv->output.fOutput_faceBox ^= 1;
		else if( cv::Rect( 250, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
			testEnv->output.fOutput_eyeROI ^= 1;
		else if( cv::Rect( 450, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
			testEnv->output.fOutput_iris ^= 1;
		else if( cv::Rect( 650, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
			testEnv->output.fOutput_eyelid ^= 1;
		else if( cv::Rect( 850, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
			testEnv->output.fOutput_drowsyDetection ^= 1;
		//else if( cv::Rect( 1050, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		//	testEnv->output.fOutput_faceBox ^= 1;


	}
	else if ( event == cv::EVENT_MOUSEMOVE )
	{

		if( cv::Rect( 50, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 1; onFuncBox[1] = 0; onFuncBox[2] = 0; onFuncBox[3] = 0; onFuncBox[4] = 0; onFuncBox[5] = 0;
		}
		else if( cv::Rect( 250, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 0; onFuncBox[1] = 1; onFuncBox[2] = 0; onFuncBox[3] = 0; onFuncBox[4] = 0; onFuncBox[5] = 0;
		}
		else if( cv::Rect( 450, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 0; onFuncBox[1] = 0; onFuncBox[2] = 1; onFuncBox[3] = 0; onFuncBox[4] = 0; onFuncBox[5] = 0;
		}
		else if( cv::Rect( 650, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 0; onFuncBox[1] = 0; onFuncBox[2] = 0; onFuncBox[3] = 1; onFuncBox[4] = 0; onFuncBox[5] = 0;
		}
		else if( cv::Rect( 850, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 0; onFuncBox[1] = 0; onFuncBox[2] = 0; onFuncBox[3] = 0; onFuncBox[4] = 1; onFuncBox[5] = 0;
		}
		else if( cv::Rect( 1050, 865, 150, 50 ).contains( cv::Point( x, y ) ) )
		{
			onFuncBox[0] = 0; onFuncBox[1] = 0; onFuncBox[2] = 0; onFuncBox[3] = 0; onFuncBox[4] = 0; onFuncBox[5] = 1;
		}
		else
		{
			onFuncBox[0] = 0; onFuncBox[1] = 0; onFuncBox[2] = 0; onFuncBox[3] = 0; onFuncBox[4] = 0; onFuncBox[5] = 0;
		}

	}
}


void output( sTestEnvironment &testEnv, sDMS *DMS )
{

	// Upsacle output
	cv::Mat outputImage;
	resize( testEnv.inputFrame.srcImage, outputImage, testEnv.inputFrame.srcImage.size() * _OUTPUT_IMAGE_RESIZE_RATIO );
	cv::String winName = "MHE Driver Monitoring System";



	if( testEnv.inputFrame.frameIdx == testEnv.inputFrame.param.dataset_firstFrame )
	{
		cv::namedWindow("MHE Driver Monitoring System");
		cv::setMouseCallback( winName, CallBackFunc, &testEnv );
	}


	// Text
	cv::Vec3b text_foreground_color( 255, 255, 255 );
	cv::Vec3b text_background_color( 100, 80, 0 );
	cv::Vec3b text_foreground_emphasize_color( 200, 150, 100 );
	cv::Vec3b text_on_color( 255, 255, 255 );
	cv::Vec3b text_off_color( 100, 100, 100 );
	int fontType = cv::FONT_HERSHEY_DUPLEX;
	int text_foreground_size = 2;
	int text_background_size = 5;
	int text_background_emphasize_size = 10;
	int text_column = 30;
	int text_row1 = 100;
	int text_row2 = 200;
	int text_row3 = 250;
	int text_row4 = 300;
	int text_row_func = 900;
	int func_column1 = 50;
	int func_column2 = 250;
	int func_column3 = 450;
	int func_column4 = 650;
	int func_column5 = 850;
	int func_column6 = 1050;





	// Output function box
	for( int idx = 0 ; idx < 6 ; idx++ )
	{
		if( onFuncBox[ idx ] == 1 )
			cv::rectangle( outputImage, cv::Rect( 50 + idx * 200, text_row_func - 35, 150, 50 ), cv::Vec3b( 255, 255, 255 ), 3 );
		else
			cv::rectangle( outputImage, cv::Rect( 50 + idx * 200, text_row_func - 35, 150, 50 ), cv::Vec3b( 255, 255, 255 ) );
	}

	/// Face box
	sprintf_s( buf_char_msg, "FACE" );
	if( onFuncBox[0] == 1 )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_background_color, text_background_emphasize_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_foreground_emphasize_color, text_foreground_size );
	}
	else
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_off_color, text_foreground_size );
	}

	if( testEnv.output.fOutput_faceBox == TRUE )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column1 + 35, text_row_func ), fontType, 1, text_on_color, text_foreground_size );
	}


	/// Eye ROI
	sprintf_s( buf_char_msg, "EYE ROI" );
	if( onFuncBox[1] == 1 )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_background_color, text_background_emphasize_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_foreground_emphasize_color, text_foreground_size );
	}
	else
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_off_color, text_foreground_size );
	}

	if( testEnv.output.fOutput_eyeROI == TRUE )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column2 + 15, text_row_func ), fontType, 1, text_on_color, text_foreground_size );
	}



	/// Iris
	sprintf_s( buf_char_msg, "IRIS" );
	if( onFuncBox[2] == 1 )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_background_color, text_background_emphasize_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_foreground_emphasize_color, text_foreground_size );
	}
	else
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_off_color, text_foreground_size );
	}

	if( testEnv.output.fOutput_iris == TRUE )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column3 + 45, text_row_func ), fontType, 1, text_on_color, text_foreground_size );
	}


	/// Eyelid
	sprintf_s( buf_char_msg, "EYELID" );
	if( onFuncBox[3] == 1 )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_background_color, text_background_emphasize_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_foreground_emphasize_color, text_foreground_size );
	}
	else
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_off_color, text_foreground_size );
	}

	if( testEnv.output.fOutput_eyelid == TRUE )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column4 + 25, text_row_func ), fontType, 1, text_on_color, text_foreground_size );
	}


	/// Application: drowsiness detection
	sprintf_s( buf_char_msg, "DROWSY" );
	if( onFuncBox[4] == 1 )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_background_color, text_background_emphasize_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_foreground_emphasize_color, text_foreground_size );
	}
	else
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_off_color, text_foreground_size );
	}

	if( testEnv.output.fOutput_drowsyDetection == TRUE )
	{
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( func_column5 + 10, text_row_func ), fontType, 1, text_on_color, text_foreground_size );
	}


	if( DMS->faceDetector.fFaceDetect == TRUE )
	{

		if( testEnv.output.fOutput_faceBox )
		{
			/// Face bounding box
			cv::rectangle( outputImage, cv::Rect( DMS->objects.face.x * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.face.y * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.face.width * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.face.height * _OUTPUT_IMAGE_RESIZE_RATIO ), cv::Scalar( 0, 0, 255 ), 3 );
		}

		if( testEnv.output.fOutput_eyeROI )
		{
			/// Eye ROI Bounding box
			if( DMS->faceDetector.fFaceDetect == TRUE )
			{
				for( int eye = 0 ; eye < 2 ; eye++ )
				{
					cv::rectangle( outputImage, cv::Rect( DMS->objects.eyeROI[eye].x * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].y * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].width * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].height * _OUTPUT_IMAGE_RESIZE_RATIO ), cv::Scalar( 0, 120, 230 ), 3 );
				}
			}
		}


		if( testEnv.output.fOutput_iris )
		{
			/// Iris visualization
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				cv::Scalar val = ( DMS->eyeDetector.eyeData[eye].fEyeState == _EYE_OPENED ) ? cv::Scalar( 0, 255, 0 ) : cv::Scalar( 0, 0, 255 );

				circle( outputImage, DMS->objects.iris[eye] * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->eyeDetector.param.irisMinRadius * _OUTPUT_IMAGE_RESIZE_RATIO, val, 2 );
			}

		}


		if( testEnv.output.fOutput_eyelid )
		{
			/// Eyelid feature point
			float resizeScale = DMS->eyeDetector.eyeResizeScale;
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( DMS->eyeDetector.eyeData[eye].fEyeLidDetect == TRUE )
				{
					//if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection != _EYE_GAZE_NONE )
					{

						//for( int idx = 0 ; idx < DMS->eyeDetector.eyeData[eye].eyelid_feat.size() ; idx++ )
						//{
						//	cv::Point featPt = DMS->eyeDetector.eyeData[eye].eyelid_feat[idx];
						//	featPt *= resizeScale;
						//	featPt *= _OUTPUT_IMAGE_RESIZE_RATIO;
						//	circle( outputImage( cv::Rect( DMS->objects.eyeROI[eye].x * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].y * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].width * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].height * _OUTPUT_IMAGE_RESIZE_RATIO ) ), featPt, 2, cv::Scalar( 200, 0, 0 ), -1 );
						//	//testEnv.output.detectImage( DMS->objects.eyeROI[eye] ).at<Vec3b>( featPt ) = Vec3b( 0, 200, 200 );
						//}

						for( int idx = 0 ; idx < DMS->eyeDetector.eyeData[eye].eyelid_inlier.size() ; idx++ )
						{
							cv::Point featPt = DMS->eyeDetector.eyeData[eye].eyelid_inlier[idx];
							featPt *= resizeScale;
							featPt *= _OUTPUT_IMAGE_RESIZE_RATIO;
							circle( outputImage( cv::Rect( DMS->objects.eyeROI[eye].x * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].y * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].width * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].height * _OUTPUT_IMAGE_RESIZE_RATIO ) ), featPt, 2, cv::Scalar( 0, 200, 200 ), -1 );
							//testEnv.output.detectImage( DMS->objects.eyeROI[eye] ).at<Vec3b>( featPt ) = Vec3b( 0, 200, 200 );
						}
					}
				}
			}

			/// Eyelid quadratic regression
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( DMS->eyeDetector.eyeData[eye].fEyeLidDetect == TRUE )
				{
					//if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection != _EYE_GAZE_NONE )
					{
						int minX = DMS->eyeDetector.eyeData[eye].eyelid_inlier[0].x;
						int maxX = DMS->eyeDetector.eyeData[eye].eyelid_inlier[ DMS->eyeDetector.eyeData[eye].eyelid_inlier.size() - 1 ].x;
						float resizeScale = DMS->eyeDetector.eyeResizeScale;
						int iy = DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(0) * sq_( minX )
							+ DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(1) * minX
							+ DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(2);
						cv::Point pt0, pt1( minX, iy );
						pt1 *= resizeScale;
						pt1 *= _OUTPUT_IMAGE_RESIZE_RATIO;
						for( int ix = minX + 1 ; ix <= maxX ; ix++ )
						{
							pt0 = pt1;
							iy = DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(0) * sq_( ix )
								+ DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(1) * ix
								+ DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(2);
							pt1 = cv::Point( ix, iy );
							pt1 *= resizeScale;
							pt1 *= _OUTPUT_IMAGE_RESIZE_RATIO;

							if( isinside_( iy, 0, DMS->eyeDetector.eyeData[eye].eyeROI.rows ) )
								line( outputImage( cv::Rect( DMS->objects.eyeROI[eye].x * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].y * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].width * _OUTPUT_IMAGE_RESIZE_RATIO, DMS->objects.eyeROI[eye].height * _OUTPUT_IMAGE_RESIZE_RATIO ) ), pt0, pt1, cv::Scalar( 0, 0, 255 ) );


						}
					
					}
				}
			}
		}

	}



	/// Frame per second
	sprintf_s( buf_char_msg, "FPS: %.2f", testEnv.FPS );
	putText( outputImage, buf_char_msg, cv::Point( text_column, text_row1 ), fontType, 1, text_background_color, text_background_size );
	putText( outputImage, buf_char_msg, cv::Point( text_column, text_row1 ), fontType, 1, text_foreground_color, text_foreground_size );

	/// Front head pose detection

	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		sprintf_s( buf_char_msg, "HEAD POSE: FRONT" );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row2 ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row2 ), fontType, 1, text_on_color, text_foreground_size );
	}
	else
	{
		sprintf_s( buf_char_msg, "HEAD POSE: NOT FRONT" );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row2 ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row2 ), fontType, 1, text_on_color, text_foreground_size );

	}

	/// Eye blink
	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		if( ( DMS->eyeDetector.eyeData[0].fEyeState == _EYE_CLOSED ) && ( DMS->eyeDetector.eyeData[1].fEyeState == _EYE_CLOSED ) )
		{
			sprintf_s( buf_char_msg, "EYE STATE: CLOSED" );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_background_color, text_background_size );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_on_color, text_foreground_size );
		}
		else
		{
			sprintf_s( buf_char_msg, "EYE STATE: OPEN" );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_background_color, text_background_size );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_on_color, text_foreground_size );
		}

	}
	else
	{
		sprintf_s( buf_char_msg, "EYE STATE" );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row3 ), fontType, 1, text_off_color, text_foreground_size );
	}


	/// Eye gaze direction
#if _OUTPUT_GAZE_TEXT
	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		if( ( DMS->eyeDetector.eyeData[0].eyeGazeDirection == _EYE_GAZE_LEFT ) && ( DMS->eyeDetector.eyeData[1].eyeGazeDirection == _EYE_GAZE_LEFT ) )
		{
			sprintf_s( buf_char_msg, "GAZE DIRECTION: LEFT" );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_background_color, text_background_size );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_on_color, text_foreground_size );
		}
		else if( ( DMS->eyeDetector.eyeData[0].eyeGazeDirection == _EYE_GAZE_RIGHT ) && ( DMS->eyeDetector.eyeData[1].eyeGazeDirection == _EYE_GAZE_RIGHT ) )
		{
			sprintf_s( buf_char_msg, "GAZE DIRECTION: RIGHT" );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_background_color, text_background_size );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_on_color, text_foreground_size );
		}
		else
		{
			sprintf_s( buf_char_msg, "GAZE DIRECTION: FRONT" );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_background_color, text_background_size );
			putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_on_color, text_foreground_size );
		}
	}
	else
	{
		sprintf_s( buf_char_msg, "GAZE DIRECTION" );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_background_color, text_background_size );
		putText( outputImage, buf_char_msg, cv::Point( text_column, text_row4 ), fontType, 1, text_off_color, text_foreground_size );
	}
#endif



	/// Application: drowsy detection

	int prevBlinkScore = eye_blink_score;
	cv::Size blinkGraph( 300, 200 );
	int eye_blink_threshold = 70;

	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		/// Eye blink
		if( DMS->faceDetector.fFaceDetect == TRUE )
		{
			if( ( DMS->eyeDetector.eyeData[0].fEyeState == _EYE_CLOSED ) && ( DMS->eyeDetector.eyeData[1].fEyeState == _EYE_CLOSED ) )
			{
				eye_blink_score = MIN( ( eye_blink_score + 1 ) * 1.2, 99 );

			}
			else if( ( DMS->eyeDetector.eyeData[0].fEyeState == _EYE_OPENED ) && ( DMS->eyeDetector.eyeData[1].fEyeState == _EYE_OPENED ) )
			{
				eye_blink_score = MAX( ( eye_blink_score - 1 ) * 0.9, 0 );
			}
		}
	}




	if( testEnv.inputFrame.frameIdx % blinkGraph.width == 0 )
	{
		eye_blink_graph = cv::Mat::zeros( blinkGraph, CV_8UC3 );
		cv::line( eye_blink_graph, cv::Point( 0, blinkGraph.height - eye_blink_threshold * 2 ), 
			cv::Point( blinkGraph.width - 1, blinkGraph.height - eye_blink_threshold * 2 ), cv::Vec3b( 0, 255, 255 ), 1 );

	}
	else
	{
		cv::line( eye_blink_graph, cv::Point( ( testEnv.inputFrame.frameIdx - 1 ) % blinkGraph.width, blinkGraph.height - prevBlinkScore  * 2 ), 
			cv::Point( ( testEnv.inputFrame.frameIdx ) % blinkGraph.width, blinkGraph.height - eye_blink_score * 2 ), cv::Vec3b( 0, 0, 255 ), 3 );
	}


	if( testEnv.output.fOutput_drowsyDetection )
	{
		eye_blink_graph.copyTo( outputImage( cv::Rect( 900, 100, 300, 200 ) ) );



		if( eye_blink_score > eye_blink_threshold )
		{
			cv::rectangle( outputImage, cv::Rect( 900, 100, 300, 200 ), cv::Vec3b( 0, 0, 150 ), 3 );
			sprintf_s( buf_char_msg, "WARNING: DROWSY" );
			putText( outputImage, buf_char_msg, cv::Point( 900, 100 ), fontType, 1, cv::Vec3b(0, 255, 255 ), 3 );
		}
		else
			cv::rectangle( outputImage, cv::Rect( 897, 97, 306, 206 ), cv::Vec3b( 0, 0, 0 ), 3 );
	}







#if _OUTPUT_SHOW_WINDOW_DETECTION_RESULTS
	imshow( winName, outputImage );
	cv::waitKey(1);
#endif

#if _OUTPUT_SAVE_IMAGE_DETECTION_RESULTS || _OUTPUT_SAVE_VIDEO_DETECTION_RESULTS
	cv::Mat imBufForWrite;
	resize( outputImage, imBufForWrite, testEnv.inputFrame.srcImage.size() );
#endif

#if _OUTPUT_SAVE_IMAGE_DETECTION_RESULTS
	GetSystemTime(&sysTime);
	sprintf_s( buf_char_msg, "%s/realtime_%d%02d%02d.%02d%02d%02d.%03d.png", output_path, sysTime.wYear, sysTime.wMonth, sysTime.wDay, ( sysTime.wHour + 9 ) % 24, sysTime.wMinute, sysTime.wSecond, sysTime.wMilliseconds );
	cv::imwrite( buf_char_msg, imBufForWrite );
#endif

#if _OUTPUT_SAVE_VIDEO_DETECTION_RESULTS
	Output_Video_Write( testEnv.output.video_output, imBufForWrite );
#endif




#if _OUTPUT_SAVE_IMAGE_PATCH_AROUND_IRIS
	/// iris patch extraction
	for( int eye = 0 ; eye < 2 ; eye++ )
	{

		int ix = DMS->objects.iris[eye].x - DMS->eyeDetector.closedEye_HOG.winSize.width / 2;
		int iy = DMS->objects.iris[eye].y - DMS->eyeDetector.closedEye_HOG.winSize.height / 2;

		cv::Mat cropImage = DMS->grayImage( Rect( ix, iy, DMS->eyeDetector.closedEye_HOG.winSize.width, DMS->eyeDetector.closedEye_HOG.winSize.height ) );

		sprintf_s( buf_char_msg, "%s/iris_%03d_%d.png", output_path, DMS->sysFlowControl.frameIdx, eye );
		imwrite( buf_char_msg, cropImage );
	}
#endif






	sprintf_s( buf_char_msg, "[SYSTEM] Overall processing time: %.3f, FPS: %.2f\n", testEnv.timeElapsed, testEnv.FPS );
	disp( buf_char_msg );













	// backup
#if 0

	cv::namedWindow("DMS");


	float fontSize = 0.75;
	int sysState_default_x = 15;
	int second_x = 170;
	int frame_y = 30;
	int faceState_y = 60;
	int irisState_y = 90;
	int eyeState_y = 120;
	cv::Scalar sysState_pass( 200, 100, 0 );
	cv::Scalar sysState_default( 50, 50, 50 );
	
	DMS->output.detectImage = DMS->srcImage.clone();


	int key = cv::waitKey(1);
	switch( key )
	{
		case '0':
		{
			DMS->output.fOutput_frameFPS ^= 1;
			break;
		}
		case '1':
		{
			DMS->output.fOutput_faceBox ^= 1;
			break;
		}
		case '2':
		{
			DMS->output.fOutput_eyeROI ^= 1;
			break;
		}
		case '3':
		{
			DMS->output.fOutput_iris ^= 1;
			break;
		}
		case '4':
		{
			DMS->output.fOutput_eyelid ^= 1;
			break;
		}
		case '5':
		{
			DMS->output.fOutput_eyeGaze ^= 1;
			break;
		}
	}


	if( DMS->output.fOutput_frameFPS )
	{
#if 1
		/// Frame number
		sprintf_s( buf_char_msg, "Frame#%d", DMS->sysFlowControl.frameIdx );
		putText( DMS->output.detectImage, buf_char_msg, cv::Point( sysState_default_x, frame_y ), cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar( 0, 50, 150 ) );
#endif


#if 1
		/// Frame per second
		sprintf_s( buf_char_msg, "FPS: %.2f", DMS->sysFlowControl.FPS );
		putText( DMS->output.detectImage, buf_char_msg, cv::Point( second_x, frame_y ), cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar( 0, 50, 150 ) );
#endif
	}


#if 0
	/// Face ROI
	cv::Size imgSize = DMS->output.detectImage.size();
	float ROS_gradFactor = 1.2;
	for( int iy = 0 ; iy < imgSize.height ; iy++ )
	{
		for( int ix = 0 ; ix < imgSize.width ; ix++ )
		{
			if( ( ix < DMS->objects.faceROI.x ) 
				|| ( ix >= DMS->objects.faceROI.x + DMS->objects.faceROI.width ) 
				|| ( iy < DMS->objects.faceROI.y ) 
				|| ( iy >= DMS->objects.faceROI.y + DMS->objects.faceROI.height ) )
			{
				DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[0] = DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[0] / ROS_gradFactor;
				DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[1] = DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[1] / ROS_gradFactor;
				DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] = DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] / ROS_gradFactor;
			}
		}
	}
#endif

	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		
		if( DMS->output.fOutput_faceBox )
		{
#if 1
			/// Face bounding box
			cv::rectangle( DMS->output.detectImage, DMS->objects.face, cv::Scalar( 0, 0, 255 ), 1 );
#endif
		}


#if 0
		/// Face grid
		float nGridX = 16;
		float nGridY = 16;

		for( int gy = 0 ; gy < nGridY ; gy++ )
		{
			for( int ix = DMS->objects.face.x ; ix < DMS->objects.face.x + DMS->objects.face.width ; ix++ )
			{
				int iy = DMS->objects.face.y + gy / nGridY * DMS->objects.face.height;
				DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] = MIN( 255, DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] * 1.5 );
			}
		}

		for( int gx = 0 ; gx < nGridX ; gx++ )
		{
			for( int iy = DMS->objects.face.y ; iy < DMS->objects.face.y + DMS->objects.face.height ; iy++ )
			{
				int ix = DMS->objects.face.x + gx / nGridX * DMS->objects.face.width;
				DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] = MIN( 255, DMS->output.detectImage.at<cv::Vec3b>(iy, ix)[2] * 1.5 );
			}
		}
#endif

		
		if( DMS->output.fOutput_eyeROI )
		{
#if 1
			/// Eye ROI Bounding box
			if( DMS->faceDetector.fFaceDetect == TRUE )
			{
				for( int eye = 0 ; eye < 2 ; eye++ )
				{
					cv::rectangle( DMS->output.detectImage, DMS->objects.eyeROI[eye], cv::Scalar( 0, 120, 230 ), 1 );
				}
			}
#endif
		}

		



		if( DMS->output.fOutput_iris )
		{
#if 1
			/// Iris visualization: Transparent
			cv::Mat overlay;
			float alpha = 0.3;
			DMS->output.detectImage.copyTo( overlay );
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				cv::Scalar val = ( DMS->eyeDetector.eyeData[eye].fEyeState == _EYE_OPENED ) ? cv::Scalar( 0, 255, 0 ) : cv::Scalar( 0, 0, 255 );

				circle( overlay, DMS->objects.iris[eye], DMS->eyeDetector.param.irisMinRadius * DMS->eyeDetector.eyeResizeScale, val, -1 );
				circle( DMS->output.detectImage, DMS->objects.iris[eye], DMS->eyeDetector.param.irisMinRadius * DMS->eyeDetector.eyeResizeScale, val, 1 );
			}
			cv::addWeighted( overlay, alpha, DMS->output.detectImage, 1 - alpha, 0, DMS->output.detectImage );
			overlay.release();
#else
			/// Iris visualization
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				cv::Scalar val = ( DMS->eyeDetector.eyeData[eye].fEyeState == _EYE_OPENED ) ? cv::Scalar( 0, 255, 0 ) : cv::Scalar( 0, 0, 255 );

				circle( DMS->output.detectImage, DMS->objects.iris[eye], DMS->eyeDetector.param.irisMinRadius, val, 2 );
			}

#endif
		}


		if( DMS->output.fOutput_eyelid )
		{
#if 1
			/// Eyelid feature point
			float resizeScale = DMS->eyeDetector.eyeResizeScale;
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection != _EYE_GAZE_NONE )
				{

					for( int idx = 0 ; idx < DMS->eyeDetector.eyeData[eye].eyelid_feat.size() ; idx++ )
					{
						cv::Point featPt = DMS->eyeDetector.eyeData[eye].eyelid_feat[idx];
						featPt *= resizeScale;
						circle( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), featPt, 1, cv::Scalar( 200, 0, 0 ), -1 );
						//DMS->output.detectImage( DMS->objects.eyeROI[eye] ).at<Vec3b>( featPt ) = Vec3b( 0, 200, 200 );
					}

					for( int idx = 0 ; idx < DMS->eyeDetector.eyeData[eye].eyelid_inlier.size() ; idx++ )
					{
						cv::Point featPt = DMS->eyeDetector.eyeData[eye].eyelid_inlier[idx];
						featPt *= resizeScale;
						circle( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), featPt, 1, cv::Scalar( 0, 200, 200 ), -1 );
						//DMS->output.detectImage( DMS->objects.eyeROI[eye] ).at<Vec3b>( featPt ) = Vec3b( 0, 200, 200 );
					}
				}
			}
#endif

#if 1
			/// Eyelid quadratic regression
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection != _EYE_GAZE_NONE )
				{
					int minX = DMS->eyeDetector.eyeData[eye].eyelid_inlier[0].x;
					int maxX = DMS->eyeDetector.eyeData[eye].eyelid_inlier[ DMS->eyeDetector.eyeData[eye].eyelid_inlier.size() - 1 ].x;
					float resizeScale = DMS->eyeDetector.eyeResizeScale;
					int iy = DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(0) * sq_( minX )
						   + DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(1) * minX
						   + DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(2);
					cv::Point pt0, pt1( minX, iy );
					pt1 *= resizeScale;
					for( int ix = minX + 1 ; ix <= maxX ; ix++ )
					{
						pt0 = pt1;
						iy = DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(0) * sq_( ix )
						   + DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(1) * ix
						   + DMS->eyeDetector.eyeData[eye].eyelid_model.at<float>(2);
						pt1 = cv::Point( ix, iy );
						pt1 *= resizeScale;

						if( isinside_( iy, 0, DMS->eyeDetector.eyeData[eye].eyeROI.rows ) )
							line( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), pt0, pt1, cv::Scalar( 0, 0, 255 ) );

				
					}
				}
			}
#endif


#if 1
			/// Eye center
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection != _EYE_GAZE_NONE )
				{
					float resizeScale = DMS->eyeDetector.eyeResizeScale;
					cv::Point pt0 = DMS->eyeDetector.eyeData[eye].eyeTopCenter;
					cv::Point pt1( pt0.x, DMS->eyeDetector.eyeData[eye].iris.y );
					pt0 *= resizeScale;
					pt1 *= resizeScale;

					line( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), pt0, pt1, cv::Scalar( 0, 0, 255 ) );
				}
			}
#endif

		}


		if( DMS->output.fOutput_eyeGaze )
		{
#if 1
			/// Eye Gaze Direction
			for( int eye = 0 ; eye < 2 ; eye++ )
			{
				if( ( DMS->eyeDetector.eyeData[eye].eyeGazeDirection == _EYE_GAZE_LEFT ) || ( DMS->eyeDetector.eyeData[eye].eyeGazeDirection == _EYE_GAZE_RIGHT ) )
				{
					float resizeScale = DMS->eyeDetector.eyeResizeScale;
					int width  = DMS->eyeDetector.eyeData[eye].eyeROI.cols;
					int height = DMS->eyeDetector.eyeData[eye].eyeROI.rows;

					cv::Point pt0, pt1, pt2, pt3;
					pt0 = cv::Point( width / 2, 0 );
					pt3 = cv::Point( width / 2, height - 1 );
					if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection == _EYE_GAZE_LEFT )
					{
						pt1 = cv::Point( 0, 0 );
						pt2 = cv::Point( 0, height - 1 );
					}
					else if( DMS->eyeDetector.eyeData[eye].eyeGazeDirection == _EYE_GAZE_RIGHT )
					{
						pt1 = cv::Point( width - 1, 0 );
						pt2 = cv::Point( width - 1, height - 1 );

					}

					pt0 *= resizeScale;
					pt1 *= resizeScale;
					pt2 *= resizeScale;
					pt3 *= resizeScale;

					line( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), pt0, pt1, cv::Scalar( 0, 0, 255 ), 5 );
					line( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), pt1, pt2, cv::Scalar( 0, 0, 255 ), 5 );
					line( DMS->output.detectImage( DMS->objects.eyeROI[eye] ), pt2, pt3, cv::Scalar( 0, 0, 255 ), 5 );
				}
			}
#endif
		}

		






	}
	else
	{
#if 1
		/// Distracted behavior warning
		if( DMS->faceDetector.distractCnt > DMS->faceDetector.param.Criteria_distract_time_sec * 30 )
		{
			int mode = ( ( DMS->sysFlowControl.frameIdx % 30 ) < 15 ) ? 0 : 1;
			cv::Scalar boxColor = ( mode == 0 ) ? cv::Scalar( 0, 0, 255 ) : cv::Scalar( 0, 255, 255 );

			cv::rectangle( DMS->output.detectImage, cv::Rect( 0, 0, DMS->imgSize.width, DMS->imgSize.height ), boxColor, 10 ); 

			sprintf_s( buf_char_msg, "Please attention!!" );
			putText( DMS->output.detectImage, buf_char_msg, cv::Point( 50, 100 ), cv::FONT_HERSHEY_DUPLEX, 2, cv::Scalar( 0, 50, 150 ), 2 );
		}
#endif

	}





	// Backup
#if 0


	// System state visualization using text
#if 0
	/// Face detection state
	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		disp( "[SYSTEM] Face: PASS\n" );
		putText( DMS->output.detectImage, "FACE", Point( sysState_default_x, faceState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_pass );
	}
	else
	{
		disp( "[SYSTEM] Face: FAIL\n" );
		putText( DMS->output.detectImage, "FACE", Point( sysState_default_x, faceState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	}
#endif


#if 0
	if( DMS->eyeDetector.fIrisDetect[0] == TRUE )
	{
		disp( "[SYSTEM] Left iris: PASS\n" );
		putText( DMS->output.detectImage, "LEFT IRIS", Point( sysState_default_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_pass );

		if( DMS->eyeDetector.fEyeDetect[0] == TRUE )
		{
			disp( "[SYSTEM] Left eye: PASS\n" );
			putText( DMS->output.detectImage, "LEFT EYE", Point( sysState_default_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_pass );
		}
		else
		{
			disp( "[SYSTEM] Left eye: FAIL\n" );
			putText( DMS->output.detectImage, "LEFT EYE", Point( sysState_default_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
		}

	}
	else
	{
		disp( "[SYSTEM] Left iris: FAIL\n" );
		putText( DMS->output.detectImage, "LEFT IRIS", Point( sysState_default_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
		disp( "[SYSTEM] Left eye: FAIL\n" );
		putText( DMS->output.detectImage, "LEFT EYE", Point( sysState_default_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	}

	if( DMS->eyeDetector.fIrisDetect[1] == TRUE )
	{
		disp( "[SYSTEM] Right iris: PASS\n" );
		putText( DMS->output.detectImage, "RIGHT IRIS", Point( second_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_pass );

		if( DMS->eyeDetector.fEyeDetect[1] == TRUE )
		{
			disp( "[SYSTEM] Right eye: PASS\n" );
			putText( DMS->output.detectImage, "RIGHT EYE", Point( second_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_pass );
		}
		else
		{
			disp( "[SYSTEM] Right eye: FAIL\n" );
			putText( DMS->output.detectImage, "RIGHT EYE", Point( second_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
		}

	}
	else
	{
		disp( "[SYSTEM] Right iris: FAIL\n" );
		putText( DMS->output.detectImage, "RIGHT IRIS", Point( second_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
		disp( "[SYSTEM] Right eye: FAIL\n" );
		putText( DMS->output.detectImage, "RIGHT EYE", Point( second_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	}

	}


	disp( "[SYSTEM] Left iris: FAIL\n" );
	putText( DMS->output.detectImage, "LEFT IRIS", Point( sysState_default_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	disp( "[SYSTEM] Left eye: FAIL\n" );
	putText( DMS->output.detectImage, "LEFT EYE", Point( sysState_default_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	disp( "[SYSTEM] Right iris: FAIL\n" );
	putText( DMS->output.detectImage, "RIGHT IRIS", Point( second_x, irisState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
	disp( "[SYSTEM] Right eye: FAIL\n" );
	putText( DMS->output.detectImage, "RIGHT EYE", Point( second_x, eyeState_y ), FONT_HERSHEY_DUPLEX, fontSize, sysState_default );
#endif





#if 0
	/// Iris response text
	if( DMS->faceDetector.fFaceDetect == TRUE )
	{
		for( int eye = 0 ; eye < 2 ; eye++ )
		{
			sprintf_s( buf_char_path, "%d", DMS->eyeDetector.eyeData[eye].irisResp );
			putText( DMS->output.detectImage, buf_char_path, DMS->objects.eyeROIfromFace[eye].tl(), cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar( 50, 50, 50 ) );
		}
	}
#endif


#if 0
	/// Iris response visualization
	cv::Mat respOutputImage = DMS->grayImage.clone();
	for( int eye = 0 ; eye < 2 ; eye++ )
	{
		cv::Mat respImage;
		resize( DMS->eyeDetector.eyeData[eye].irisRespImage, respImage, cv::Size( DMS->eyeDetector.eyeData[eye].face2eyeROI.width, DMS->eyeDetector.eyeData[eye].face2eyeROI.height ) );

		respImage.copyTo( respOutputImage( DMS->objects.eyeROI[eye] ) );
		respImage.release();
	}

	sprintf_s( buf_char_msg, "%s/Result_%04d_irisResp.png", output_path, DMS->sysFlowControl.frameIdx );
	imwrite( buf_char_msg, respOutputImage );
	respOutputImage.release();
#endif



#endif




#endif



}


void Output_Video_Init( cv::VideoWriter *srcVideo, char *fileName, unsigned char fps, cv::Size size )
{

	sprintf_s( buf_char_path, "%s/%s.avi", output_path, fileName );
	srcVideo->open( buf_char_path, CV_FOURCC('M','J','P','G'), fps, size, TRUE );
}

void Output_Video_Write( cv::VideoWriter *srcVideo, cv::Mat srcImage )
{
	srcVideo->write( srcImage );
}

void Output_Video_Close( cv::VideoWriter *srcVideo )
{
	srcVideo->release();
}





void Output_Save_Image_Neuron_Vector_Map( char *fileName, int nCommitNeurons, int nCat, int vectorWidth, int vectorHeight, int nMapCols )
{
#if _OUTPUT_SAVE_IMAGE_NEURON_VECTOR_MAP
	unsigned char featVec[256];
	int nRows = nCommitNeurons / nMapCols;
	int neuronMapHeight = vectorHeight * nRows;
	int neuronMapWidth  = vectorWidth * nMapCols;
	cv::Mat neuronVecImage = cv::Mat::zeros( vectorHeight * nRows, vectorWidth * nMapCols, CV_8U );

	for( int catIdx = 1 ; catIdx < 1 + nCat ; catIdx++ )
	{
		// Initialization
		memset( neuronVecImage.data, 0, neuronMapWidth * neuronMapHeight * sizeof(unsigned char) );
		int catCnt = 0;

		// Access to neuron vector
		for( int idx = 0 ; idx < nCommitNeurons ; idx++ )
		{
			int context, aif, minif, category;
			ReadNeuron( idx, featVec, &context, &aif, &minif, &category );

			if( category == catIdx )
			{
				int nx = catCnt % nMapCols;
				int ny = catCnt / nMapCols;

				unsigned char *patch = &neuronVecImage.data[ ( nx * vectorWidth ) + ( ny * vectorHeight ) * neuronMapWidth ];
				for( int iy = 0 ; iy < vectorHeight ; iy++ )
				{
					for( int ix = 0 ; ix < vectorWidth ; ix++ )
						patch[ ix + iy * neuronMapWidth ] = featVec[ ix + iy * vectorWidth ];
				}

				catCnt++;
			}
		}

		sprintf_s( buf_char_path, "%s/%s_cat%d.png", output_path, fileName, catIdx );
		imwrite( buf_char_path, neuronVecImage );
	}

	neuronVecImage.release();

#endif
}




cv::Mat get_hogdescriptor_visual_image( cv::Mat& origImg, std::vector< float>& descriptorValues, cv::Size winSize, cv::Size cellSize, int gradientBinSize,
	int scaleFactor, double viz_factor,	cv::Scalar LineColor )
{   
	cv::Mat visual_image;
	resize(origImg, visual_image, cv::Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
	cvtColor(visual_image, visual_image, CV_GRAY2BGR);

	// dividing 180¡Æ into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14/(float)gradientBinSize; 

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter   = new int*[cells_in_y_dir];
	for (int y=0; y< cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x=0; x< cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin=0; bin< gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx=0; blockx< blocks_in_x_dir; blockx++)
	{
		for (int blocky=0; blocky< blocks_in_y_dir; blocky++)            
		{
			// 4 cells per block ...
			for (int cellNr=0; cellNr< 4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr==1) celly++;
				if (cellNr==2) cellx++;
				if (cellNr==3)
				{
					cellx++;
					celly++;
				}

				for (int bin=0; bin< gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[ descriptorDataIdx ];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (int celly=0; celly< cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx< cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin=0; bin< gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}


	// draw cells
	for (int celly=0; celly< cells_in_y_dir; celly++)
	{
		for (int cellx=0; cellx< cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width/2;
			int my = drawY + cellSize.height/2;

			cv::rectangle(visual_image,
				cv::Point(drawX*scaleFactor,drawY*scaleFactor ),
				cv::Point((drawX+cellSize.width)*scaleFactor,
				( drawY + cellSize.height ) * scaleFactor ),
				CV_RGB(100,100,100),
				1);

			// draw in each cell all 9 gradient strengths
			for (int bin=0; bin< gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength==0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

				float dirVecX = cos( currRad );
				float dirVecY = sin( currRad );
				float maxVecLen = cellSize.width/2;
				float scale = viz_factor; // just a visual_imagealization scale,
										  // to see the lines better

										  // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
					cv::Point(x1*scaleFactor,y1*scaleFactor),
					cv::Point(x2*scaleFactor,y2*scaleFactor),
					LineColor,
					1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y=0; y< cells_in_y_dir; y++)
	{
		for (int x=0; x< cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];            
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;

}


void Output_Save_Neuron_KNF( char *fileName )
{
	sprintf_s( buf_char_msg, "%s/%s", output_path, fileName );
	SaveNeurons( buf_char_msg );

}