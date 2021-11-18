#include <Windows.h>	

#include "DMS_Algo_Global_Parameter.h"'
#include "DMS_Algo_typedef.h"
#include "DMS_Algo_Parameters.h"
#include "DMS_Algorithm.h"

#include "DMS_Simulation_typedef.h"
#include "DMS_Simulation_User_Parameter.h"
#include "DMS_Simulation_Global_Parameter.h"
#include "DMS_Simulation_Parameters.h"
#include "DMS_Simulation_Output.h"

#include "DMS_Simulation_Utilities.h"



// Proto typing
static void Interface_Initialization( sTestEnvironment &testEnv );
static void Set_TestSet( sInputFrame &inputFrame );
static void Init_CAM( cv::VideoCapture &input_CAM );
static unsigned char *Read_Image( sInputFrame &inputFrame );
void Terminate_interface( sTestEnvironment &testEnv );


static char buf_char[1024];
SYSTEMTIME now;





void DMS_Simulation_Bench()
{
	
	sTestEnvironment testEnv;
	Set_Parameter_inputFrame( testEnv.inputFrame.param );
	Get_Dataset_List( testEnv.inputFrame.datasetList );
	Set_output_path( "" );


	for( testEnv.inputFrame.datasetIdx = 0 ; testEnv.inputFrame.datasetIdx < testEnv.inputFrame.datasetList.size() ; testEnv.inputFrame.datasetIdx++ )
	{


		// Test environment initialization
		Interface_Initialization( testEnv );


		// System initialization
		
		disp( "[PROC] Initialization\n" );
		stopWatch(_STOPWATCH_START);
		sDMS *DMS = DMS_Algorithm_Initialization();
		stopWatch(_STOPWATCH_STOP);
		



		for( testEnv.inputFrame.frameIdx = testEnv.inputFrame.param.dataset_firstFrame; 
			 testEnv.inputFrame.frameIdx <= testEnv.inputFrame.dataset_lastFrameIdx ; 
			 testEnv.inputFrame.frameIdx++ )
		{
			// Measure latency frame by frame
			sprintf_s( buf_char, "- Frame #%u\n", testEnv.inputFrame.frameIdx );
			disp( buf_char );

			

			// Read Image
			disp( "[PROC] Read image\n" );
			stopWatch(_STOPWATCH_START);
			Read_Image( testEnv.inputFrame );
			stopWatch(_STOPWATCH_STOP);

			// Execute DMS Vision algorithm
			DMS_Algorithm( DMS, testEnv.inputFrame.grayImage.data );


			// Check elapsed time
			testEnv.timeElapsed = getTime_elapsed( testEnv.timeCapture );
			float FPS = 1000 / (float)testEnv.timeElapsed;
			testEnv.FPS = ( ( 1000 / (float)testEnv.timeElapsed ) + 30 * testEnv.FPS ) / (float) ( 30 + 1 );		/// CME of FPS for 30 frames




			// Output visualization
			stopWatch(_STOPWATCH_START);
			output( testEnv, DMS );
			disp( "[PROC] Output\n" );
			stopWatch(_STOPWATCH_STOP);





		}

		delete DMS;
		Terminate_interface( testEnv );
	}





	disp( "\n- End of line\n" );
}


void Interface_Initialization( sTestEnvironment &testEnv )
{
	// Timer initialization
	testEnv.timeCapture = getTime_init();
	GetSystemTime(&now);

	if( testEnv.inputFrame.param.inputSource == _INPUT_SOURCE_REALTIME_CAMERA )
		sprintf_s( buf_char, "real-time_%d%02d%02d_%02d%02d%02d", now.wYear, now.wMonth, now.wDay, (now.wHour + 9) % 24, now.wMinute, now.wSecond, now.wMilliseconds );
	else
		sprintf_s( buf_char, "%s", testEnv.inputFrame.datasetList[ testEnv.inputFrame.datasetIdx ].name.c_str() );

	Set_output_path( buf_char );


	Set_TestSet( testEnv.inputFrame );
	testEnv.FPS = 30;


	// Output
	testEnv.output.image_output = cv::Mat::zeros( cv::Size( _INPUT_IMAGE_WIDTH, _INPUT_IMAGE_HEIGHT ), CV_8UC3 );
	testEnv.output.video_output = new cv::VideoWriter;

#if _OUTPUT_SAVE_VIDEO_DETECTION_RESULTS

	if( testEnv.inputFrame.param.inputSource == _INPUT_SOURCE_REALTIME_CAMERA )
		sprintf_s( buf_char, "real-time_%d%02d%02d_%02d%02d%02d", now.wYear, now.wMonth, now.wDay, (now.wHour + 9) % 24, now.wMinute, now.wSecond, now.wMilliseconds );
	else
		sprintf_s( buf_char, "%s", testEnv.inputFrame.datasetList[ testEnv.inputFrame.datasetIdx ].name.c_str() );

	Output_Video_Init( testEnv.output.video_output, "video_output", testEnv.inputFrame.param.output_video_FPS, cv::Size( _INPUT_IMAGE_WIDTH, _INPUT_IMAGE_HEIGHT ) );
#endif


	Set_Parameter_Output( testEnv.output );


}

void Set_TestSet( sInputFrame &inputFrame )
{
	if( inputFrame.param.inputSource == _INPUT_SOURCE_REALTIME_CAMERA )
	{
		Init_CAM( inputFrame.input_CAM );
		inputFrame.dataset_lastFrameIdx = inputFrame.param.dataset_lastFrame;
	}
	else
	{
		cv::glob( inputFrame.datasetList[ inputFrame.datasetIdx ].path, inputFrame.dataset, FALSE );
		sprintf_s( buf_char, "[SYSTEM] Dataset '%s' is loaded.\n", inputFrame.datasetList[ inputFrame.datasetIdx ].name.c_str() );
		disp( buf_char );

		if( inputFrame.param.dataset_lastFrame == -1 )
			inputFrame.dataset_lastFrameIdx = inputFrame.dataset.size() - 1;
		else
			inputFrame.dataset_lastFrameIdx = MIN( inputFrame.param.dataset_lastFrame, inputFrame.dataset.size() - 1 );
	}
}

void Init_CAM( cv::VideoCapture &input_CAM )
{
	input_CAM.open(0);
	//input_CAM.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M','J','P','G'));
	//input_CAM.set(CV_CAP_PROP_EXPOSURE, -5);
	input_CAM.set(CV_CAP_PROP_FRAME_WIDTH,_INPUT_IMAGE_WIDTH);
	input_CAM.set(CV_CAP_PROP_FRAME_HEIGHT,_INPUT_IMAGE_HEIGHT);
	input_CAM.set(CV_CAP_PROP_FPS,_INPUT_FRAME_RATE);
	cv::waitKey(10);
}




unsigned char *Read_Image( sInputFrame &inputFrame )
{
	if( inputFrame.param.inputSource == _INPUT_SOURCE_REALTIME_CAMERA )
	{
		inputFrame.input_CAM >> inputFrame.srcImage;
		flip( inputFrame.srcImage, inputFrame.srcImage, 1 );
		cvtColor( inputFrame.srcImage, inputFrame.grayImage, cv::COLOR_RGB2GRAY );
	}
	else
	{
		inputFrame.srcImage = imread( inputFrame.dataset[ inputFrame.frameIdx ], cv::IMREAD_COLOR );
		cvtColor( inputFrame.srcImage, inputFrame.grayImage, cv::COLOR_RGB2GRAY );
	}



	cv::Size imgSize = inputFrame.srcImage.size();

	if( ( imgSize.width != _INPUT_IMAGE_WIDTH ) || ( imgSize.height != _INPUT_IMAGE_HEIGHT ) )
	{
		disp( "[ERROR] Not matching image size" );
		system( "pause" );
	}

	return inputFrame.grayImage.data;
		
}




void Terminate_interface( sTestEnvironment &testEnv )
{
	Output_Video_Close( testEnv.output.video_output );
	testEnv.inputFrame.dataset.clear();
}
