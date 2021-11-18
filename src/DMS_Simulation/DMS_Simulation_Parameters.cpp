//
#include "DMS_Simulation_Global_Parameter.h"
#include "DMS_Simulation_typedef.h"	
#include "DMS_Simulation_Dataset_Definition.h"
#include "DMS_Simulation_Utilities.h"


static char buf_char[1024];

void Set_Parameter_inputFrame( sParam_inputFrame &param )
{
	// System flow control
	/// _INPUT_SOURCE_DATASET, _INPUT_SOURCE_REALTIME_CAMERA
	param.inputSource = _INPUT_SOURCE_REALTIME_CAMERA;
	param.dataset_firstFrame = 0;
	param.dataset_lastFrame  = INT_MAX;//635;//-1;//310;//138;//-1;//9999;

	param.output_video_FPS = 20;


}

void Get_Dataset_List( std::vector<sPathName> &list )
{
	cv::String *dataset_list;

	//dataset_list = PARTRON_NIR_IMAGE_SAMPLES;

	//dataset_list = JABIL_NIR_IMAGE_SAMPLES;
	dataset_list = MHE1804_NIR;
	//dataset_list = MHE1712_instance;
	//dataset_list = MHE1709;
	//dataset_list = MHE1712_ALL_B02;
	//dataset_list = MHE1712_ALL_B;
	//dataset_list = MHE1712_ALL_G;
	//dataset_list = MHE1712_ALL_H;

	for( int idx = 0 ; idx< _MAX_DATASET ; idx++ )
	{
		if( dataset_list[idx].size() != 0 )
		{
			sDataset pathName;
			cv::String dir, folderName, folderNameUpper;
			pathName.path = dataset_list[idx];

			separate_dir_and_file( pathName.path, dir, folderName );

			folderNameUpper = get_fileName_from_fullPath( dir );

			pathName.name = folderNameUpper + "_" + folderName;

			list.push_back( pathName );
		}
	}
}


void Set_Parameter_Output( sOutput &output )
{
	output.fOutput_frameFPS = 1;
	output.fOutput_faceBox = 1;
	output.fOutput_eyeROI = 0;
	output.fOutput_iris = 1;
	output.fOutput_eyelid = 1;
	output.fOutput_eyeGaze = 1;
	output.fOutput_drowsyDetection = 0;
}


