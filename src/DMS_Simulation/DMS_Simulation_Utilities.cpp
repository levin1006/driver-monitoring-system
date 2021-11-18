#include <stdio.h>
#include <chrono>

#include "opencv2\highgui.hpp"

#include "DMS_Simulation_Output.h"


#define _N_STOPWATCH 10
static char buf_char[256];
static std::chrono::steady_clock::time_point tStart[_N_STOPWATCH];
static unsigned char bufIdx = 0;
void stopWatch( unsigned char fRun, unsigned char dispFPS )
{
	if( fRun == 1 )
	{
		if( bufIdx + 1 < _N_STOPWATCH )
			tStart[bufIdx++] = std::chrono::high_resolution_clock::now();
		else
		{
			disp("[ERROR] No more stopwatch");
			system("pause");
		}
	}
	else
	{
		if( bufIdx != 0 )
		{
			auto elapsed = std::chrono::high_resolution_clock::now() - tStart[--bufIdx];
			long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
			sprintf_s( buf_char, "[INFO] Elapsed Time: %.3fms", ( microseconds / 1000.0 ) );

			if( dispFPS == 1 )
				sprintf_s( buf_char, "%s, FPS: %d\n", buf_char, 1000000 / ( microseconds ) );
			else
				sprintf_s( buf_char, "%s\n", buf_char );

			disp( buf_char ); 
		}
	}
}


std::chrono::steady_clock::time_point getTime_init()
{
	return std::chrono::high_resolution_clock::now();
}


float getTime_elapsed( std::chrono::steady_clock::time_point &ref )
{
	std::chrono::steady_clock::time_point now = std::chrono::high_resolution_clock::now();
	auto elapsed = now - ref;
	ref = now;
	return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000;
}




void separate_dir_and_file( cv::String fullPath, cv::String &dir, cv::String &fileName )
{
	std::string fileNameWithExt;

	// Separate file name and path
	std::string::size_type found = fullPath.find_last_of("/\\");
	if( found != std::string::npos )
	{
		dir = fullPath.substr( 0, found );
		fileName = fullPath.substr( found + 1 );
	}
	else
		printf("[ERROR] Cannot find image file in the directory");



}

cv::String get_fileName_from_fullPath( cv::String fullPath )
{
	std::string fileNameWithExt;
	
	// Separate file name and path
	std::string::size_type found = fullPath.find_last_of("/\\");
	if( found != std::string::npos )
		fileNameWithExt = fullPath.substr( found + 1 );
	else
		printf("[ERROR] Cannot find image file in the directory");

	return fileNameWithExt;
}

cv::String get_fileName_removing_extension( cv::String fileNameWithExt )
{
	std::string fileName;

	// Separate file name and extension
	std::string::size_type found = fileNameWithExt.find_last_of( "." );
	if( found != std::string::npos )
		fileName = fileNameWithExt.substr( 0, found );
	else
		printf("[ERROR] Cannot find image file in the directory");

	return fileName;
}


