#pragma once

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "DMS_Algo_typedef.h"
#include "DMS_Simulation_typedef.h"




void disp( char *message );


void Set_output_path( std::string folderName );

void output( sTestEnvironment &testEnv, sDMS *DMS );

void Output_Save_Image_Neuron_Vector_Map( char *fileName, int nCommitNeurons, int nCat, int vectorWidth, int vectorHeight, int nMapCols );

void Output_Video_Init( cv::VideoWriter *srcVideo, char *fileName, unsigned char fps, cv::Size size );
void Output_Video_Write( cv::VideoWriter *srcVideo, cv::Mat BBoxImage );
void Output_Video_Close( cv::VideoWriter *srcVideo );


void Output_Save_Neuron_KNF( char *fileName );