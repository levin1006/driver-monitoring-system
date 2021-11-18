#pragma once

#include "DMS_Algo_typedef.h"

char *Get_knowledge_path( int opMode );

void Set_Parameter_FaceDetection( sParam_faceDetector &param, unsigned char fInit );

void Set_Parameter_EyeDetection( sParam_eyeDetector &param );

void Set_Parameter_Closed_Eye_HOG( sHOG &closedEye_HOG );