
#include "opencv2/highgui.hpp"

#include "DMS_Algo_Utilities.h"


int Get_Integral_Rect_32S( cv::Mat intgImage, cv::Rect rect )
{
	return intgImage.at<int>(rect.y + rect.height, rect.x + rect.width) 
		 - intgImage.at<int>(rect.y + rect.height, rect.x) 
		 - intgImage.at<int>(rect.y				 , rect.x + rect.width) 
		 + intgImage.at<int>(rect.y				 , rect.x) ;

}

float Get_Integral_Rect_32F( cv::Mat intgImage, cv::Rect rect )
{
	return intgImage.at<float>(rect.y + rect.height, rect.x + rect.width) 
		 - intgImage.at<float>(rect.y + rect.height, rect.x) 
		 - intgImage.at<float>(rect.y			   , rect.x + rect.width) 
		 + intgImage.at<float>(rect.y			   , rect.x) ;

}
