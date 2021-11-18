

#include "DMS_Algo_typedef.h"
#include "DMS_Algo_HOG_OpenCV.h"


int FeatureExtraction_HOG_Descriptor( cv::Mat grayImage, HOGDescriptor &HOG, unsigned char *featVec )
{
	std::vector<float> descriptor;

	HOG.compute( grayImage, descriptor, cv::Size(1,1) );

	/// Convert to feature vector for training neurons
	for( int idx = 0 ; idx < descriptor.size() ; idx++ )
		featVec[idx] = descriptor[idx] * 256 + 0.5;

	return descriptor.size();
}


