

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
//#include "features2d.hpp"

#include "GVAPI.h"

#include "DMS_Algo_Global_Parameter.h"
#include "definition_macro.h"
#include "DMS_Algo_typedef.h"

#include "DMS_Algo_Utilities.h"
#include "DMS_Algo_Parameters.h"

#include "DMS_Algo_Feature_Extraction.h"



// Prototyping


static void Eyelid_Feature_Point_Detection( sEyeData &eyeData, sParam_eyeDetector param );

static void Eyelid_Quadratic_Curve_Fitting_RANSAC( std::vector<cv::Point> points, std::vector<cv::Point> &inliers, cv::Mat *model, float residual_threshold = 1, int nIterations = 100 );

void Eye_Gaze_Direction_Detection( sEyeData &eyeData, sParam_eyeDetector param );


static char buf_char[1024];



void Eye_Detection_Preprocessing( sEyeDetector &eyeDetector, cv::Mat faceImage, unsigned char fFaceDetect )
{
	int faceWidth = faceImage.cols;
	int faceHeight = faceImage.rows;
	short kernel[3] = { -1.0, 0, 1.0 };
	cv::Mat filterX( 1, 3, CV_16S, kernel );
	cv::Mat filterY( 3, 1, CV_16S, kernel );

	if( fFaceDetect == TRUE )
	{
		eyeDetector.eyeData[0].face2eyeROI.x	  = faceWidth  * eyeDetector.param.eyeRoiVsFace.x;
		eyeDetector.eyeData[0].face2eyeROI.y	  = faceHeight * eyeDetector.param.eyeRoiVsFace.y;
		eyeDetector.eyeData[0].face2eyeROI.width  = faceWidth  * eyeDetector.param.eyeRoiVsFace.width;
		eyeDetector.eyeData[0].face2eyeROI.height = faceHeight * eyeDetector.param.eyeRoiVsFace.height;

		eyeDetector.eyeData[1].face2eyeROI.x	  = faceWidth  * ( 1 - eyeDetector.param.eyeRoiVsFace.x - eyeDetector.param.eyeRoiVsFace.width );
		eyeDetector.eyeData[1].face2eyeROI.y	  = faceHeight * eyeDetector.param.eyeRoiVsFace.y;
		eyeDetector.eyeData[1].face2eyeROI.width  = faceWidth  * eyeDetector.param.eyeRoiVsFace.width;
		eyeDetector.eyeData[1].face2eyeROI.height = faceHeight * eyeDetector.param.eyeRoiVsFace.height;


		/// Initialize variables
		eyeDetector.eyeResizeScale = eyeDetector.eyeData[0].face2eyeROI.width / (float)eyeDetector.param.eyeRoiResize.width;


		for( int eye = 0 ; eye < 2 ; eye++ )
		{


			resize( faceImage( eyeDetector.eyeData[eye].face2eyeROI ), eyeDetector.eyeData[eye].eyeROI, eyeDetector.param.eyeRoiResize );


			/// Integral image
			integral( eyeDetector.eyeData[eye].eyeROI, eyeDetector.eyeData[eye].eyeROI_intg, CV_32S );

			/// Gradient
			filter2D( eyeDetector.eyeData[eye].eyeROI, eyeDetector.eyeData[eye].eyeGx, CV_16S, filterX, cv::Point (-1, -1), 0, cv::BORDER_REPLICATE );
			filter2D( eyeDetector.eyeData[eye].eyeROI, eyeDetector.eyeData[eye].eyeGy, CV_16S, filterY, cv::Point (-1, -1), 0, cv::BORDER_REPLICATE );


			/// Edge map
			eyeDetector.eyeData[eye].eyeEdge = abs( eyeDetector.eyeData[eye].eyeGx ) + abs( eyeDetector.eyeData[eye].eyeGy );

			/// Integral image of edge maip
			eyeDetector.eyeData[eye].eyeEdge.convertTo( eyeDetector.eyeData[eye].eyeEdge, CV_8U );
			integral( eyeDetector.eyeData[eye].eyeEdge, eyeDetector.eyeData[eye].eyeEdge_intg, CV_8U );

		}



	}


}


void Eye_Blink_Detection( sEyeData &eyeData, sHOG closedEye_HOG, sParam_eyeDetector param )
{

	// Iris region recognition
	unsigned char featVec[ _NEURON_MAX_VECTOR_SIZE ];
	int width  = param.irisMinRadius * 4;
	int height = param.irisMinRadius * 4;

	int ix = eyeData.iris.x - width / 2;
	if( ix < 0 )
		ix = 0;
	else if( ix + width > eyeData.eyeROI.cols )
		ix = eyeData.eyeROI.cols - width;
	else
		ix = ix;

	int iy = eyeData.iris.y - height / 2;
	if( iy < 0 )
		iy = 0;
	else if( iy + height > eyeData.eyeROI.rows )
		iy = eyeData.eyeROI.rows - height;
	else
		iy = iy;

	cv::Mat resizeImage;
	resize( eyeData.eyeROI( cv::Rect( ix, iy, width, height ) ), resizeImage, param.closedEyeRegion );

	/// HOG feature extraction
	FeatureExtraction_HOG_Descriptor( resizeImage, closedEye_HOG.HOG, featVec );


	int NM_distOut, NM_catOut, NM_nidOut;
	int ncount = BestMatch( featVec, closedEye_HOG.vecLen, &NM_distOut, &NM_catOut, &NM_nidOut );


	if( ncount > 0 )
		eyeData.fEyeState = _EYE_CLOSED;
	else
		eyeData.fEyeState = _EYE_OPENED;

}






void Iris_Detection( sEyeData &eyeData, sParam_eyeDetector param )
{

	int width		  = eyeData.eyeROI.cols;
	int height		  = eyeData.eyeROI.rows;
	int irisMinRadius = param.irisMinRadius;
	int irisMaxRadius = param.irisMaxRadius;
	int winSize		  = irisMinRadius * 4;
	int winSizeHalf   = winSize / 2;
	int winSizeQt	  = winSize / 4;
	float winArea	  = SQUARE( winSize );
	float winAreaHalf = SQUARE( winSizeHalf );



	std::vector<cv::Point2i> iris;
	cv::Mat respMap( eyeData.eyeROI.size(), CV_32S );

	/// Windowing using around-iris-mask
	/// ¡á: white / ¡à: black / ¢É: empty
	/// ¡á¡á¡á¡á¡á¡á¡á¡á ¡é
	/// ¡á¡á¡á¡á¡á¡á¡á¡á
	/// ¡á¡á¡à¡à¡à¡à¡á¡á
	/// ¡á¡á¡à¡à¡à¡à¡á¡á
	/// ¡á¡á¡à¡à¡à¡à¡á¡á
	/// ¡á¡á¡à¡à¡à¡à¡á¡á
	/// ¡á¡á¡á¡á¡á¡á¡á¡á
	/// ¡á¡á¡á¡á¡á¡á¡á¡á
	int max = -10000000000;
	int min = 100000000000;
		
	for( int iy = 0 ; iy < height - winSize ; iy++ )
	{
		for( int ix = 0 ; ix < width - winSize ; ix++)
		{
			int blackSum = Get_Integral_Rect_32S( eyeData.eyeROI_intg, cv::Rect( ix + winSizeQt, iy + winSizeQt, winSizeHalf, winSizeHalf ) );
			int whiteSum = Get_Integral_Rect_32S( eyeData.eyeROI_intg, cv::Rect( ix, iy, winSize, winSize ) ) - blackSum;

			int blackAvg = blackSum / winAreaHalf;
			int whiteAvg = whiteSum / (winArea - winAreaHalf );
			int resp1 = whiteAvg - 1 * blackAvg;


			int resp2 = Get_Integral_Rect_32S( eyeData.eyeEdge_intg, cv::Rect( ix + winSizeQt, iy + winSizeQt, winSizeHalf, winSizeHalf ) ) / winAreaHalf;


			float resp;
			resp = 1.0 * resp1 + 0.0 * resp2;




#if 0
			/// CHT
			resp = 0;
			for( int py = -irisMaxRadius ; py < irisMaxRadius ; py++ )
			{
				for( int px = -irisMaxRadius ; px < irisMaxRadius ; px++ )
				{
					float dist = abs( px ) + abs( py );
					if( dist > irisMinRadius )
					{
						float theta = atanf( py / (float)px );
						theta = ( px > 0 ) ? theta : theta + CV_PI;
							


						float wx = cos( theta );

						//if( abs( wx ) > 0.7 ) // horizontal edge only
						//{
							float wy = sin( theta );
							float sx = eyeData.eyeGx.at<short>( iy + winSizeHalf + py, ix + winSizeHalf + px );
							float sy = eyeData.eyeGy.at<short>( iy + winSizeHalf + py, ix + winSizeHalf + px );

							resp += wx * sx + wy * sy;
						//}


					}
				}
			}

			resp += ( 255 - blackAvg ) * 0.0;
#endif





			respMap.at<int>( iy + winSizeHalf, ix + winSizeHalf ) = resp;
			max = MAX( max, resp );
			min = MIN( min, resp );
		}
	}

	/// Pick best match
	eyeData.irisRespImage = cv::Mat::zeros( eyeData.eyeROI.size(), CV_8U );
	for( int iy = winSizeHalf ; iy < height - winSizeHalf ; iy++ )
	{
		for( int ix = winSizeHalf ; ix < width - winSizeHalf ; ix++ )
		{
			/// for visualization
			eyeData.irisRespImage.at<unsigned char>( iy, ix ) = ( respMap.at<int>( iy, ix ) - min ) / (float)( max - min ) * 255;

			if( respMap.at<int>( iy, ix ) == max )
				iris.push_back( cv::Point2i( ix, iy ) );
		}
	}


		

	/// Clustering: averaging
	eyeData.iris.x = 0;  
	eyeData.iris.y = 0;
	for( int idx = 0 ; idx <  iris.size() ; idx++ )
	{
		eyeData.iris.x += iris[idx].x;  
		eyeData.iris.y += iris[idx].y;
	}
	eyeData.iris.x /= (float)iris.size();
	eyeData.iris.y /= (float)iris.size();

	eyeData.irisResp = max;

	respMap.release();
	iris.clear();

}


void Eye_Lid_Detection( sEyeData &eyeData, sParam_eyeDetector param )
{
	Eyelid_Feature_Point_Detection( eyeData, param );

	if( eyeData.eyelid_feat.size() > param.minFeaturePoints )
	{
		eyeData.fEyeLidDetect = TRUE;
		Eyelid_Quadratic_Curve_Fitting_RANSAC( eyeData.eyelid_feat, eyeData.eyelid_inlier, &eyeData.eyelid_model, param.eyelid_RANSAC_residual_threshold, param.eyelid_RANSAC_nIteration );
	}
	else
		eyeData.fEyeLidDetect = FALSE;

}


void Eyelid_Feature_Point_Detection( sEyeData &eyeData, sParam_eyeDetector param )
{

	float verticalRange = 1.5;
	int height			= eyeData.eyeROI.rows;
	int width			= eyeData.eyeROI.cols;
	int irisMinRadius	= param.irisMinRadius;
	
	int nSamples		= param.nSamples;	
	int sampleInterval	= param.sampleInterval;
	cv::Point searchRange( nSamples / 2 * sampleInterval, irisMinRadius );

	int edgeWeight		= param.edgeWeight;
	int darknessWeight	= param.darknessWeight;
	int maxBrightness	= param.maxBrightness;
	int maxEdge			= param.maxEdge;
	int threshold_eyelidResp = param.threshold_eyelidResp;
	
	int ref_ix = eyeData.iris.x;

	for( int is = -searchRange.x ; is <= searchRange.x ; is += sampleInterval )
	{
		int ix = ref_ix + is;
		if( isinside_( ix, 0, width ) )
		{
				
			int yMin = max_( 0, (int)( eyeData.iris.y - verticalRange * searchRange.y ) );
			int yMax = min_( height, (int)( eyeData.iris.y + verticalRange * searchRange.y ) );


			/// Upper
			/// Windowing using around-iris-mask
			/// ¡á: white / ¡à: black / ¢É: empty
			/// ¡á¡á
			/// ¡à¡à
			int maxUpper = -100000000;
			cv::Mat respMap = cv::Mat::zeros( cv::Size(1, yMax - yMin ), CV_32S );
			for( int iy = yMin, idx = 0 ; iy < yMax ; iy++, idx++ )
			{
				if( eyeData.eyeGy.at<short>( iy, ix ) < 0 )
				{
					int resp = edgeWeight * eyeData.eyeEdge.at<unsigned char>( iy, ix )
							   + darknessWeight * ( maxBrightness - eyeData.eyeROI.at<unsigned char>( iy, ix ) );
						
					respMap.at<float>( idx ) = resp;
					maxUpper = max_( maxUpper, resp );
				}

			}



			for( int idx = 0 ; idx < yMax - yMin ; idx++ )
			{
				if( respMap.at<float>( idx ) == maxUpper )
				{
					if( maxUpper > threshold_eyelidResp )
						eyeData.eyelid_feat.push_back( cv::Point( ix, yMin + idx ) );
				}

			}

			respMap.release();
		}
	}

}


void Eyelid_Quadratic_Curve_Fitting_RANSAC( std::vector<cv::Point> points, std::vector<cv::Point> &inliers, cv::Mat *model, float residual_threshold, int nIterations )
{
	//if( flag_random_sample == 1 )
	//	srand(time(NULL));

	int nPoints = points.size();


	// Build matrix
	cv::Mat A( nPoints, 3, CV_32F );
	cv::Mat B( nPoints, 1, CV_32F );

	for( int i = 0 ; i < nPoints ; i++ )
	{
		A.at<float>(i,0) = sq_( points[i].x );
		A.at<float>(i,1) = points[i].x;
		A.at<float>(i,2) = 1.0;

		B.at<float>(i,0) = points[i].y;
	}


	// RANSAC fitting 

	int max_cnt = 0;
	cv::Mat best_model( 3, 1, CV_32F ) ;

	for( int i = 0 ; i < nIterations ; i++ )
	{
		/// Random sampling - 3 point  
		int k[3] = {-1, } ;
		k[0] = floor( ( std::rand() % nPoints ) );

		do
		{
			k[1] = floor( ( std::rand() % nPoints ) );
		}
		while( ( k[1] == k[0] ) || ( k[1] < 0 ) );

		do
		{
			k[2] = floor( ( std::rand() % nPoints ) );
		}
		while( ( k[2] == k[0] ) || ( k[2] == k[1] ) || ( k[2] < 0 ) );

		/// Model estimation
		cv::Mat AA( 3, 3, CV_32F );
		cv::Mat BB( 3, 1, CV_32F );
		for( int j = 0 ; j < 3 ; j++ )
		{
			AA.at<float>(j,0) = sq_( points[ k[j] ].x );
			AA.at<float>(j,1) = points[ k[j] ].x;
			AA.at<float>(j,2) = 1.0 ;

			BB.at<float>(j,0) = points[ k[j] ].y;
		}

		cv::Mat AA_pinv( 3, 3, CV_32F ) ;
		invert( AA, AA_pinv, cv::DECOMP_SVD );

		cv::Mat X = AA_pinv * BB;	

		/// Evaluation 
		cv::Mat residual( nPoints, 1, CV_32F ) ;
		residual = abs( B - A * X ) ;
		int cnt = 0 ;
		for( int j = 0 ; j < nPoints ; j++ )
		{
			if( residual.at<float>(j,0) < residual_threshold ) 
				cnt++ ;
		}

		if( cnt > max_cnt ) 
		{
			best_model = X ;
			max_cnt = cnt ;
		}
	}

	/// Refine model
	cv::Mat residual = cv::abs( A * best_model - B ) ;
	for( int i = 0 ; i < nPoints ; i++ )
	{
		if( residual.at<float>( i, 0 ) < residual_threshold ) 
			inliers.push_back( points[i] ) ;
	}

	cv::Mat A2( inliers.size() ,3, CV_32F ) ;
	cv::Mat B2( inliers.size() ,1, CV_32F ) ;

	for( int i = 0 ; i < inliers.size() ; i++ )
	{
		A2.at<float>(i,0) = sq_( inliers[i].x );
		A2.at<float>(i,1) = inliers[i].x;
		A2.at<float>(i,2) = 1.0;

		B2.at<float>(i,0) = inliers[i].y;
	}

	cv::Mat A2_pinv( 3, inliers.size(), CV_32F ) ;
	invert( A2, A2_pinv, cv::DECOMP_SVD );

	cv::Mat X = A2_pinv * B2 ;

	*model = X.clone();
}


void Eye_Gaze_Direction_Detection( sEyeData &eyeData, sParam_eyeDetector param )
{
	/// Validity check
	if( ( eyeData.fEyeLidDetect == TRUE )	
	 && ( eyeData.eyelid_model.at<float>(0) > param.threshold_eyelidConvexity )
	 && ( eyeData.fEyeState == _EYE_OPENED )
		)
	{
		cv::Point eyeTopCenter;
		/// f(x) = ax^2 + bx + c
		/// f'(x) = 2ax + b
		/// x_min = -b / 2a
		eyeTopCenter.x = eyeData.eyelid_model.at<float>(1) / ( - 2 * eyeData.eyelid_model.at<float>(0) ) + 0.5;
		eyeTopCenter.y = eyeData.eyelid_model.at<float>(0) * sq_( eyeTopCenter.x )
					   + eyeData.eyelid_model.at<float>(1) * eyeTopCenter.x
					   + eyeData.eyelid_model.at<float>(2);

		eyeData.eyeTopCenter = eyeTopCenter;


		int irisLatOffset = eyeData.iris.x - eyeTopCenter.x;

		if( irisLatOffset < - param.threshold_irisLatOffset )
			eyeData.eyeGazeDirection = _EYE_GAZE_LEFT;
		else if( irisLatOffset > param.threshold_irisLatOffset )
			eyeData.eyeGazeDirection = _EYE_GAZE_RIGHT;
		else
			eyeData.eyeGazeDirection = _EYE_GAZE_CENTER;

	}
	else
		eyeData.eyeGazeDirection = _EYE_GAZE_NONE;

} 