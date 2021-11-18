#pragma once

//#include "opencv2/core/core_c.h"
//#include "opencv2/core/types.hpp"


/****************************************************************************************\
*                         haar.hpp                           *
\****************************************************************************************/

#define CV_HAAR_FEATURE_MAX_LOCAL 3

typedef int sumtype;
typedef double sqsumtype;

typedef struct CvHidHaarFeature
{
	struct
	{
		sumtype *p0, *p1, *p2, *p3;
		float weight;
	}
	rect[CV_HAAR_FEATURE_MAX_LOCAL];
} CvHidHaarFeature;


typedef struct CvHidHaarTreeNode
{
	CvHidHaarFeature feature;
	float threshold;
	int left;
	int right;
} CvHidHaarTreeNode;


typedef struct CvHidHaarClassifier
{
	int count;
	//CvHaarFeature* orig_feature;
	CvHidHaarTreeNode* node;
	float* alpha;
} CvHidHaarClassifier;

#define calc_sumf(rect,offset) \
    static_cast<float>((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

namespace cv_haar_avx
{
#if 0 /*CV_TRY_AVX*/
#define CV_HAAR_USE_AVX 1
#else
#define CV_HAAR_USE_AVX 0
#endif

#if CV_HAAR_USE_AVX
	// AVX version icvEvalHidHaarClassifier.  Process 8 CvHidHaarClassifiers per call. Check AVX support before invocation!!
	double icvEvalHidHaarClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
	double icvEvalHidHaarStumpClassifierAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
	double icvEvalHidHaarStumpClassifierTwoRectAVX(CvHidHaarClassifier* classifier, double variance_norm_factor, size_t p_offset);
#endif
}


/****************************************************************************************\
*                         objdetect_c.h                           *
\****************************************************************************************/

#define CV_HAAR_MAGIC_VAL    0x42500000
#define CV_TYPE_NAME_HAAR    "opencv-haar-classifier"

#define CV_IS_HAAR_CLASSIFIER( haar )                                                    \
    ((haar) != NULL &&                                                                   \
    (((const CvHaarClassifierCascade*)(haar))->flags & CV_MAGIC_MASK)==CV_HAAR_MAGIC_VAL)

#define CV_HAAR_FEATURE_MAX  3
#define CV_HAAR_STAGE_MAX 1000

typedef struct CvHaarFeature
{
	int tilted;
	struct
	{
		CvRect r;
		float weight;
	} rect[CV_HAAR_FEATURE_MAX];
} CvHaarFeature;

typedef struct CvHaarClassifier
{
	int count;
	CvHaarFeature* haar_feature;
	float* threshold;
	int* left;
	int* right;
	float* alpha;
} CvHaarClassifier;

typedef struct CvHaarStageClassifier
{
	int  count;
	float threshold;
	CvHaarClassifier* classifier;

	int next;
	int child;
	int parent;
} CvHaarStageClassifier;

typedef struct CvHidHaarClassifierCascade CvHidHaarClassifierCascade;

typedef struct CvHaarClassifierCascade
{
	int  flags;
	int  count;
	CvSize orig_window_size;
	CvSize real_window_size;
	double scale;
	CvHaarStageClassifier* stage_classifier;
	CvHidHaarClassifierCascade* hid_cascade;
} CvHaarClassifierCascade;

typedef struct CvAvgComp
{
	CvRect rect;
	int neighbors;
} CvAvgComp;


#define CV_HAAR_DO_CANNY_PRUNING    1
#define CV_HAAR_SCALE_IMAGE         2
#define CV_HAAR_FIND_BIGGEST_OBJECT 4
#define CV_HAAR_DO_ROUGH_SEARCH     8


/****************************************************************************************\
*                         objdetect.hpp                           *
\****************************************************************************************/
class CV_EXPORTS SimilarRects
{
public:
	SimilarRects(double _eps) : eps(_eps) {}
	inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) const
	{
		double delta = eps * ((std::min)(r1.width, r2.width) + (std::min)(r1.height, r2.height)) * 0.5;
		return std::abs(r1.x - r2.x) <= delta &&
			std::abs(r1.y - r2.y) <= delta &&
			std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
			std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
	}
	double eps;
};



/****************************************************************************************\
*                         cascadedetect.hpp                           *
\****************************************************************************************/

CvHaarClassifierCascade* load_haarcascade_frontalface_alt2();

void detectMultiScaleOldFormat( const cv::Mat& image, CvHaarClassifierCascade *oldCascade,
	std::vector<cv::Rect>& objects,
	std::vector<int>& rejectLevels,
	std::vector<double>& levelWeights,
	std::vector<CvAvgComp>& vecAvgComp,
	double scaleFactor, int minNeighbors,
	int flags, cv::Size minObjectSize, cv::Size maxObjectSize,
	bool outputRejectLevels = false );