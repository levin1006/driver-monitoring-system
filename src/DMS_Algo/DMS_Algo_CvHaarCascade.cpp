
#include "imgproc.hpp"
#include "core/private.hpp"
#include "opencv2/core/utility.hpp"

#include "DMS_Algo_CvHaarCascade.h"
#include "haarcascade_frontalface_alt2_rawdata.h"


CvSeq*
cvHaarDetectObjectsForROC( const CvArr* _img,
	CvHaarClassifierCascade* cascade, CvMemStorage* storage,
	std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
	double scaleFactor, int minNeighbors, int flags,
	CvSize minSize, CvSize maxSize, bool outputRejectLevels );



/****************************************************************************************\
*                         cascadedetect.cpp                           *
\****************************************************************************************/

void groupRectangles(std::vector<cv::Rect>& rectList, int groupThreshold, double eps,
	std::vector<int>* weights, std::vector<double>* levelWeights)
{

	if( groupThreshold <= 0 || rectList.empty() )
	{
		if( weights && !levelWeights )
		{
			size_t i, sz = rectList.size();
			weights->resize(sz);
			for( i = 0; i < sz; i++ )
				(*weights)[i] = 1;
		}
		return;
	}

	std::vector<int> labels;
	int nclasses = cv::partition(rectList, labels, SimilarRects(eps));

	std::vector<cv::Rect> rrects(nclasses);
	std::vector<int> rweights(nclasses, 0);
	std::vector<int> rejectLevels(nclasses, 0);
	std::vector<double> rejectWeights(nclasses, DBL_MIN);
	int i, j, nlabels = (int)labels.size();
	for( i = 0; i < nlabels; i++ )
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		rweights[cls]++;
	}

	bool useDefaultWeights = false;

	if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
	{
		for( i = 0; i < nlabels; i++ )
		{
			int cls = labels[i];
			if( (*weights)[i] > rejectLevels[cls] )
			{
				rejectLevels[cls] = (*weights)[i];
				rejectWeights[cls] = (*levelWeights)[i];
			}
			else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
				rejectWeights[cls] = (*levelWeights)[i];
		}
	}
	else
		useDefaultWeights = true;

	for( i = 0; i < nclasses; i++ )
	{
		cv::Rect r = rrects[i];
		float s = 1.f/rweights[i];
		rrects[i] = cv::Rect(cv::saturate_cast<int>(r.x*s),
			cv::saturate_cast<int>(r.y*s),
			cv::saturate_cast<int>(r.width*s),
			cv::saturate_cast<int>(r.height*s));
	}

	rectList.clear();
	if( weights )
		weights->clear();
	if( levelWeights )
		levelWeights->clear();

	for( i = 0; i < nclasses; i++ )
	{
		cv::Rect r1 = rrects[i];
		int n1 = rweights[i];
		double w1 = rejectWeights[i];
		int l1 = rejectLevels[i];

		// filter out rectangles which don't have enough similar rectangles
		if( n1 <= groupThreshold )
			continue;
		// filter out small face rectangles inside large rectangles
		for( j = 0; j < nclasses; j++ )
		{
			int n2 = rweights[j];

			if( j == i || n2 <= groupThreshold )
				continue;
			cv::Rect r2 = rrects[j];

			int dx = cv::saturate_cast<int>( r2.width * eps );
			int dy = cv::saturate_cast<int>( r2.height * eps );

			if( i != j &&
				r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3) )
				break;
		}

		if( j == nclasses )
		{
			rectList.push_back(r1);
			if( weights )
				weights->push_back(useDefaultWeights ? n1 : l1);
			if( levelWeights )
				levelWeights->push_back(w1);
		}
	}
}


void groupRectangles(std::vector<cv::Rect>& rectList, int groupThreshold, double eps)
{

	groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<int>& weights, int groupThreshold, double eps)
{

	groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(std::vector<cv::Rect>& rectList, std::vector<int>& rejectLevels,
	std::vector<double>& levelWeights, int groupThreshold, double eps)
{

	groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}


// DH 2018.04.19
// Manually load frontal face training data
CvHaarClassifierCascade* load_haarcascade_frontalface_alt2()
{
	int stage, tree, node, feat;
	int nStages = 20;
	int nTrees[20] = { 3, 9, 14, 19, 19, 27, 31, 39, 45, 47, 53, 67, 63, 71, 75, 78, 91, 97, 90, 109 };
	int nNodes = 2;	// all trees have two nodes
	int rawDataIdx = 0;

	CvHaarClassifierCascade* cascade = NULL;

	int block_size = sizeof(*cascade) + nStages * sizeof( *cascade->stage_classifier );

	cascade = (CvHaarClassifierCascade*)cvAlloc( block_size );
	memset( cascade, 0, block_size );

	cascade->stage_classifier = (CvHaarStageClassifier*)(cascade + 1);
	cascade->flags = CV_HAAR_MAGIC_VAL;
	cascade->count = nStages;


	// window size
	cascade->orig_window_size.width  = 20;
	cascade->orig_window_size.height = 20;


	// Reconstruct dataset structure
	for( int i = 0 ; i < nStages ; i++ )
	{
		cascade->stage_classifier[i].classifier = 
			(CvHaarClassifier*) cvAlloc( nTrees[i] 
				* sizeof( cascade->stage_classifier[i].classifier[0] ) );

		cascade->stage_classifier[i].count = nTrees[i];

		for( int j = 0; j < nTrees[i] ; j++ )
		{
			CvHaarClassifier* classifier = &cascade->stage_classifier[i].classifier[j];

			classifier->count = nNodes; 

			classifier->haar_feature = NULL;
			classifier->haar_feature = 
				(CvHaarFeature*) cvAlloc( nNodes * 
				( sizeof( *classifier->haar_feature ) +
					sizeof( *classifier->threshold ) +
					sizeof( *classifier->left ) +
					sizeof( *classifier->right ) ) +
					(nNodes + 1) * sizeof( *classifier->alpha ) );

			classifier->threshold = (float*) ( classifier->haar_feature + classifier->count );
			classifier->left	  = (int*)   ( classifier->threshold    + classifier->count );
			classifier->right	  = (int*)   ( classifier->left         + classifier->count );
			classifier->alpha	  = (float*) ( classifier->right        + classifier->count );



			for( int k = 0, last_idx = 0; k < nNodes ; k++ )
			{

				// find the number of features in the same node
				int cnt = 1;
				for( int l = rawDataIdx ; l < 4535 ; l++ )
				{
					if( l == 4534 )
						continue;
					else if( ( rawData[l][6] == rawData[l+1][6] ) && ( rawData[l][7] == rawData[l+1][7] ) )
						cnt++;
					else
						break;
				}

				for( int l = 0 ; l < cnt ; l++ )
				{
					CvRect r;

					r.x		 = rawData[rawDataIdx+l][0];
					r.y		 = rawData[rawDataIdx+l][1];
					r.width  = rawData[rawDataIdx+l][2];
					r.height = rawData[rawDataIdx+l][3];

					classifier->haar_feature[k].rect[l].weight = rawData[rawDataIdx+l][4];
					classifier->haar_feature[k].rect[l].r = r;

				}

				for( int l = cnt ; l < CV_HAAR_FEATURE_MAX; l++ )
				{
					classifier->haar_feature[k].rect[l].weight = 0;
					classifier->haar_feature[k].rect[l].r = cvRect( 0, 0, 0, 0 );
				}

				classifier->haar_feature[k].tilted = rawData[rawDataIdx][5];
				classifier->threshold[k] = rawData[rawDataIdx][6];

				int leftNode = rawData[rawDataIdx][7];
				if( leftNode )
					classifier->left[k] = leftNode;
				else
				{
					classifier->left[k] = -last_idx;
					classifier->alpha[last_idx++] = rawData[rawDataIdx][8];
				}

				int rightNode = rawData[rawDataIdx][9];
				if( rightNode )
					classifier->right[k] = rightNode;
				else
				{
					classifier->right[k] = -last_idx;
					classifier->alpha[last_idx++] = rawData[rawDataIdx][10];
				}

				rawDataIdx += cnt;
			}
		}

		cascade->stage_classifier[i].threshold = rawData[rawDataIdx-1][11];
		cascade->stage_classifier[i].parent    = rawData[rawDataIdx-1][12];
		cascade->stage_classifier[i].next      = rawData[rawDataIdx-1][13];
		cascade->stage_classifier[i].child     = -1;

		int parent = cascade->stage_classifier[i].parent;
		if( ( parent != -1 ) && ( cascade->stage_classifier[parent].child == -1 ) )
			cascade->stage_classifier[parent].child = i;

	}

	return cascade;
}


struct getRect { cv::Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

void detectMultiScaleOldFormat( const cv::Mat& image, CvHaarClassifierCascade *oldCascade,
	std::vector<cv::Rect>& objects,
	std::vector<int>& rejectLevels,
	std::vector<double>& levelWeights,
	std::vector<CvAvgComp>& vecAvgComp,
	double scaleFactor, int minNeighbors,
	int flags, cv::Size minObjectSize, cv::Size maxObjectSize,
	bool outputRejectLevels )
{
	cv::MemStorage storage(cvCreateMemStorage(0));
	CvMat _image = image;
	CvSeq* _objects =NULL;
	_objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
		minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
	cv::Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
	objects.resize(vecAvgComp.size());
	std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
}

/****************************************************************************************\
*                         haar.cpp                           *
\****************************************************************************************/

#if CV_SSE2
#   if 1 /*!CV_SSE4_1 && !CV_SSE4_2*/
#       define _mm_blendv_pd(a, b, m) _mm_xor_pd(a, _mm_and_pd(_mm_xor_pd(b, a), m))
#       define _mm_blendv_ps(a, b, m) _mm_xor_ps(a, _mm_and_ps(_mm_xor_ps(b, a), m))
#   endif
#endif

#if CV_HAAR_USE_AVX
#  if defined _MSC_VER
#    pragma warning( disable : 4752 )
#  endif
#else
#  if CV_SSE2
#    define CV_HAAR_USE_SSE 1
#  endif
#endif

/* these settings affect the quality of detection: change with care */
#define CV_ADJUST_FEATURES 1
#define CV_ADJUST_WEIGHTS  0

typedef struct CvHidHaarStageClassifier
{
	int  count;
	float threshold;
	CvHidHaarClassifier* classifier;
	int two_rects;

	struct CvHidHaarStageClassifier* next;
	struct CvHidHaarStageClassifier* child;
	struct CvHidHaarStageClassifier* parent;
} CvHidHaarStageClassifier;


typedef struct CvHidHaarClassifierCascade
{
	int  count;
	int  has_tilted_features;
	double inv_window_area;
	CvMat sum, sqsum, tilted;
	CvHidHaarStageClassifier* stage_classifier;
	sqsumtype *pq0, *pq1, *pq2, *pq3;
	sumtype *p0, *p1, *p2, *p3;

	void** ipp_stages;
	bool  is_tree;
	bool  isStumpBased;
} CvHidHaarClassifierCascade;


const int icv_object_win_border = 1;
const float icv_stage_threshold_bias = 0.0001f;

static CvHaarClassifierCascade*
icvCreateHaarClassifierCascade( int stage_count )
{
	CvHaarClassifierCascade* cascade = 0;

	int block_size = sizeof(*cascade) + stage_count*sizeof(*cascade->stage_classifier);

	if( stage_count <= 0 )
		CV_Error( CV_StsOutOfRange, "Number of stages should be positive" );

	cascade = (CvHaarClassifierCascade*)cvAlloc( block_size );
	memset( cascade, 0, block_size );

	cascade->stage_classifier = (CvHaarStageClassifier*)(cascade + 1);
	cascade->flags = CV_HAAR_MAGIC_VAL;
	cascade->count = stage_count;

	return cascade;
}

static void
icvReleaseHidHaarClassifierCascade( CvHidHaarClassifierCascade** _cascade )
{
	if( _cascade && *_cascade )
	{

		cvFree( _cascade );
	}
}

/* create more efficient internal representation of haar classifier cascade */
static CvHidHaarClassifierCascade*
icvCreateHidHaarClassifierCascade( CvHaarClassifierCascade* cascade )
{
	CvRect* ipp_features = 0;
	float *ipp_weights = 0, *ipp_thresholds = 0, *ipp_val1 = 0, *ipp_val2 = 0;
	int* ipp_counts = 0;

	CvHidHaarClassifierCascade* out = 0;

	int i, j, k, l;
	int datasize;
	int total_classifiers = 0;
	int total_nodes = 0;
	char errorstr[1000];
	CvHidHaarClassifier* haar_classifier_ptr;
	CvHidHaarTreeNode* haar_node_ptr;
	CvSize orig_window_size;
	bool has_tilted_features = false;
	int max_count = 0;

	if( !CV_IS_HAAR_CLASSIFIER(cascade) )
		CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

	if( cascade->hid_cascade )
		CV_Error( CV_StsError, "hid_cascade has been already created" );

	if( !cascade->stage_classifier )
		CV_Error( CV_StsNullPtr, "" );

	if( cascade->count <= 0 )
		CV_Error( CV_StsOutOfRange, "Negative number of cascade stages" );

	orig_window_size = cascade->orig_window_size;

	/* check input structure correctness and calculate total memory size needed for
	internal representation of the classifier cascade */
	for( i = 0; i < cascade->count; i++ )
	{
		CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;

		if( !stage_classifier->classifier ||
			stage_classifier->count <= 0 )
		{
			sprintf_s( errorstr, "header of the stage classifier #%d is invalid "
				"(has null pointers or non-positive classfier count)", i );
			CV_Error( CV_StsError, errorstr );
		}

		max_count = MAX( max_count, stage_classifier->count );
		total_classifiers += stage_classifier->count;

		for( j = 0; j < stage_classifier->count; j++ )
		{
			CvHaarClassifier* classifier = stage_classifier->classifier + j;

			total_nodes += classifier->count;
			for( l = 0; l < classifier->count; l++ )
			{
				for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
				{
					if( classifier->haar_feature[l].rect[k].r.width )
					{
						CvRect r = classifier->haar_feature[l].rect[k].r;
						int tilted = classifier->haar_feature[l].tilted;
						has_tilted_features = has_tilted_features | (tilted != 0);
						if( r.width < 0 || r.height < 0 || r.y < 0 ||
							r.x + r.width > orig_window_size.width
							||
							(!tilted &&
							(r.x < 0 || r.y + r.height > orig_window_size.height))
							||
							(tilted && (r.x - r.height < 0 ||
								r.y + r.width + r.height > orig_window_size.height)))
						{
							sprintf_s( errorstr, "rectangle #%d of the classifier #%d of "
								"the stage classifier #%d is not inside "
								"the reference (original) cascade window", k, j, i );
							CV_Error( CV_StsNullPtr, errorstr );
						}
					}
				}
			}
		}
	}

	// this is an upper boundary for the whole hidden cascade size
	datasize = sizeof(CvHidHaarClassifierCascade) +
		sizeof(CvHidHaarStageClassifier)*cascade->count +
		sizeof(CvHidHaarClassifier) * total_classifiers +
		sizeof(CvHidHaarTreeNode) * total_nodes +
		sizeof(void*)*(total_nodes + total_classifiers);

	out = (CvHidHaarClassifierCascade*)cvAlloc( datasize );
	memset( out, 0, sizeof(*out) );

	/* init header */
	out->count = cascade->count;
	out->stage_classifier = (CvHidHaarStageClassifier*)(out + 1);
	haar_classifier_ptr = (CvHidHaarClassifier*)(out->stage_classifier + cascade->count);
	haar_node_ptr = (CvHidHaarTreeNode*)(haar_classifier_ptr + total_classifiers);

	out->isStumpBased = true;
	out->has_tilted_features = has_tilted_features;
	out->is_tree = false;

	/* initialize internal representation */
	for( i = 0; i < cascade->count; i++ )
	{
		CvHaarStageClassifier* stage_classifier = cascade->stage_classifier + i;
		CvHidHaarStageClassifier* hid_stage_classifier = out->stage_classifier + i;

		hid_stage_classifier->count = stage_classifier->count;
		hid_stage_classifier->threshold = stage_classifier->threshold - icv_stage_threshold_bias;
		hid_stage_classifier->classifier = haar_classifier_ptr;
		hid_stage_classifier->two_rects = 1;
		haar_classifier_ptr += stage_classifier->count;

		hid_stage_classifier->parent = (stage_classifier->parent == -1)
			? NULL : out->stage_classifier + stage_classifier->parent;
		hid_stage_classifier->next = (stage_classifier->next == -1)
			? NULL : out->stage_classifier + stage_classifier->next;
		hid_stage_classifier->child = (stage_classifier->child == -1)
			? NULL : out->stage_classifier + stage_classifier->child;

		out->is_tree = out->is_tree || (hid_stage_classifier->next != NULL);

		for( j = 0; j < stage_classifier->count; j++ )
		{
			CvHaarClassifier* classifier = stage_classifier->classifier + j;
			CvHidHaarClassifier* hid_classifier = hid_stage_classifier->classifier + j;
			int node_count = classifier->count;
			float* alpha_ptr = (float*)(haar_node_ptr + node_count);

			hid_classifier->count = node_count;
			hid_classifier->node = haar_node_ptr;
			hid_classifier->alpha = alpha_ptr;

			for( l = 0; l < node_count; l++ )
			{
				CvHidHaarTreeNode* node = hid_classifier->node + l;
				CvHaarFeature* feature = classifier->haar_feature + l;
				memset( node, -1, sizeof(*node) );
				node->threshold = classifier->threshold[l];
				node->left = classifier->left[l];
				node->right = classifier->right[l];

				if( fabs(feature->rect[2].weight) < DBL_EPSILON ||
					feature->rect[2].r.width == 0 ||
					feature->rect[2].r.height == 0 )
					memset( &(node->feature.rect[2]), 0, sizeof(node->feature.rect[2]) );
				else
					hid_stage_classifier->two_rects = 0;
			}

			memcpy( alpha_ptr, classifier->alpha, (node_count+1)*sizeof(alpha_ptr[0]));
			haar_node_ptr =
				(CvHidHaarTreeNode*)cvAlignPtr(alpha_ptr+node_count+1, sizeof(void*));

			out->isStumpBased = out->isStumpBased && (node_count == 1);
		}
	}


	cascade->hid_cascade = out;
	assert( (char*)haar_node_ptr - (char*)out <= datasize );

	cvFree( &ipp_features );
	cvFree( &ipp_weights );
	cvFree( &ipp_thresholds );
	cvFree( &ipp_val1 );
	cvFree( &ipp_val2 );
	cvFree( &ipp_counts );

	return out;
}


#define sum_elem_ptr(sum,row,col)  \
    ((sumtype*)CV_MAT_ELEM_PTR_FAST((sum),(row),(col),sizeof(sumtype)))

#define sqsum_elem_ptr(sqsum,row,col)  \
    ((sqsumtype*)CV_MAT_ELEM_PTR_FAST((sqsum),(row),(col),sizeof(sqsumtype)))

#define calc_sum(rect,offset) \
    ((rect).p0[offset] - (rect).p1[offset] - (rect).p2[offset] + (rect).p3[offset])

CV_IMPL void
cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* _cascade,
	const CvArr* _sum,
	const CvArr* _sqsum,
	const CvArr* _tilted_sum,
	double scale )
{
	CvMat sum_stub, *sum = (CvMat*)_sum;
	CvMat sqsum_stub, *sqsum = (CvMat*)_sqsum;
	CvMat tilted_stub, *tilted = (CvMat*)_tilted_sum;
	CvHidHaarClassifierCascade* cascade;
	int coi0 = 0, coi1 = 0;
	int i;
	CvRect equRect;
	double weight_scale;

	if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
		CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier pointer" );

	if( scale <= 0 )
		CV_Error( CV_StsOutOfRange, "Scale must be positive" );

	sum = cvGetMat( sum, &sum_stub, &coi0 );
	sqsum = cvGetMat( sqsum, &sqsum_stub, &coi1 );

	if( coi0 || coi1 )
		CV_Error( CV_BadCOI, "COI is not supported" );

	if( !CV_ARE_SIZES_EQ( sum, sqsum ))
		CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );

	if( CV_MAT_TYPE(sqsum->type) != CV_64FC1 ||
		CV_MAT_TYPE(sum->type) != CV_32SC1 )
		CV_Error( CV_StsUnsupportedFormat,
			"Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

	if( !_cascade->hid_cascade )
		icvCreateHidHaarClassifierCascade(_cascade);

	cascade = _cascade->hid_cascade;

	if( cascade->has_tilted_features )
	{
		tilted = cvGetMat( tilted, &tilted_stub, &coi1 );

		if( CV_MAT_TYPE(tilted->type) != CV_32SC1 )
			CV_Error( CV_StsUnsupportedFormat,
				"Only (32s, 64f, 32s) combination of (sum,sqsum,tilted_sum) formats is allowed" );

		if( sum->step != tilted->step )
			CV_Error( CV_StsUnmatchedSizes,
				"Sum and tilted_sum must have the same stride (step, widthStep)" );

		if( !CV_ARE_SIZES_EQ( sum, tilted ))
			CV_Error( CV_StsUnmatchedSizes, "All integral images must have the same size" );
		cascade->tilted = *tilted;
	}

	_cascade->scale = scale;
	_cascade->real_window_size.width = cvRound( _cascade->orig_window_size.width * scale );
	_cascade->real_window_size.height = cvRound( _cascade->orig_window_size.height * scale );

	cascade->sum = *sum;
	cascade->sqsum = *sqsum;

	equRect.x = equRect.y = cvRound(scale);
	equRect.width = cvRound((_cascade->orig_window_size.width-2)*scale);
	equRect.height = cvRound((_cascade->orig_window_size.height-2)*scale);
	weight_scale = 1./(equRect.width*equRect.height);
	cascade->inv_window_area = weight_scale;

	cascade->p0 = sum_elem_ptr(*sum, equRect.y, equRect.x);
	cascade->p1 = sum_elem_ptr(*sum, equRect.y, equRect.x + equRect.width );
	cascade->p2 = sum_elem_ptr(*sum, equRect.y + equRect.height, equRect.x );
	cascade->p3 = sum_elem_ptr(*sum, equRect.y + equRect.height,
		equRect.x + equRect.width );

	cascade->pq0 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x);
	cascade->pq1 = sqsum_elem_ptr(*sqsum, equRect.y, equRect.x + equRect.width );
	cascade->pq2 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height, equRect.x );
	cascade->pq3 = sqsum_elem_ptr(*sqsum, equRect.y + equRect.height,
		equRect.x + equRect.width );

	/* init pointers in haar features according to real window size and
	given image pointers */
	for( i = 0; i < _cascade->count; i++ )
	{
		int j, k, l;
		for( j = 0; j < cascade->stage_classifier[i].count; j++ )
		{
			for( l = 0; l < cascade->stage_classifier[i].classifier[j].count; l++ )
			{
				CvHaarFeature* feature =
					&_cascade->stage_classifier[i].classifier[j].haar_feature[l];
				/* CvHidHaarClassifier* classifier =
				cascade->stage_classifier[i].classifier + j; */
				CvHidHaarFeature* hidfeature =
					&cascade->stage_classifier[i].classifier[j].node[l].feature;
				double sum0 = 0, area0 = 0;
				CvRect r[3];

				int base_w = -1, base_h = -1;
				int new_base_w = 0, new_base_h = 0;
				int kx, ky;
				int flagx = 0, flagy = 0;
				int x0 = 0, y0 = 0;
				int nr;

				/* align blocks */
				for( k = 0; k < CV_HAAR_FEATURE_MAX; k++ )
				{
					if( !hidfeature->rect[k].p0 )
						break;
					r[k] = feature->rect[k].r;
					base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].width-1) );
					base_w = (int)CV_IMIN( (unsigned)base_w, (unsigned)(r[k].x - r[0].x-1) );
					base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].height-1) );
					base_h = (int)CV_IMIN( (unsigned)base_h, (unsigned)(r[k].y - r[0].y-1) );
				}

				nr = k;

				base_w += 1;
				base_h += 1;
				kx = r[0].width / base_w;
				ky = r[0].height / base_h;

				if( kx <= 0 )
				{
					flagx = 1;
					new_base_w = cvRound( r[0].width * scale ) / kx;
					x0 = cvRound( r[0].x * scale );
				}

				if( ky <= 0 )
				{
					flagy = 1;
					new_base_h = cvRound( r[0].height * scale ) / ky;
					y0 = cvRound( r[0].y * scale );
				}

				for( k = 0; k < nr; k++ )
				{
					CvRect tr;
					double correction_ratio;

					if( flagx )
					{
						tr.x = (r[k].x - r[0].x) * new_base_w / base_w + x0;
						tr.width = r[k].width * new_base_w / base_w;
					}
					else
					{
						tr.x = cvRound( r[k].x * scale );
						tr.width = cvRound( r[k].width * scale );
					}

					if( flagy )
					{
						tr.y = (r[k].y - r[0].y) * new_base_h / base_h + y0;
						tr.height = r[k].height * new_base_h / base_h;
					}
					else
					{
						tr.y = cvRound( r[k].y * scale );
						tr.height = cvRound( r[k].height * scale );
					}

#if CV_ADJUST_WEIGHTS
					{
						// RAINER START
						const float orig_feature_size =  (float)(feature->rect[k].r.width)*feature->rect[k].r.height;
						const float orig_norm_size = (float)(_cascade->orig_window_size.width)*(_cascade->orig_window_size.height);
						const float feature_size = float(tr.width*tr.height);
						//const float normSize    = float(equRect.width*equRect.height);
						float target_ratio = orig_feature_size / orig_norm_size;
						//float isRatio = featureSize / normSize;
						//correctionRatio = targetRatio / isRatio / normSize;
						correction_ratio = target_ratio / feature_size;
						// RAINER END
					}
#else
					correction_ratio = weight_scale * (!feature->tilted ? 1 : 0.5);
#endif

					if( !feature->tilted )
					{
						hidfeature->rect[k].p0 = sum_elem_ptr(*sum, tr.y, tr.x);
						hidfeature->rect[k].p1 = sum_elem_ptr(*sum, tr.y, tr.x + tr.width);
						hidfeature->rect[k].p2 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x);
						hidfeature->rect[k].p3 = sum_elem_ptr(*sum, tr.y + tr.height, tr.x + tr.width);
					}
					else
					{
						hidfeature->rect[k].p2 = sum_elem_ptr(*tilted, tr.y + tr.width, tr.x + tr.width);
						hidfeature->rect[k].p3 = sum_elem_ptr(*tilted, tr.y + tr.width + tr.height,
							tr.x + tr.width - tr.height);
						hidfeature->rect[k].p0 = sum_elem_ptr(*tilted, tr.y, tr.x);
						hidfeature->rect[k].p1 = sum_elem_ptr(*tilted, tr.y + tr.height, tr.x - tr.height);
					}

					hidfeature->rect[k].weight = (float)(feature->rect[k].weight * correction_ratio);

					if( k == 0 )
						area0 = tr.width * tr.height;
					else
						sum0 += hidfeature->rect[k].weight * tr.width * tr.height;
				}

				hidfeature->rect[0].weight = (float)(-sum0/area0);
			} /* l */
		} /* j */
	}
}


CV_INLINE
double icvEvalHidHaarClassifier( CvHidHaarClassifier* classifier,
	double variance_norm_factor,
	size_t p_offset )
{
	int idx = 0;
	{
		do
		{
			CvHidHaarTreeNode* node = classifier->node + idx;
			double t = node->threshold * variance_norm_factor;

			double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
			sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;

			if( node->feature.rect[2].p0 )
				sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;

			idx = sum < t ? node->left : node->right;
		}
		while( idx > 0 );
	}
	return classifier->alpha[-idx];
}



static int
cvRunHaarClassifierCascadeSum( const CvHaarClassifierCascade* _cascade,
	CvPoint pt, double& stage_sum, int start_stage )
{
	int p_offset, pq_offset;
	int i, j;
	double mean, variance_norm_factor;
	CvHidHaarClassifierCascade* cascade;

	if( !CV_IS_HAAR_CLASSIFIER(_cascade) )
		CV_Error( !_cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid cascade pointer" );

	cascade = _cascade->hid_cascade;
	if( !cascade )
		CV_Error( CV_StsNullPtr, "Hidden cascade has not been created.\n"
			"Use cvSetImagesForHaarClassifierCascade" );

	if( pt.x < 0 || pt.y < 0 ||
		pt.x + _cascade->real_window_size.width >= cascade->sum.width ||
		pt.y + _cascade->real_window_size.height >= cascade->sum.height )
		return -1;

	p_offset = pt.y * (cascade->sum.step/sizeof(sumtype)) + pt.x;
	pq_offset = pt.y * (cascade->sqsum.step/sizeof(sqsumtype)) + pt.x;
	mean = calc_sum(*cascade,p_offset)*cascade->inv_window_area;
	variance_norm_factor = cascade->pq0[pq_offset] - cascade->pq1[pq_offset] -
		cascade->pq2[pq_offset] + cascade->pq3[pq_offset];
	variance_norm_factor = variance_norm_factor*cascade->inv_window_area - mean*mean;
	if( variance_norm_factor >= 0. )
		variance_norm_factor = std::sqrt(variance_norm_factor);
	else
		variance_norm_factor = 1.;

	if( cascade->is_tree )
	{
		CvHidHaarStageClassifier* ptr = cascade->stage_classifier;
		assert( start_stage == 0 );

		while( ptr )
		{
			stage_sum = 0.0;
			j = 0;
			for( ; j < ptr->count; j++ )
			{
				stage_sum += icvEvalHidHaarClassifier( ptr->classifier + j, variance_norm_factor, p_offset );
			}

			if( stage_sum >= ptr->threshold )
			{
				ptr = ptr->child;
			}
			else
			{
				while( ptr && ptr->next == NULL ) ptr = ptr->parent;
				if( ptr == NULL )
					return 0;
				ptr = ptr->next;
			}
		}
	}
	else if( cascade->isStumpBased )
	{
		for( i = start_stage; i < cascade->count; i++ )
		{
			stage_sum = 0.0;
			if( cascade->stage_classifier[i].two_rects )
			{
				for( j = 0; j < cascade->stage_classifier[i].count; j++ )
				{
					CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
					CvHidHaarTreeNode* node = classifier->node;
					double t = node->threshold*variance_norm_factor;
					double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
					sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
					stage_sum += classifier->alpha[sum >= t];
				}
			}
			else
			{
				for( j = 0; j < cascade->stage_classifier[i].count; j++ )
				{
					CvHidHaarClassifier* classifier = cascade->stage_classifier[i].classifier + j;
					CvHidHaarTreeNode* node = classifier->node;
					double t = node->threshold*variance_norm_factor;
					double sum = calc_sum(node->feature.rect[0],p_offset) * node->feature.rect[0].weight;
					sum += calc_sum(node->feature.rect[1],p_offset) * node->feature.rect[1].weight;
					if( node->feature.rect[2].p0 )
						sum += calc_sum(node->feature.rect[2],p_offset) * node->feature.rect[2].weight;
					stage_sum += classifier->alpha[sum >= t];
				}
			}
			if( stage_sum < cascade->stage_classifier[i].threshold )
				return -i;
		}
	}
	else
	{
		for( i = start_stage; i < cascade->count; i++ )
		{
			stage_sum = 0.0;
			int k = 0;
			for(; k < cascade->stage_classifier[i].count; k++ )
			{

				stage_sum += icvEvalHidHaarClassifier(
					cascade->stage_classifier[i].classifier + k,
					variance_norm_factor, p_offset );
			}

			if( stage_sum < cascade->stage_classifier[i].threshold )
				return -i;
		}
	}
	return 1;
}


CV_IMPL int
cvRunHaarClassifierCascade( const CvHaarClassifierCascade* _cascade,
	CvPoint pt, int start_stage )
{
	double stage_sum;
	return cvRunHaarClassifierCascadeSum(_cascade, pt, stage_sum, start_stage);
}

namespace cv
{

	const size_t PARALLEL_LOOP_BATCH_SIZE = 100;

	class HaarDetectObjects_ScaleImage_Invoker : public ParallelLoopBody
	{
	public:
		HaarDetectObjects_ScaleImage_Invoker( const CvHaarClassifierCascade* _cascade,
			int _stripSize, double _factor,
			const Mat& _sum1, const Mat& _sqsum1, Mat* _norm1,
			Mat* _mask1, Rect _equRect, std::vector<Rect>& _vec,
			std::vector<int>& _levels, std::vector<double>& _weights,
			bool _outputLevels, Mutex *_mtx )
		{
			cascade = _cascade;
			stripSize = _stripSize;
			factor = _factor;
			sum1 = _sum1;
			sqsum1 = _sqsum1;
			norm1 = _norm1;
			mask1 = _mask1;
			equRect = _equRect;
			vec = &_vec;
			rejectLevels = _outputLevels ? &_levels : 0;
			levelWeights = _outputLevels ? &_weights : 0;
			mtx = _mtx;
		}

		void operator()( const Range& range ) const
		{
			Size winSize0 = cascade->orig_window_size;
			Size winSize(cvRound(winSize0.width*factor), cvRound(winSize0.height*factor));
			int y1 = range.start*stripSize, y2 = std::min(range.end*stripSize, sum1.rows - 1 - winSize0.height);

			if (y2 <= y1 || sum1.cols <= 1 + winSize0.width)
				return;

			Size ssz(sum1.cols - 1 - winSize0.width, y2 - y1);
			int x, y, ystep = factor > 2 ? 1 : 2;

			std::vector<Rect> vecLocal;
			std::vector<int> rejectLevelsLocal;
			std::vector<double> levelWeightsLocal;

			for( y = y1; y < y2; y += ystep )
				for( x = 0; x < ssz.width; x += ystep )
				{
					double gypWeight;
					int result = cvRunHaarClassifierCascadeSum( cascade, cvPoint(x,y), gypWeight, 0 );
					if( rejectLevels )
					{
						if( result == 1 )
							result = -1*cascade->count;
						if( cascade->count + result < 4 )
						{
							vecLocal.push_back(Rect(cvRound(x*factor), cvRound(y*factor),
								winSize.width, winSize.height));
							rejectLevelsLocal.push_back(-result);
							levelWeightsLocal.push_back(gypWeight);

							if (vecLocal.size() >= PARALLEL_LOOP_BATCH_SIZE)
							{
								mtx->lock();
								vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
								rejectLevels->insert(rejectLevels->end(), rejectLevelsLocal.begin(), rejectLevelsLocal.end());
								levelWeights->insert(levelWeights->end(), levelWeightsLocal.begin(), levelWeightsLocal.end());
								mtx->unlock();

								vecLocal.clear();
								rejectLevelsLocal.clear();
								levelWeightsLocal.clear();
							}
						}
					}
					else
					{
						if( result > 0 )
						{
							vecLocal.push_back(Rect(cvRound(x*factor), cvRound(y*factor),
								winSize.width, winSize.height));

							if (vecLocal.size() >= PARALLEL_LOOP_BATCH_SIZE)
							{
								mtx->lock();
								vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
								mtx->unlock();

								vecLocal.clear();
							}
						}
					}
				}

			if (rejectLevelsLocal.size())
			{
				mtx->lock();
				vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
				rejectLevels->insert(rejectLevels->end(), rejectLevelsLocal.begin(), rejectLevelsLocal.end());
				levelWeights->insert(levelWeights->end(), levelWeightsLocal.begin(), levelWeightsLocal.end());
				mtx->unlock();
			}
			else
				if (vecLocal.size())
				{
					mtx->lock();
					vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
					mtx->unlock();
				}
		}

		const CvHaarClassifierCascade* cascade;
		int stripSize;
		double factor;
		Mat sum1, sqsum1, *norm1, *mask1;
		Rect equRect;
		std::vector<Rect>* vec;
		std::vector<int>* rejectLevels;
		std::vector<double>* levelWeights;
		Mutex* mtx;
	};


	class HaarDetectObjects_ScaleCascade_Invoker : public ParallelLoopBody
	{
	public:
		HaarDetectObjects_ScaleCascade_Invoker( const CvHaarClassifierCascade* _cascade,
			Size _winsize, const Range& _xrange, double _ystep,
			size_t _sumstep, const int** _p, const int** _pq,
			std::vector<Rect>& _vec, Mutex* _mtx )
		{
			cascade = _cascade;
			winsize = _winsize;
			xrange = _xrange;
			ystep = _ystep;
			sumstep = _sumstep;
			p = _p; pq = _pq;
			vec = &_vec;
			mtx = _mtx;
		}

		void operator()( const Range& range ) const
		{
			int iy, startY = range.start, endY = range.end;
			const int *p0 = p[0], *p1 = p[1], *p2 = p[2], *p3 = p[3];
			const int *pq0 = pq[0], *pq1 = pq[1], *pq2 = pq[2], *pq3 = pq[3];
			bool doCannyPruning = p0 != 0;
			int sstep = (int)(sumstep/sizeof(p0[0]));

			std::vector<Rect> vecLocal;

			for( iy = startY; iy < endY; iy++ )
			{
				int ix, y = cvRound(iy*ystep), ixstep = 1;
				for( ix = xrange.start; ix < xrange.end; ix += ixstep )
				{
					int x = cvRound(ix*ystep); // it should really be ystep, not ixstep

					if( doCannyPruning )
					{
						int offset = y*sstep + x;
						int s = p0[offset] - p1[offset] - p2[offset] + p3[offset];
						int sq = pq0[offset] - pq1[offset] - pq2[offset] + pq3[offset];
						if( s < 100 || sq < 20 )
						{
							ixstep = 2;
							continue;
						}
					}

					int result = cvRunHaarClassifierCascade( cascade, cvPoint(x, y), 0 );
					if( result > 0 )
					{
						vecLocal.push_back(Rect(x, y, winsize.width, winsize.height));

						if (vecLocal.size() >= PARALLEL_LOOP_BATCH_SIZE)
						{
							mtx->lock();
							vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
							mtx->unlock();

							vecLocal.clear();
						}
					}
					ixstep = result != 0 ? 1 : 2;
				}
			}

			if (vecLocal.size())
			{
				mtx->lock();
				vec->insert(vec->end(), vecLocal.begin(), vecLocal.end());
				mtx->unlock();
			}
		}

		const CvHaarClassifierCascade* cascade;
		double ystep;
		size_t sumstep;
		Size winsize;
		Range xrange;
		const int** p;
		const int** pq;
		std::vector<Rect>* vec;
		Mutex* mtx;
	};


}


CvSeq*
cvHaarDetectObjectsForROC( const CvArr* _img,
	CvHaarClassifierCascade* cascade, CvMemStorage* storage,
	std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
	double scaleFactor, int minNeighbors, int flags,
	CvSize minSize, CvSize maxSize, bool outputRejectLevels )
{
	const double GROUP_EPS = 0.2;
	CvMat stub, *img = (CvMat*)_img;
	cv::Ptr<CvMat> temp, sum, tilted, sqsum, normImg, sumcanny, imgSmall;
	CvSeq* result_seq = 0;
	cv::Ptr<CvMemStorage> temp_storage;

	std::vector<cv::Rect> allCandidates;
	std::vector<cv::Rect> rectList;
	std::vector<int> rweights;
	double factor;
	int coi;
	bool doCannyPruning = (flags & CV_HAAR_DO_CANNY_PRUNING) != 0;
	bool findBiggestObject = (flags & CV_HAAR_FIND_BIGGEST_OBJECT) != 0;
	bool roughSearch = (flags & CV_HAAR_DO_ROUGH_SEARCH) != 0;
	cv::Mutex mtx;

	if( !CV_IS_HAAR_CLASSIFIER(cascade) )
		CV_Error( !cascade ? CV_StsNullPtr : CV_StsBadArg, "Invalid classifier cascade" );

	if( !storage )
		CV_Error( CV_StsNullPtr, "Null storage pointer" );

	img = cvGetMat( img, &stub, &coi );
	if( coi )
		CV_Error( CV_BadCOI, "COI is not supported" );

	if( CV_MAT_DEPTH(img->type) != CV_8U )
		CV_Error( CV_StsUnsupportedFormat, "Only 8-bit images are supported" );

	if( scaleFactor <= 1 )
		CV_Error( CV_StsOutOfRange, "scale factor must be > 1" );

	if( findBiggestObject )
		flags &= ~CV_HAAR_SCALE_IMAGE;

	if( maxSize.height == 0 || maxSize.width == 0 )
	{
		maxSize.height = img->rows;
		maxSize.width = img->cols;
	}

	temp.reset(cvCreateMat( img->rows, img->cols, CV_8UC1 ));
	sum.reset(cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));
	sqsum.reset(cvCreateMat( img->rows + 1, img->cols + 1, CV_64FC1 ));

	if( !cascade->hid_cascade )
		icvCreateHidHaarClassifierCascade(cascade);

	if( cascade->hid_cascade->has_tilted_features )
		tilted.reset(cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));

	result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvAvgComp), storage );

	if( CV_MAT_CN(img->type) > 1 )
	{
		cvCvtColor( img, temp, CV_BGR2GRAY );
		img = temp;
	}

	if( findBiggestObject )
		flags &= ~(CV_HAAR_SCALE_IMAGE|CV_HAAR_DO_CANNY_PRUNING);

	if( flags & CV_HAAR_SCALE_IMAGE )
	{
		CvSize winSize0 = cascade->orig_window_size;

		imgSmall.reset(cvCreateMat( img->rows + 1, img->cols + 1, CV_8UC1 ));

		for( factor = 1; ; factor *= scaleFactor )
		{
			CvSize winSize(cvRound(winSize0.width*factor),
				cvRound(winSize0.height*factor));
			CvSize sz(cvRound( img->cols/factor ), cvRound( img->rows/factor ));
			CvSize sz1(sz.width - winSize0.width + 1, sz.height - winSize0.height + 1);

			CvRect equRect(icv_object_win_border, icv_object_win_border,
				winSize0.width - icv_object_win_border*2,
				winSize0.height - icv_object_win_border*2);

			CvMat img1, sum1, sqsum1, norm1, tilted1, mask1;
			CvMat* _tilted = 0;

			if( sz1.width <= 0 || sz1.height <= 0 )
				break;
			if( winSize.width > maxSize.width || winSize.height > maxSize.height )
				break;
			if( winSize.width < minSize.width || winSize.height < minSize.height )
				continue;

			img1 = cvMat( sz.height, sz.width, CV_8UC1, imgSmall->data.ptr );
			sum1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, sum->data.ptr );
			sqsum1 = cvMat( sz.height+1, sz.width+1, CV_64FC1, sqsum->data.ptr );
			if( tilted )
			{
				tilted1 = cvMat( sz.height+1, sz.width+1, CV_32SC1, tilted->data.ptr );
				_tilted = &tilted1;
			}
			norm1 = cvMat( sz1.height, sz1.width, CV_32FC1, normImg ? normImg->data.ptr : 0 );
			mask1 = cvMat( sz1.height, sz1.width, CV_8UC1, temp->data.ptr );

			cvResize( img, &img1, CV_INTER_LINEAR );
			cvIntegral( &img1, &sum1, &sqsum1, _tilted );

			int ystep = factor > 2 ? 1 : 2;
			const int LOCS_PER_THREAD = 1000;
			int stripCount = ((sz1.width/ystep)*(sz1.height + ystep-1)/ystep + LOCS_PER_THREAD/2)/LOCS_PER_THREAD;
			stripCount = std::min(std::max(stripCount, 1), 100);


			cvSetImagesForHaarClassifierCascade( cascade, &sum1, &sqsum1, _tilted, 1. );

			cv::Mat _norm1 = cv::cvarrToMat(&norm1), _mask1 = cv::cvarrToMat(&mask1);
			cv::parallel_for_(cv::Range(0, stripCount),
				cv::HaarDetectObjects_ScaleImage_Invoker(cascade,
				(((sz1.height + stripCount - 1)/stripCount + ystep-1)/ystep)*ystep,
					factor, cv::cvarrToMat(&sum1), cv::cvarrToMat(&sqsum1), &_norm1, &_mask1,
					cv::Rect(equRect), allCandidates, rejectLevels, levelWeights, outputRejectLevels, &mtx));
		}
	}
	else
	{
		int n_factors = 0;
		cv::Rect scanROI;

		cvIntegral( img, sum, sqsum, tilted );

		if( doCannyPruning )
		{
			sumcanny.reset(cvCreateMat( img->rows + 1, img->cols + 1, CV_32SC1 ));
			cvCanny( img, temp, 0, 50, 3 );
			cvIntegral( temp, sumcanny );
		}

		for( n_factors = 0, factor = 1;
			factor*cascade->orig_window_size.width < img->cols - 10 &&
			factor*cascade->orig_window_size.height < img->rows - 10;
			n_factors++, factor *= scaleFactor )
			;

		if( findBiggestObject )
		{
			scaleFactor = 1./scaleFactor;
			factor *= scaleFactor;
		}
		else
			factor = 1;

		for( ; n_factors-- > 0; factor *= scaleFactor )
		{
			const double ystep = std::max( 2., factor );
			CvSize winSize(cvRound( cascade->orig_window_size.width * factor ),
				cvRound( cascade->orig_window_size.height * factor ));
			CvRect equRect;
			int *p[4] = {0,0,0,0};
			int *pq[4] = {0,0,0,0};
			int startX = 0, startY = 0;
			int endX = cvRound((img->cols - winSize.width) / ystep);
			int endY = cvRound((img->rows - winSize.height) / ystep);

			if( winSize.width < minSize.width || winSize.height < minSize.height )
			{
				if( findBiggestObject )
					break;
				continue;
			}

			if ( winSize.width > maxSize.width || winSize.height > maxSize.height )
			{
				if( !findBiggestObject )
					break;
				continue;
			}

			cvSetImagesForHaarClassifierCascade( cascade, sum, sqsum, tilted, factor );
			cvZero( temp );

			if( doCannyPruning )
			{
				equRect.x = cvRound(winSize.width*0.15);
				equRect.y = cvRound(winSize.height*0.15);
				equRect.width = cvRound(winSize.width*0.7);
				equRect.height = cvRound(winSize.height*0.7);

				p[0] = (int*)(sumcanny->data.ptr + equRect.y*sumcanny->step) + equRect.x;
				p[1] = (int*)(sumcanny->data.ptr + equRect.y*sumcanny->step)
					+ equRect.x + equRect.width;
				p[2] = (int*)(sumcanny->data.ptr + (equRect.y + equRect.height)*sumcanny->step) + equRect.x;
				p[3] = (int*)(sumcanny->data.ptr + (equRect.y + equRect.height)*sumcanny->step)
					+ equRect.x + equRect.width;

				pq[0] = (int*)(sum->data.ptr + equRect.y*sum->step) + equRect.x;
				pq[1] = (int*)(sum->data.ptr + equRect.y*sum->step)
					+ equRect.x + equRect.width;
				pq[2] = (int*)(sum->data.ptr + (equRect.y + equRect.height)*sum->step) + equRect.x;
				pq[3] = (int*)(sum->data.ptr + (equRect.y + equRect.height)*sum->step)
					+ equRect.x + equRect.width;
			}

			if( scanROI.area() > 0 )
			{
				//adjust start_height and stop_height
				startY = cvRound(scanROI.y / ystep);
				endY = cvRound((scanROI.y + scanROI.height - winSize.height) / ystep);

				startX = cvRound(scanROI.x / ystep);
				endX = cvRound((scanROI.x + scanROI.width - winSize.width) / ystep);
			}

			cv::parallel_for_(cv::Range(startY, endY),
				cv::HaarDetectObjects_ScaleCascade_Invoker(cascade, winSize, cv::Range(startX, endX),
					ystep, sum->step, (const int**)p,
					(const int**)pq, allCandidates, &mtx ));

			if( findBiggestObject && !allCandidates.empty() && scanROI.area() == 0 )
			{
				rectList.resize(allCandidates.size());
				std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

				groupRectangles(rectList, std::max(minNeighbors, 1), GROUP_EPS);

				if( !rectList.empty() )
				{
					size_t i, sz = rectList.size();
					cv::Rect maxRect;

					for( i = 0; i < sz; i++ )
					{
						if( rectList[i].area() > maxRect.area() )
							maxRect = rectList[i];
					}

					allCandidates.push_back(maxRect);

					scanROI = maxRect;
					int dx = cvRound(maxRect.width*GROUP_EPS);
					int dy = cvRound(maxRect.height*GROUP_EPS);
					scanROI.x = std::max(scanROI.x - dx, 0);
					scanROI.y = std::max(scanROI.y - dy, 0);
					scanROI.width = std::min(scanROI.width + dx*2, img->cols-1-scanROI.x);
					scanROI.height = std::min(scanROI.height + dy*2, img->rows-1-scanROI.y);

					double minScale = roughSearch ? 0.6 : 0.4;
					minSize.width = cvRound(maxRect.width*minScale);
					minSize.height = cvRound(maxRect.height*minScale);
				}
			}
		}
	}

	rectList.resize(allCandidates.size());
	if(!allCandidates.empty())
		std::copy(allCandidates.begin(), allCandidates.end(), rectList.begin());

	if( minNeighbors != 0 || findBiggestObject )
	{
		if( outputRejectLevels )
		{
			groupRectangles(rectList, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
		}
		else
		{
			groupRectangles(rectList, rweights, std::max(minNeighbors, 1), GROUP_EPS);
		}
	}
	else
		rweights.resize(rectList.size(),0);

	if( findBiggestObject && rectList.size() )
	{
		CvAvgComp result_comp = {CvRect(),0};

		for( size_t i = 0; i < rectList.size(); i++ )
		{
			cv::Rect r = rectList[i];
			if( r.area() > cv::Rect(result_comp.rect).area() )
			{
				result_comp.rect = r;
				result_comp.neighbors = rweights[i];
			}
		}
		cvSeqPush( result_seq, &result_comp );
	}
	else
	{
		for( size_t i = 0; i < rectList.size(); i++ )
		{
			CvAvgComp c;
			c.rect = rectList[i];
			c.neighbors = !rweights.empty() ? rweights[i] : 0;
			cvSeqPush( result_seq, &c );
		}
	}

	return result_seq;
}




CV_IMPL void
cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** _cascade )
{
	if( _cascade && *_cascade )
	{
		int i, j;
		CvHaarClassifierCascade* cascade = *_cascade;

		for( i = 0; i < cascade->count; i++ )
		{
			for( j = 0; j < cascade->stage_classifier[i].count; j++ )
				cvFree( &cascade->stage_classifier[i].classifier[j].haar_feature );
			cvFree( &cascade->stage_classifier[i].classifier );
		}
		icvReleaseHidHaarClassifierCascade( &cascade->hid_cascade );
		cvFree( _cascade );
	}
}


/****************************************************************************************\
*                                  Persistence functions                                 *
\****************************************************************************************/

/* field names */

#define ICV_HAAR_SIZE_NAME            "size"
#define ICV_HAAR_STAGES_NAME          "stages"
#define ICV_HAAR_TREES_NAME           "trees"
#define ICV_HAAR_FEATURE_NAME         "feature"
#define ICV_HAAR_RECTS_NAME           "rects"
#define ICV_HAAR_TILTED_NAME          "tilted"
#define ICV_HAAR_THRESHOLD_NAME       "threshold"
#define ICV_HAAR_LEFT_NODE_NAME       "left_node"
#define ICV_HAAR_LEFT_VAL_NAME        "left_val"
#define ICV_HAAR_RIGHT_NODE_NAME      "right_node"
#define ICV_HAAR_RIGHT_VAL_NAME       "right_val"
#define ICV_HAAR_STAGE_THRESHOLD_NAME "stage_threshold"
#define ICV_HAAR_PARENT_NAME          "parent"
#define ICV_HAAR_NEXT_NAME            "next"

static int
icvIsHaarClassifier( const void* struct_ptr )
{
	return CV_IS_HAAR_CLASSIFIER( struct_ptr );
}

static void*
icvReadHaarClassifier( CvFileStorage* fs, CvFileNode* node )
{
	CvHaarClassifierCascade* cascade = NULL;

	char buf[256];
	CvFileNode* seq_fn = NULL; /* sequence */
	CvFileNode* fn = NULL;
	CvFileNode* stages_fn = NULL;
	CvSeqReader stages_reader;
	int n;
	int i, j, k, l;
	int parent, next;

	stages_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_STAGES_NAME );
	if( !stages_fn || !CV_NODE_IS_SEQ( stages_fn->tag) )
		CV_Error( CV_StsError, "Invalid stages node" );

	n = stages_fn->data.seq->total;
	cascade = icvCreateHaarClassifierCascade(n);

	/* read size */
	seq_fn = cvGetFileNodeByName( fs, node, ICV_HAAR_SIZE_NAME );
	if( !seq_fn || !CV_NODE_IS_SEQ( seq_fn->tag ) || seq_fn->data.seq->total != 2 )
		CV_Error( CV_StsError, "size node is not a valid sequence." );
	fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 0 );
	if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
		CV_Error( CV_StsError, "Invalid size node: width must be positive integer" );
	cascade->orig_window_size.width = fn->data.i;
	fn = (CvFileNode*) cvGetSeqElem( seq_fn->data.seq, 1 );
	if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0 )
		CV_Error( CV_StsError, "Invalid size node: height must be positive integer" );
	cascade->orig_window_size.height = fn->data.i;

	cvStartReadSeq( stages_fn->data.seq, &stages_reader );
	for( i = 0; i < n; ++i )
	{
		CvFileNode* stage_fn;
		CvFileNode* trees_fn;
		CvSeqReader trees_reader;

		stage_fn = (CvFileNode*) stages_reader.ptr;
		if( !CV_NODE_IS_MAP( stage_fn->tag ) )
		{
			sprintf_s( buf, "Invalid stage %d", i );
			CV_Error( CV_StsError, buf );
		}

		trees_fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_TREES_NAME );
		if( !trees_fn || !CV_NODE_IS_SEQ( trees_fn->tag )
			|| trees_fn->data.seq->total <= 0 )
		{
			sprintf_s( buf, "Trees node is not a valid sequence. (stage %d)", i );
			CV_Error( CV_StsError, buf );
		}

		cascade->stage_classifier[i].classifier =
			(CvHaarClassifier*) cvAlloc( trees_fn->data.seq->total
				* sizeof( cascade->stage_classifier[i].classifier[0] ) );
		for( j = 0; j < trees_fn->data.seq->total; ++j )
		{
			cascade->stage_classifier[i].classifier[j].haar_feature = NULL;
		}
		cascade->stage_classifier[i].count = trees_fn->data.seq->total;

		cvStartReadSeq( trees_fn->data.seq, &trees_reader );
		for( j = 0; j < trees_fn->data.seq->total; ++j )
		{
			CvFileNode* tree_fn;
			CvSeqReader tree_reader;
			CvHaarClassifier* classifier;
			int last_idx;

			classifier = &cascade->stage_classifier[i].classifier[j];
			tree_fn = (CvFileNode*) trees_reader.ptr;
			if( !CV_NODE_IS_SEQ( tree_fn->tag ) || tree_fn->data.seq->total <= 0 )
			{
				sprintf_s( buf, "Tree node is not a valid sequence."
					" (stage %d, tree %d)", i, j );
				CV_Error( CV_StsError, buf );
			}

			classifier->count = tree_fn->data.seq->total;
			classifier->haar_feature = (CvHaarFeature*) cvAlloc(
				classifier->count * ( sizeof( *classifier->haar_feature ) +
					sizeof( *classifier->threshold ) +
					sizeof( *classifier->left ) +
					sizeof( *classifier->right ) ) +
					(classifier->count + 1) * sizeof( *classifier->alpha ) );
			classifier->threshold = (float*) (classifier->haar_feature+classifier->count);
			classifier->left = (int*) (classifier->threshold + classifier->count);
			classifier->right = (int*) (classifier->left + classifier->count);
			classifier->alpha = (float*) (classifier->right + classifier->count);

			cvStartReadSeq( tree_fn->data.seq, &tree_reader );
			for( k = 0, last_idx = 0; k < tree_fn->data.seq->total; ++k )
			{
				CvFileNode* node_fn;
				CvFileNode* feature_fn;
				CvFileNode* rects_fn;
				CvSeqReader rects_reader;

				node_fn = (CvFileNode*) tree_reader.ptr;
				if( !CV_NODE_IS_MAP( node_fn->tag ) )
				{
					sprintf_s( buf, "Tree node %d is not a valid map. (stage %d, tree %d)",
						k, i, j );
					CV_Error( CV_StsError, buf );
				}
				feature_fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_FEATURE_NAME );
				if( !feature_fn || !CV_NODE_IS_MAP( feature_fn->tag ) )
				{
					sprintf_s( buf, "Feature node is not a valid map. "
						"(stage %d, tree %d, node %d)", i, j, k );
					CV_Error( CV_StsError, buf );
				}
				rects_fn = cvGetFileNodeByName( fs, feature_fn, ICV_HAAR_RECTS_NAME );
				if( !rects_fn || !CV_NODE_IS_SEQ( rects_fn->tag )
					|| rects_fn->data.seq->total < 1
					|| rects_fn->data.seq->total > CV_HAAR_FEATURE_MAX )
				{
					sprintf_s( buf, "Rects node is not a valid sequence. "
						"(stage %d, tree %d, node %d)", i, j, k );
					CV_Error( CV_StsError, buf );
				}
				cvStartReadSeq( rects_fn->data.seq, &rects_reader );
				for( l = 0; l < rects_fn->data.seq->total; ++l )
				{
					CvFileNode* rect_fn;
					CvRect r;

					rect_fn = (CvFileNode*) rects_reader.ptr;
					if( !CV_NODE_IS_SEQ( rect_fn->tag ) || rect_fn->data.seq->total != 5 )
					{
						sprintf_s( buf, "Rect %d is not a valid sequence. "
							"(stage %d, tree %d, node %d)", l, i, j, k );
						CV_Error( CV_StsError, buf );
					}

					fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 0 );
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
					{
						sprintf_s( buf, "x coordinate must be non-negative integer. "
							"(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
						CV_Error( CV_StsError, buf );
					}
					r.x = fn->data.i;
					fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 1 );
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i < 0 )
					{
						sprintf_s( buf, "y coordinate must be non-negative integer. "
							"(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
						CV_Error( CV_StsError, buf );
					}
					r.y = fn->data.i;
					fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 2 );
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
						|| r.x + fn->data.i > cascade->orig_window_size.width )
					{
						sprintf_s( buf, "width must be positive integer and "
							"(x + width) must not exceed window width. "
							"(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
						CV_Error( CV_StsError, buf );
					}
					r.width = fn->data.i;
					fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 3 );
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= 0
						|| r.y + fn->data.i > cascade->orig_window_size.height )
					{
						sprintf_s( buf, "height must be positive integer and "
							"(y + height) must not exceed window height. "
							"(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
						CV_Error( CV_StsError, buf );
					}
					r.height = fn->data.i;
					fn = CV_SEQ_ELEM( rect_fn->data.seq, CvFileNode, 4 );
					if( !CV_NODE_IS_REAL( fn->tag ) )
					{
						sprintf_s( buf, "weight must be real number. "
							"(stage %d, tree %d, node %d, rect %d)", i, j, k, l );
						CV_Error( CV_StsError, buf );
					}

					classifier->haar_feature[k].rect[l].weight = (float) fn->data.f;
					classifier->haar_feature[k].rect[l].r = r;

					CV_NEXT_SEQ_ELEM( sizeof( *rect_fn ), rects_reader );
				} /* for each rect */
				for( l = rects_fn->data.seq->total; l < CV_HAAR_FEATURE_MAX; ++l )
				{
					classifier->haar_feature[k].rect[l].weight = 0;
					classifier->haar_feature[k].rect[l].r = cvRect( 0, 0, 0, 0 );
				}

				fn = cvGetFileNodeByName( fs, feature_fn, ICV_HAAR_TILTED_NAME);
				if( !fn || !CV_NODE_IS_INT( fn->tag ) )
				{
					sprintf_s( buf, "tilted must be 0 or 1. "
						"(stage %d, tree %d, node %d)", i, j, k );
					CV_Error( CV_StsError, buf );
				}
				classifier->haar_feature[k].tilted = ( fn->data.i != 0 );
				fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_THRESHOLD_NAME);
				if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
				{
					sprintf_s( buf, "threshold must be real number. "
						"(stage %d, tree %d, node %d)", i, j, k );
					CV_Error( CV_StsError, buf );
				}
				classifier->threshold[k] = (float) fn->data.f;
				fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_LEFT_NODE_NAME);
				if( fn )
				{
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
						|| fn->data.i >= tree_fn->data.seq->total )
					{
						sprintf_s( buf, "left node must be valid node number. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					/* left node */
					classifier->left[k] = fn->data.i;
				}
				else
				{
					fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_LEFT_VAL_NAME );
					if( !fn )
					{
						sprintf_s( buf, "left node or left value must be specified. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					if( !CV_NODE_IS_REAL( fn->tag ) )
					{
						sprintf_s( buf, "left value must be real number. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					/* left value */
					if( last_idx >= classifier->count + 1 )
					{
						sprintf_s( buf, "Tree structure is broken: too many values. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					classifier->left[k] = -last_idx;
					classifier->alpha[last_idx++] = (float) fn->data.f;
				}
				fn = cvGetFileNodeByName( fs, node_fn,ICV_HAAR_RIGHT_NODE_NAME);
				if( fn )
				{
					if( !CV_NODE_IS_INT( fn->tag ) || fn->data.i <= k
						|| fn->data.i >= tree_fn->data.seq->total )
					{
						sprintf_s( buf, "right node must be valid node number. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					/* right node */
					classifier->right[k] = fn->data.i;
				}
				else
				{
					fn = cvGetFileNodeByName( fs, node_fn, ICV_HAAR_RIGHT_VAL_NAME );
					if( !fn )
					{
						sprintf_s( buf, "right node or right value must be specified. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					if( !CV_NODE_IS_REAL( fn->tag ) )
					{
						sprintf_s( buf, "right value must be real number. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					/* right value */
					if( last_idx >= classifier->count + 1 )
					{
						sprintf_s( buf, "Tree structure is broken: too many values. "
							"(stage %d, tree %d, node %d)", i, j, k );
						CV_Error( CV_StsError, buf );
					}
					classifier->right[k] = -last_idx;
					classifier->alpha[last_idx++] = (float) fn->data.f;
				}

				CV_NEXT_SEQ_ELEM( sizeof( *node_fn ), tree_reader );
			} /* for each node */
			if( last_idx != classifier->count + 1 )
			{
				sprintf_s( buf, "Tree structure is broken: too few values. "
					"(stage %d, tree %d)", i, j );
				CV_Error( CV_StsError, buf );
			}

			CV_NEXT_SEQ_ELEM( sizeof( *tree_fn ), trees_reader );
		} /* for each tree */

		fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_STAGE_THRESHOLD_NAME);
		if( !fn || !CV_NODE_IS_REAL( fn->tag ) )
		{
			sprintf_s( buf, "stage threshold must be real number. (stage %d)", i );
			CV_Error( CV_StsError, buf );
		}
		cascade->stage_classifier[i].threshold = (float) fn->data.f;

		parent = i - 1;
		next = -1;

		fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_PARENT_NAME );
		if( !fn || !CV_NODE_IS_INT( fn->tag )
			|| fn->data.i < -1 || fn->data.i >= cascade->count )
		{
			sprintf_s( buf, "parent must be integer number. (stage %d)", i );
			CV_Error( CV_StsError, buf );
		}
		parent = fn->data.i;
		fn = cvGetFileNodeByName( fs, stage_fn, ICV_HAAR_NEXT_NAME );
		if( !fn || !CV_NODE_IS_INT( fn->tag )
			|| fn->data.i < -1 || fn->data.i >= cascade->count )
		{
			sprintf_s( buf, "next must be integer number. (stage %d)", i );
			CV_Error( CV_StsError, buf );
		}
		next = fn->data.i;

		cascade->stage_classifier[i].parent = parent;
		cascade->stage_classifier[i].next = next;
		cascade->stage_classifier[i].child = -1;

		if( parent != -1 && cascade->stage_classifier[parent].child == -1 )
		{
			cascade->stage_classifier[parent].child = i;
		}

		CV_NEXT_SEQ_ELEM( sizeof( *stage_fn ), stages_reader );
	} /* for each stage */

	return cascade;
}

static void
icvWriteHaarClassifier( CvFileStorage* fs, const char* name, const void* struct_ptr,
	CvAttrList attributes )
{
	int i, j, k, l;
	char buf[256];
	const CvHaarClassifierCascade* cascade = (const CvHaarClassifierCascade*) struct_ptr;

	/* TODO: parameters check */

	cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_HAAR, attributes );

	cvStartWriteStruct( fs, ICV_HAAR_SIZE_NAME, CV_NODE_SEQ | CV_NODE_FLOW );
	cvWriteInt( fs, NULL, cascade->orig_window_size.width );
	cvWriteInt( fs, NULL, cascade->orig_window_size.height );
	cvEndWriteStruct( fs ); /* size */

	cvStartWriteStruct( fs, ICV_HAAR_STAGES_NAME, CV_NODE_SEQ );
	for( i = 0; i < cascade->count; ++i )
	{
		cvStartWriteStruct( fs, NULL, CV_NODE_MAP );
		sprintf_s( buf, "stage %d", i );
		cvWriteComment( fs, buf, 1 );

		cvStartWriteStruct( fs, ICV_HAAR_TREES_NAME, CV_NODE_SEQ );

		for( j = 0; j < cascade->stage_classifier[i].count; ++j )
		{
			CvHaarClassifier* tree = &cascade->stage_classifier[i].classifier[j];

			cvStartWriteStruct( fs, NULL, CV_NODE_SEQ );
			sprintf_s( buf, "tree %d", j );
			cvWriteComment( fs, buf, 1 );

			for( k = 0; k < tree->count; ++k )
			{
				CvHaarFeature* feature = &tree->haar_feature[k];

				cvStartWriteStruct( fs, NULL, CV_NODE_MAP );
				if( k )
				{
					sprintf_s( buf, "node %d", k );
				}
				else
				{
					sprintf_s( buf, "root node" );
				}
				cvWriteComment( fs, buf, 1 );

				cvStartWriteStruct( fs, ICV_HAAR_FEATURE_NAME, CV_NODE_MAP );

				cvStartWriteStruct( fs, ICV_HAAR_RECTS_NAME, CV_NODE_SEQ );
				for( l = 0; l < CV_HAAR_FEATURE_MAX && feature->rect[l].r.width != 0; ++l )
				{
					cvStartWriteStruct( fs, NULL, CV_NODE_SEQ | CV_NODE_FLOW );
					cvWriteInt(  fs, NULL, feature->rect[l].r.x );
					cvWriteInt(  fs, NULL, feature->rect[l].r.y );
					cvWriteInt(  fs, NULL, feature->rect[l].r.width );
					cvWriteInt(  fs, NULL, feature->rect[l].r.height );
					cvWriteReal( fs, NULL, feature->rect[l].weight );
					cvEndWriteStruct( fs ); /* rect */
				}
				cvEndWriteStruct( fs ); /* rects */
				cvWriteInt( fs, ICV_HAAR_TILTED_NAME, feature->tilted );
				cvEndWriteStruct( fs ); /* feature */

				cvWriteReal( fs, ICV_HAAR_THRESHOLD_NAME, tree->threshold[k]);

				if( tree->left[k] > 0 )
				{
					cvWriteInt( fs, ICV_HAAR_LEFT_NODE_NAME, tree->left[k] );
				}
				else
				{
					cvWriteReal( fs, ICV_HAAR_LEFT_VAL_NAME,
						tree->alpha[-tree->left[k]] );
				}

				if( tree->right[k] > 0 )
				{
					cvWriteInt( fs, ICV_HAAR_RIGHT_NODE_NAME, tree->right[k] );
				}
				else
				{
					cvWriteReal( fs, ICV_HAAR_RIGHT_VAL_NAME,
						tree->alpha[-tree->right[k]] );
				}

				cvEndWriteStruct( fs ); /* split */
			}

			cvEndWriteStruct( fs ); /* tree */
		}

		cvEndWriteStruct( fs ); /* trees */

		cvWriteReal( fs, ICV_HAAR_STAGE_THRESHOLD_NAME, cascade->stage_classifier[i].threshold);
		cvWriteInt( fs, ICV_HAAR_PARENT_NAME, cascade->stage_classifier[i].parent );
		cvWriteInt( fs, ICV_HAAR_NEXT_NAME, cascade->stage_classifier[i].next );

		cvEndWriteStruct( fs ); /* stage */
	} /* for each stage */

	cvEndWriteStruct( fs ); /* stages */
	cvEndWriteStruct( fs ); /* root */
}

static void*
icvCloneHaarClassifier( const void* struct_ptr )
{
	CvHaarClassifierCascade* cascade = NULL;

	int i, j, k, n;
	const CvHaarClassifierCascade* cascade_src =
		(const CvHaarClassifierCascade*) struct_ptr;

	n = cascade_src->count;
	cascade = icvCreateHaarClassifierCascade(n);
	cascade->orig_window_size = cascade_src->orig_window_size;

	for( i = 0; i < n; ++i )
	{
		cascade->stage_classifier[i].parent = cascade_src->stage_classifier[i].parent;
		cascade->stage_classifier[i].next = cascade_src->stage_classifier[i].next;
		cascade->stage_classifier[i].child = cascade_src->stage_classifier[i].child;
		cascade->stage_classifier[i].threshold = cascade_src->stage_classifier[i].threshold;

		cascade->stage_classifier[i].count = 0;
		cascade->stage_classifier[i].classifier =
			(CvHaarClassifier*) cvAlloc( cascade_src->stage_classifier[i].count
				* sizeof( cascade->stage_classifier[i].classifier[0] ) );

		cascade->stage_classifier[i].count = cascade_src->stage_classifier[i].count;

		for( j = 0; j < cascade->stage_classifier[i].count; ++j )
			cascade->stage_classifier[i].classifier[j].haar_feature = NULL;

		for( j = 0; j < cascade->stage_classifier[i].count; ++j )
		{
			const CvHaarClassifier* classifier_src =
				&cascade_src->stage_classifier[i].classifier[j];
			CvHaarClassifier* classifier =
				&cascade->stage_classifier[i].classifier[j];

			classifier->count = classifier_src->count;
			classifier->haar_feature = (CvHaarFeature*) cvAlloc(
				classifier->count * ( sizeof( *classifier->haar_feature ) +
					sizeof( *classifier->threshold ) +
					sizeof( *classifier->left ) +
					sizeof( *classifier->right ) ) +
					(classifier->count + 1) * sizeof( *classifier->alpha ) );
			classifier->threshold = (float*) (classifier->haar_feature+classifier->count);
			classifier->left = (int*) (classifier->threshold + classifier->count);
			classifier->right = (int*) (classifier->left + classifier->count);
			classifier->alpha = (float*) (classifier->right + classifier->count);
			for( k = 0; k < classifier->count; ++k )
			{
				classifier->haar_feature[k] = classifier_src->haar_feature[k];
				classifier->threshold[k] = classifier_src->threshold[k];
				classifier->left[k] = classifier_src->left[k];
				classifier->right[k] = classifier_src->right[k];
				classifier->alpha[k] = classifier_src->alpha[k];
			}
			classifier->alpha[classifier->count] =
				classifier_src->alpha[classifier->count];
		}
	}

	return cascade;
}


CvType haar_type( CV_TYPE_NAME_HAAR, icvIsHaarClassifier,
(CvReleaseFunc)cvReleaseHaarClassifierCascade,
icvReadHaarClassifier, icvWriteHaarClassifier,
icvCloneHaarClassifier );
