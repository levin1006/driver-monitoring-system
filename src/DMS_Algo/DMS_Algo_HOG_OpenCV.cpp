
#include "DMS_Algo_HOG_OpenCV.h"

using namespace cv;

/****************************************************************************************\
*                         hog.cpp						                             *
\****************************************************************************************/

size_t HOGDescriptor::getDescriptorSize() const
{
	CV_Assert(blockSize.width % cellSize.width == 0 &&
		blockSize.height % cellSize.height == 0);
	CV_Assert((winSize.width - blockSize.width) % blockStride.width == 0 &&
		(winSize.height - blockSize.height) % blockStride.height == 0 );

	return (size_t)nbins*
		(blockSize.width/cellSize.width)*
		(blockSize.height/cellSize.height)*
		((winSize.width - blockSize.width)/blockStride.width + 1)*
		((winSize.height - blockSize.height)/blockStride.height + 1);
}

double HOGDescriptor::getWinSigma() const
{
	return winSigma > 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
}


void HOGDescriptor::copyTo(HOGDescriptor& c) const
{
	c.winSize = winSize;
	c.blockSize = blockSize;
	c.blockStride = blockStride;
	c.cellSize = cellSize;
	c.nbins = nbins;
	c.derivAperture = derivAperture;
	c.winSigma = winSigma;
	c.histogramNormType = histogramNormType;
	c.L2HysThreshold = L2HysThreshold;
	c.gammaCorrection = gammaCorrection;
	c.nlevels = nlevels;
	c.signedGradient = signedGradient;
}

void HOGDescriptor::computeGradient(const Mat& img, Mat& grad, Mat& qangle,
	Size paddingTL, Size paddingBR) const
{
	Size gradsize(img.cols + paddingTL.width + paddingBR.width,
		img.rows + paddingTL.height + paddingBR.height);
	grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
	qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation

	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);

	int i, x, y;

	Mat_<float> _lut(1, 256);
	const float* const lut = &_lut(0,0);

	if( gammaCorrection )
		for( i = 0; i < 256; i++ )
			_lut(0,i) = std::sqrt((float)i);
	else
		for( i = 0; i < 256; i++ )
			_lut(0,i) = (float)i;

	AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + gradsize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;

	for( x = -1; x < gradsize.width + 1; x++ )
		xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
			wholeSize.width, borderType) - roiofs.x;
	for( y = -1; y < gradsize.height + 1; y++ )
		ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
			wholeSize.height, borderType) - roiofs.y;

	// x- & y- derivatives for the whole row
	int width = gradsize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* const dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	float angleScale = signedGradient ? (float)(nbins/(2.0*CV_PI)) : (float)(nbins/CV_PI);
	for( y = 0; y < gradsize.height; y++ )
	{
		const uchar* imgPtr  = img.ptr(ymap[y]);
		//In case subimage is used ptr() generates an assert for next and prev rows
		//(see http://code.opencv.org/issues/4149)
		const uchar* prevPtr = img.data + img.step*ymap[y-1];
		const uchar* nextPtr = img.data + img.step*ymap[y+1];

		float* gradPtr = grad.ptr<float>(y);
		uchar* qanglePtr = qangle.ptr(y);

		for( x = 0; x < width; x++ )
		{
			int x1 = xmap[x];
			dbuf[x] = (float)(lut[imgPtr[xmap[x+1]]] - lut[imgPtr[xmap[x-1]]]);
			dbuf[width + x] = (float)(lut[nextPtr[x1]] - lut[prevPtr[x1]]);
		}

		// computing angles and magnidutes
		cartToPolar( Dx, Dy, Mag, Angle, false );

		// filling the result matrix
		x = 0;
		for( ; x < width; x++ )
		{
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			gradPtr[x*2] = mag*(1.f - angle);
			gradPtr[x*2+1] = mag*angle;

			if( hidx < 0 )
				hidx += nbins;
			else if( hidx >= nbins )
				hidx -= nbins;

			CV_Assert( (unsigned)hidx < (unsigned)nbins );

			qanglePtr[x*2] = (uchar)hidx;
			hidx++;
			hidx &= hidx < nbins ? -1 : 0;
			qanglePtr[x*2+1] = (uchar)hidx;
		}
	}
}

struct HOGCache
{
	struct BlockData
	{
		BlockData() :
			histOfs(0), imgOffset()
		{ }

		int histOfs;
		Point imgOffset;
	};

	struct PixData
	{
		size_t gradOfs, qangleOfs;
		int histOfs[4];
		float histWeights[4];
		float gradWeight;
	};

	HOGCache();
	HOGCache(const HOGDescriptor* descriptor,
		const Mat& img, const Size& paddingTL, const Size& paddingBR,
		bool useCache, const Size& cacheStride);
	virtual ~HOGCache() { }
	virtual void init(const HOGDescriptor* descriptor,
		const Mat& img, const Size& paddingTL, const Size& paddingBR,
		bool useCache, const Size& cacheStride);

	Size windowsInImage(const Size& imageSize, const Size& winStride) const;
	Rect getWindow(const Size& imageSize, const Size& winStride, int idx) const;

	const float* getBlock(Point pt, float* buf);
	virtual void normalizeBlockHistogram(float* histogram) const;

	std::vector<PixData> pixData;
	std::vector<BlockData> blockData;

	bool useCache;
	std::vector<int> ymaxCached;
	Size winSize;
	Size cacheStride;
	Size nblocks, ncells;
	int blockHistogramSize;
	int count1, count2, count4;
	Point imgoffset;
	Mat_<float> blockCache;
	Mat_<uchar> blockCacheFlags;

	Mat grad, qangle;
	const HOGDescriptor* descriptor;
};

HOGCache::HOGCache() :
	blockHistogramSize(), count1(), count2(), count4()
{
	useCache = false;
	descriptor = 0;
}

HOGCache::HOGCache(const HOGDescriptor* _descriptor,
	const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
	bool _useCache, const Size& _cacheStride)
{
	init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void HOGCache::init(const HOGDescriptor* _descriptor,
	const Mat& _img, const Size& _paddingTL, const Size& _paddingBR,
	bool _useCache, const Size& _cacheStride)
{
	descriptor = _descriptor;
	cacheStride = _cacheStride;
	useCache = _useCache;

	descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR);
	imgoffset = _paddingTL;

	winSize = descriptor->winSize;
	Size blockSize = descriptor->blockSize;
	Size blockStride = descriptor->blockStride;
	Size cellSize = descriptor->cellSize;
	int i, j, nbins = descriptor->nbins;
	int rawBlockSize = blockSize.width*blockSize.height;

	nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
		(winSize.height - blockSize.height)/blockStride.height + 1);
	ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);
	blockHistogramSize = ncells.width*ncells.height*nbins;

	if( useCache )
	{
		Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
			(winSize.height/cacheStride.height)+1);

		blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
		blockCacheFlags.create(cacheSize);

		size_t cacheRows = blockCache.rows;
		ymaxCached.resize(cacheRows);
		for(size_t ii = 0; ii < cacheRows; ii++ )
			ymaxCached[ii] = -1;
	}

	Mat_<float> weights(blockSize);
	float sigma = (float)descriptor->getWinSigma();
	float scale = 1.f/(sigma*sigma*2);

	{
		AutoBuffer<float> di(blockSize.height), dj(blockSize.width);
		float* _di = (float*)di, *_dj = (float*)dj;
		float bh = blockSize.height * 0.5f, bw = blockSize.width * 0.5f;

		i = 0;
		for ( ; i < blockSize.height; ++i)
		{
			_di[i] = i - bh;
			_di[i] *= _di[i];
		}

		j = 0;
		for ( ; j < blockSize.width; ++j)
		{
			_dj[j] = j - bw;
			_dj[j] *= _dj[j];
		}

		for(i = 0; i < blockSize.height; i++)
			for(j = 0; j < blockSize.width; j++)
				weights(i,j) = std::exp(-(_di[i] + _dj[j])*scale);
	}

	blockData.resize(nblocks.width*nblocks.height);
	pixData.resize(rawBlockSize*3);

	// Initialize 2 lookup tables, pixData & blockData.
	// Here is why:
	//
	// The detection algorithm runs in 4 nested loops (at each pyramid layer):
	//  loop over the windows within the input image
	//    loop over the blocks within each window
	//      loop over the cells within each block
	//        loop over the pixels in each cell
	//
	// As each of the loops runs over a 2-dimensional array,
	// we could get 8(!) nested loops in total, which is very-very slow.
	//
	// To speed the things up, we do the following:
	//   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
	//         inside we compute the current search window using getWindow() method.
	//         Yes, it involves some overhead (function call + couple of divisions),
	//         but it's tiny in fact.
	//   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
	//         to set up gradient and histogram pointers.
	//   3. loops over cells and pixels in each cell are merged
	//       (since there is no overlap between cells, each pixel in the block is processed once)
	//      and also unrolled. Inside we use PixData[k] to access the gradient values and
	//      update the histogram
	//

	count1 = count2 = count4 = 0;
	for( j = 0; j < blockSize.width; j++ )
		for( i = 0; i < blockSize.height; i++ )
		{
			PixData* data = 0;
			float cellX = (j+0.5f)/cellSize.width - 0.5f;
			float cellY = (i+0.5f)/cellSize.height - 0.5f;
			int icellX0 = cvFloor(cellX);
			int icellY0 = cvFloor(cellY);
			int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
			cellX -= icellX0;
			cellY -= icellY0;

			if( (unsigned)icellX0 < (unsigned)ncells.width &&
				(unsigned)icellX1 < (unsigned)ncells.width )
			{
				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					data = &pixData[rawBlockSize*2 + (count4++)];
					data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
					data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
					data->histWeights[1] = cellX*(1.f - cellY);
					data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
					data->histWeights[2] = (1.f - cellX)*cellY;
					data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[3] = cellX*cellY;
				}
				else
				{
					data = &pixData[rawBlockSize + (count2++)];
					if( (unsigned)icellY0 < (unsigned)ncells.height )
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
					data->histWeights[0] = (1.f - cellX)*cellY;
					data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			else
			{
				if( (unsigned)icellX0 < (unsigned)ncells.width )
				{
					icellX1 = icellX0;
					cellX = 1.f - cellX;
				}

				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					data = &pixData[rawBlockSize + (count2++)];
					data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
					data->histWeights[0] = cellX*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
				else
				{
					data = &pixData[count1++];
					if( (unsigned)icellY0 < (unsigned)ncells.height )
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
					data->histWeights[0] = cellX*cellY;
					data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			data->gradOfs = (grad.cols*i + j)*2;
			data->qangleOfs = (qangle.cols*i + j)*2;
			data->gradWeight = weights(i,j);
		}

	assert( count1 + count2 + count4 == rawBlockSize );
	// defragment pixData
	for( j = 0; j < count2; j++ )
		pixData[j + count1] = pixData[j + rawBlockSize];
	for( j = 0; j < count4; j++ )
		pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
	count2 += count1;
	count4 += count2;

	// initialize blockData
	for( j = 0; j < nblocks.width; j++ )
		for( i = 0; i < nblocks.height; i++ )
		{
			BlockData& data = blockData[j*nblocks.height + i];
			data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
			data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
		}
}

const float* HOGCache::getBlock(Point pt, float* buf)
{
	float* blockHist = buf;
	assert(descriptor != 0);

	//    Size blockSize = descriptor->blockSize;
	pt += imgoffset;

	//    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
	//        (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

	if( useCache )
	{
		CV_Assert( pt.x % cacheStride.width == 0 &&
			pt.y % cacheStride.height == 0 );
		Point cacheIdx(pt.x/cacheStride.width,
			(pt.y/cacheStride.height) % blockCache.rows);
		if( pt.y != ymaxCached[cacheIdx.y] )
		{
			Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
			cacheRow = (uchar)0;
			ymaxCached[cacheIdx.y] = pt.y;
		}

		blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
		uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
		if( computedFlag != 0 )
			return blockHist;
		computedFlag = (uchar)1; // set it at once, before actual computing
	}

	int k, C1 = count1, C2 = count2, C4 = count4;
	const float* gradPtr = grad.ptr<float>(pt.y) + pt.x*2;
	const uchar* qanglePtr = qangle.ptr(pt.y) + pt.x*2;

	//    CV_Assert( blockHist != 0 );
	memset(blockHist, 0, sizeof(float) * blockHistogramSize);

	const PixData* _pixData = &pixData[0];

	for( k = 0; k < C1; k++ )
	{
		const PixData& pk = _pixData[k];
		const float* const a = gradPtr + pk.gradOfs;
		float w = pk.gradWeight*pk.histWeights[0];
		const uchar* h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];

		float* hist = blockHist + pk.histOfs[0];
		float t0 = hist[h0] + a[0]*w;
		float t1 = hist[h1] + a[1]*w;
		hist[h0] = t0; hist[h1] = t1;
	}

	for( ; k < C2; k++ )
	{
		const PixData& pk = _pixData[k];
		const float* const a = gradPtr + pk.gradOfs;
		float w, t0, t1, a0 = a[0], a1 = a[1];
		const uchar* const h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];

		float* hist = blockHist + pk.histOfs[0];
		w = pk.gradWeight*pk.histWeights[0];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[1];
		w = pk.gradWeight*pk.histWeights[1];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;
	}

	for( ; k < C4; k++ )
	{
		const PixData& pk = _pixData[k];
		const float* a = gradPtr + pk.gradOfs;
		float w, t0, t1, a0 = a[0], a1 = a[1];
		const uchar* h = qanglePtr + pk.qangleOfs;
		int h0 = h[0], h1 = h[1];

		float* hist = blockHist + pk.histOfs[0];
		w = pk.gradWeight*pk.histWeights[0];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[1];
		w = pk.gradWeight*pk.histWeights[1];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[2];
		w = pk.gradWeight*pk.histWeights[2];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;

		hist = blockHist + pk.histOfs[3];
		w = pk.gradWeight*pk.histWeights[3];
		t0 = hist[h0] + a0*w;
		t1 = hist[h1] + a1*w;
		hist[h0] = t0; hist[h1] = t1;
	}

	normalizeBlockHistogram(blockHist);

	return blockHist;
}

void HOGCache::normalizeBlockHistogram(float* _hist) const
{
	float* hist = &_hist[0], sum = 0.0f, partSum[4];
	size_t i = 0, sz = blockHistogramSize;

	partSum[0] = 0.0f;
	partSum[1] = 0.0f;
	partSum[2] = 0.0f;
	partSum[3] = 0.0f;
	for ( ; i <= sz - 4; i += 4)
	{
		partSum[0] += hist[i] * hist[i];
		partSum[1] += hist[i+1] * hist[i+1];
		partSum[2] += hist[i+2] * hist[i+2];
		partSum[3] += hist[i+3] * hist[i+3];
	}

	float t0 = partSum[0] + partSum[1];
	float t1 = partSum[2] + partSum[3];
	sum = t0 + t1;
	for ( ; i < sz; ++i)
		sum += hist[i]*hist[i];

	float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;
	i = 0, sum = 0.0f;

	partSum[0] = 0.0f;
	partSum[1] = 0.0f;
	partSum[2] = 0.0f;
	partSum[3] = 0.0f;
	for( ; i <= sz - 4; i += 4)
	{
		hist[i] = std::min(hist[i]*scale, thresh);
		hist[i+1] = std::min(hist[i+1]*scale, thresh);
		hist[i+2] = std::min(hist[i+2]*scale, thresh);
		hist[i+3] = std::min(hist[i+3]*scale, thresh);
		partSum[0] += hist[i]*hist[i];
		partSum[1] += hist[i+1]*hist[i+1];
		partSum[2] += hist[i+2]*hist[i+2];
		partSum[3] += hist[i+3]*hist[i+3];
	}

	t0 = partSum[0] + partSum[1];
	t1 = partSum[2] + partSum[3];
	sum = t0 + t1;
	for( ; i < sz; ++i)
	{
		hist[i] = std::min(hist[i]*scale, thresh);
		sum += hist[i]*hist[i];
	}

	scale = 1.f/(std::sqrt(sum)+1e-3f), i = 0;

	for ( ; i < sz; ++i)
		hist[i] *= scale;
}

Size HOGCache::windowsInImage(const Size& imageSize, const Size& winStride) const
{
	return Size((imageSize.width - winSize.width)/winStride.width + 1,
		(imageSize.height - winSize.height)/winStride.height + 1);
}

Rect HOGCache::getWindow(const Size& imageSize, const Size& winStride, int idx) const
{
	int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
	int y = idx / nwindowsX;
	int x = idx - nwindowsX*y;
	return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
}

static inline int gcd(int a, int b)
{
	if( a < b )
		std::swap(a, b);
	while( b > 0 )
	{
		int r = a % b;
		a = b;
		b = r;
	}
	return a;
}


void HOGDescriptor::compute(InputArray _img, std::vector<float>& descriptors,
	Size winStride, Size padding, const std::vector<Point>& locations) const
{

		if( winStride == Size() )
			winStride = cellSize;
	Size cacheStride(gcd(winStride.width, blockStride.width),
		gcd(winStride.height, blockStride.height));

	Size imgSize = _img.size();

	size_t nwindows = locations.size();
	padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
	padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
	Size paddedImgSize(imgSize.width + padding.width*2, imgSize.height + padding.height*2);

		Mat img = _img.getMat();
	HOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);

	if( !nwindows )
		nwindows = cache.windowsInImage(paddedImgSize, winStride).area();

	const HOGCache::BlockData* blockData = &cache.blockData[0];

	int nblocks = cache.nblocks.area();
	int blockHistogramSize = cache.blockHistogramSize;
	size_t dsize = getDescriptorSize();
	descriptors.resize(dsize*nwindows);

	// for each window
	for( size_t i = 0; i < nwindows; i++ )
	{
		float* descriptor = &descriptors[i*dsize];

		Point pt0;
		if( !locations.empty() )
		{
			pt0 = locations[i];
			if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
				pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
				continue;
		}
		else
		{
			pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
			//            CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
		}

		for( int j = 0; j < nblocks; j++ )
		{
			const HOGCache::BlockData& bj = blockData[j];
			Point pt = pt0 + bj.imgOffset;

			float* dst = descriptor + bj.histOfs;
			const float* src = cache.getBlock(pt, dst);
			if( src != dst )
				memcpy(dst, src, blockHistogramSize * sizeof(float));
		}
	}
}