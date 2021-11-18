#pragma once


#include "opencv2/core.hpp"



/****************************************************************************************\
*                         objdetect.hpp						                             *
\****************************************************************************************/

struct CV_EXPORTS_W HOGDescriptor
{
public:
	enum { L2Hys = 0 //!< Default histogramNormType
	};
	enum { DEFAULT_NLEVELS = 64 //!< Default nlevels value.
	};
	/**@brief Creates the HOG descriptor and detector with default params.

	aqual to HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9, 1 )
	*/
	CV_WRAP HOGDescriptor() : winSize(64,128), blockSize(16,16), blockStride(8,8),
		cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
		histogramNormType(HOGDescriptor::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
		free_coef(-1.f), nlevels(HOGDescriptor::DEFAULT_NLEVELS), signedGradient(false)
	{}

	/** @overload
	@param _winSize sets winSize with given value.
	@param _blockSize sets blockSize with given value.
	@param _blockStride sets blockStride with given value.
	@param _cellSize sets cellSize with given value.
	@param _nbins sets nbins with given value.
	@param _derivAperture sets derivAperture with given value.
	@param _winSigma sets winSigma with given value.
	@param _histogramNormType sets histogramNormType with given value.
	@param _L2HysThreshold sets L2HysThreshold with given value.
	@param _gammaCorrection sets gammaCorrection with given value.
	@param _nlevels sets nlevels with given value.
	@param _signedGradient sets signedGradient with given value.
	*/
	CV_WRAP HOGDescriptor(cv::Size _winSize, cv::Size _blockSize, cv::Size _blockStride,
		cv::Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
		int _histogramNormType=HOGDescriptor::L2Hys,
		double _L2HysThreshold=0.2, bool _gammaCorrection=false,
		int _nlevels=HOGDescriptor::DEFAULT_NLEVELS, bool _signedGradient=false)
		: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
		nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
		histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
		gammaCorrection(_gammaCorrection), free_coef(-1.f), nlevels(_nlevels), signedGradient(_signedGradient)
	{}

	/** @overload
	@param d the HOGDescriptor which cloned to create a new one.
	*/
	HOGDescriptor(const HOGDescriptor& d)
	{
		d.copyTo(*this);
	}

	/**@brief Default destructor.
	*/
	virtual ~HOGDescriptor() {}

	/**@brief Returns the number of coefficients required for the classification.
	*/
	CV_WRAP size_t getDescriptorSize() const;

	/** @brief Checks if detector size equal to descriptor size.
	*/
	CV_WRAP bool checkDetectorSize() const;

	/** @brief Returns winSigma value
	*/
	CV_WRAP double getWinSigma() const;

	/** @brief clones the HOGDescriptor
	@param c cloned HOGDescriptor
	*/
	virtual void copyTo(HOGDescriptor& c) const;

	/**@example train_HOG.cpp
	*/
	/** @brief Computes HOG descriptors of given image.
	@param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
	@param descriptors Matrix of the type CV_32F
	@param winStride Window stride. It must be a multiple of block stride.
	@param padding Padding
	@param locations Vector of Point
	*/
	CV_WRAP virtual void compute(cv::InputArray img,
		CV_OUT std::vector<float>& descriptors,
		cv::Size winStride = cv::Size(), cv::Size padding = cv::Size(),
		const std::vector<cv::Point>& locations = std::vector<cv::Point>()) const;

	/** @brief  Computes gradients and quantized gradient orientations.
	@param img Matrix contains the image to be computed
	@param grad Matrix of type CV_32FC2 contains computed gradients
	@param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
	@param paddingTL Padding from top-left
	@param paddingBR Padding from bottom-right
	*/
	CV_WRAP virtual void computeGradient(const cv::Mat& img, CV_OUT cv::Mat& grad, CV_OUT cv::Mat& angleOfs,
		cv::Size paddingTL = cv::Size(), cv::Size paddingBR = cv::Size()) const;

	//! Detection window size. Align to block size and block stride. Default value is Size(64,128).
	CV_PROP cv::Size winSize;

	//! Block size in pixels. Align to cell size. Default value is Size(16,16).
	CV_PROP cv::Size blockSize;

	//! Block stride. It must be a multiple of cell size. Default value is Size(8,8).
	CV_PROP cv::Size blockStride;

	//! Cell size. Default value is Size(8,8).
	CV_PROP cv::Size cellSize;

	//! Number of bins used in the calculation of histogram of gradients. Default value is 9.
	CV_PROP int nbins;

	//! not documented
	CV_PROP int derivAperture;

	//! Gaussian smoothing window parameter.
	CV_PROP double winSigma;

	//! histogramNormType
	CV_PROP int histogramNormType;

	//! L2-Hys normalization method shrinkage.
	CV_PROP double L2HysThreshold;

	//! Flag to specify whether the gamma correction preprocessing is required or not.
	CV_PROP bool gammaCorrection;

	//! not documented
	float free_coef;

	//! Maximum number of detection window increases. Default value is 64
	CV_PROP int nlevels;

	//! Indicates signed gradient will be used or not
	CV_PROP bool signedGradient;

};