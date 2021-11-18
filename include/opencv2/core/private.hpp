/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_CORE_PRIVATE_HPP
#define OPENCV_CORE_PRIVATE_HPP


#include "opencv2/core.hpp"
#include "cvconfig.h"

#include <opencv2/core/utils/trace.hpp>

#ifdef HAVE_EIGEN
#  if defined __GNUC__ && defined __APPLE__
#    pragma GCC diagnostic ignored "-Wshadow"
#  endif
#  include <Eigen/Core>
#  include "opencv2/core/eigen.hpp"
#endif

#ifdef HAVE_TBB
#  include "tbb/tbb.h"
#  include "tbb/task.h"
#  undef min
#  undef max
#endif

//! @cond IGNORED

namespace cv
{
#ifdef HAVE_TBB

    typedef tbb::blocked_range<int> BlockedRange;

    template<typename Body> static inline
    void parallel_for( const BlockedRange& range, const Body& body )
    {
        tbb::parallel_for(range, body);
    }

    typedef tbb::split Split;

    template<typename Body> static inline
    void parallel_reduce( const BlockedRange& range, Body& body )
    {
        tbb::parallel_reduce(range, body);
    }

    typedef tbb::concurrent_vector<Rect> ConcurrentRectVector;
#else
    class BlockedRange
    {
    public:
        BlockedRange() : _begin(0), _end(0), _grainsize(0) {}
        BlockedRange(int b, int e, int g=1) : _begin(b), _end(e), _grainsize(g) {}
        int begin() const { return _begin; }
        int end() const { return _end; }
        int grainsize() const { return _grainsize; }

    protected:
        int _begin, _end, _grainsize;
    };

    template<typename Body> static inline
    void parallel_for( const BlockedRange& range, const Body& body )
    {
        body(range);
    }
    typedef std::vector<Rect> ConcurrentRectVector;

    class Split {};

    template<typename Body> static inline
    void parallel_reduce( const BlockedRange& range, Body& body )
    {
        body(range);
    }
#endif

    // Returns a static string if there is a parallel framework,
    // NULL otherwise.
    CV_EXPORTS const char* currentParallelFramework();
} //namespace cv

/****************************************************************************************\
*                                  Common declarations                                   *
\****************************************************************************************/

/* the alignment of all the allocated buffers */
#define  CV_MALLOC_ALIGN    64

/* IEEE754 constants and macros */
#define  CV_TOGGLE_FLT(x) ((x)^((int)(x) < 0 ? 0x7fffffff : 0))
#define  CV_TOGGLE_DBL(x) ((x)^((int64)(x) < 0 ? CV_BIG_INT(0x7fffffffffffffff) : 0))

static inline void* cvAlignPtr( const void* ptr, int align = 32 )
{
    CV_DbgAssert ( (align & (align-1)) == 0 );
    return (void*)( ((size_t)ptr + align - 1) & ~(size_t)(align-1) );
}

static inline int cvAlign( int size, int align )
{
    CV_DbgAssert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

#ifdef IPL_DEPTH_8U
static inline cv::Size cvGetMatSize( const CvMat* mat )
{
    return cv::Size(mat->cols, mat->rows);
}
#endif

namespace cv
{
CV_EXPORTS void scalarToRawData(const cv::Scalar& s, void* buf, int type, int unroll_to = 0);
}

// property implementation macros

#define CV_IMPL_PROPERTY_RO(type, name, member) \
    inline type get##name() const { return member; }

#define CV_HELP_IMPL_PROPERTY(r_type, w_type, name, member) \
    CV_IMPL_PROPERTY_RO(r_type, name, member) \
    inline void set##name(w_type val) { member = val; }

#define CV_HELP_WRAP_PROPERTY(r_type, w_type, name, internal_name, internal_obj) \
    r_type get##name() const { return internal_obj.get##internal_name(); } \
    void set##name(w_type val) { internal_obj.set##internal_name(val); }

#define CV_IMPL_PROPERTY(type, name, member) CV_HELP_IMPL_PROPERTY(type, type, name, member)
#define CV_IMPL_PROPERTY_S(type, name, member) CV_HELP_IMPL_PROPERTY(type, const type &, name, member)

#define CV_WRAP_PROPERTY(type, name, internal_name, internal_obj)  CV_HELP_WRAP_PROPERTY(type, type, name, internal_name, internal_obj)
#define CV_WRAP_PROPERTY_S(type, name, internal_name, internal_obj) CV_HELP_WRAP_PROPERTY(type, const type &, name, internal_name, internal_obj)

#define CV_WRAP_SAME_PROPERTY(type, name, internal_obj) CV_WRAP_PROPERTY(type, name, name, internal_obj)
#define CV_WRAP_SAME_PROPERTY_S(type, name, internal_obj) CV_WRAP_PROPERTY_S(type, name, name, internal_obj)


#define CV_INSTRUMENT_REGION_META(...)

#define CV_INSTRUMENT_REGION_()                            CV_TRACE_FUNCTION()
#define CV_INSTRUMENT_REGION_NAME(...)                     CV_TRACE_REGION(__VA_ARGS__)
#define CV_INSTRUMENT_REGION_MT_FORK()

#define CV_INSTRUMENT_REGION_IPP()                         CV__TRACE_REGION_("IPP", CV_TRACE_NS::details::REGION_FLAG_IMPL_IPP)
#define CV_INSTRUMENT_FUN_IPP(FUN, ...) ((FUN)(__VA_ARGS__))
#define CV_INSTRUMENT_MARK_IPP(...)

#define CV_INSTRUMENT_REGION_OPENCL()                      CV__TRACE_REGION_("OpenCL", CV_TRACE_NS::details::REGION_FLAG_IMPL_OPENCL)
#define CV_INSTRUMENT_REGION_OPENCL_COMPILE(...)
#define CV_INSTRUMENT_REGION_OPENCL_RUN(...)
#define CV_INSTRUMENT_MARK_OPENCL(...)
#endif

#ifdef __CV_AVX_GUARD
#define CV_INSTRUMENT_REGION() __CV_AVX_GUARD CV_INSTRUMENT_REGION_()
#else
#define CV_INSTRUMENT_REGION() CV_INSTRUMENT_REGION_()
#endif

//! @endcond
