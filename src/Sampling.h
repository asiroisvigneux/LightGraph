// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Sampling.h
///
/// @brief This file contains some utility functions to sample and hdr
/// texture in the context of environment lighting using importance sampling.
///
/// @note Most of those functions come directly from PBRT with very little
/// modifications: https://github.com/mmp/pbrt-v3


#ifndef SRC_SAMPLING_H_
#define SRC_SAMPLING_H_


#include <vector>
#include <memory>
#include <iostream>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <OpenEXR/ImfMatrixAttribute.h>
#include <OpenEXR/ImfArray.h>
#include <OpenEXR/ImathBox.h>

#include <openvdb/openvdb.h>


using namespace openvdb;
using namespace std;


/// @brief This function clamps a value to a parameterize range
/// @param val  the value to clamp
/// @param low  the lower bound of the value
/// @param high the upper bound of the value
/// @return     returns the clamp value
template <typename T, typename U, typename V>
inline T Clamp(T val, U low, V high) {
    if (val < low)
        return low;
    else if (val > high)
        return high;
    else
        return val;
}


/// @brief Binary search in a 1d array
/// @param size     dimensions of the array
/// @param pred     condition to test against
/// @return returns the index of the found position
template <typename Predicate>
int FindInterval(int size, const Predicate &pred) {
    int first = 0, len = size;
    while (len > 0) {
        int half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return Clamp(first - 1, 0, size - 2);
}


/// @brief Creates a 1d distribution based on a 1d array
struct Distribution1D {
    // Distribution1D Public Methods
    /// @brief Create a 1D distribution based on the values in the provided 1d array
    /// @param f provided array to drive the distribution
    /// @param n array length
    Distribution1D(const float *f, int n) : func(f, f + n), cdf(n + 1) {
        // Compute integral of step function at $x_i$
        cdf[0] = 0;
        for (int i = 1; i < n + 1; ++i) cdf[i] = cdf[i - 1] + func[i - 1] / n;

        // Transform step function integral into CDF
        funcInt = cdf[n];
        if (funcInt == 0) {
            for (int i = 1; i < n + 1; ++i) cdf[i] = float(i) / float(n);
        } else {
            for (int i = 1; i < n + 1; ++i) cdf[i] /= funcInt;
        }
    }

    int Count() const { return (int)func.size(); }

    /// @brief Sample values from the predefined 1d distribution
    /// @param u    a random uniform 1d sample
    /// @param pdf  the probability associated to this sample
    /// @param off  index of the value to sample based on the provided u sample
    /// @return     returns the position to sample in the [0,1) range
    float SampleContinuous(float u, float *pdf, int *off = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int offset = FindInterval((int)cdf.size(),
                                  [&](int index) { return cdf[index] <= u; });
        if (off) *off = offset;

        // Compute offset along CDF segment
        float du = u - cdf[offset];
        if ((cdf[offset + 1] - cdf[offset]) > 0) {
            du /= (cdf[offset + 1] - cdf[offset]);
        }

        // Compute PDF for sampled offset
        if (pdf) *pdf = (funcInt > 0) ? func[offset] / funcInt : 0;

        // Return $x\in{}[0,1)$ corresponding to sample
        return (offset + du) / Count();
    }

    // Distribution1D Public Data
    std::vector<float> func, cdf;
    float funcInt;
};


/// @brief Creates a 2d distribution based on a 2d array
class Distribution2D {
public:
    /// @brief Create a 2D distribution based on the values in the provided 2d
    /// array by computing the conditional and marginal distribution
    /// @param func 2d array to use to generate the distribution
    /// @param nu   horizontal index
    /// @param nv   vertical index
    Distribution2D(const float *func, int nu, int nv) {
        pConditionalV.reserve(nv);
        for (int v = 0; v < nv; ++v) {
            // Compute conditional sampling distribution for $\tilde{v}$
            pConditionalV.emplace_back(new Distribution1D(&func[v * nu], nu));
        }
        // Compute marginal sampling distribution $p[\tilde{v}]$
        std::vector<float> marginalFunc;
        marginalFunc.reserve(nv);
        for (int v = 0; v < nv; ++v) {
            marginalFunc.push_back(pConditionalV[v]->funcInt);
        }
        pMarginal.reset(new Distribution1D(&marginalFunc[0], nv));
    }

    /// @brief Sample values from the predefined 2d distribution
    /// @param u    a random uniform 2d sample
    /// @param pdf  the probability associated to this sample
    /// @return     returns the 2d vector to sample according to the 2d distribution
    Vec2f SampleContinuous(const Vec2f &u, float *pdf) const {
        float pdfs[2];
        int v;
        float d1 = pMarginal->SampleContinuous(u[1], &pdfs[1], &v);
        float d0 = pConditionalV[v]->SampleContinuous(u[0], &pdfs[0]);
        *pdf = pdfs[0] * pdfs[1];
        return Vec2f(d0, d1);
    }

private:
    // Distribution2D Private Data
    std::vector<std::unique_ptr<Distribution1D>> pConditionalV;
    std::unique_ptr<Distribution1D> pMarginal;
};


/// @brief Read an exr file and store the pixel values in a 2d array
/// @param fileName file path of the exr to read
/// @param pixels   the 2d array to receive the pixel data
/// @param width    the width of the image in pixels
/// @param height   the height of the image in pixels
void readExr (const char fileName[],
       Imf::Array2D<Imf::Rgba> &pixels,
       int &width,
       int &height)
{
    Imf::RgbaInputFile file (fileName);
    Imath::Box2i dw = file.dataWindow();

    width  = dw.max.x - dw.min.x + 1;
    height = dw.max.y - dw.min.y + 1;
    pixels.resizeErase(height, width);

    file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
    file.readPixels(dw.min.y, dw.max.y);
}


#endif /* SRC_SAMPLING_H_ */
