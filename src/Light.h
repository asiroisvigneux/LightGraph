// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Light.h
///
/// @author Alexandre Sirois-Vigneux
///
/// @brief This file defines the different light types supported by the renderer.


#ifndef SRC_LIGHT_H_
#define SRC_LIGHT_H_


#include <limits>

#include <openvdb/openvdb.h>

#include "Sampling.h"


using namespace openvdb;


/// @brief Base class to all lights
class Light {

public:
    const Mat4R mXform;
    const Vec3R mColor;
    const float mIntensity;
    const size_t mSamples;

    /// @brief Define the basic components of all lights
    /// @param xform        transformation matrix of the light
    /// @param color        color of the light
    /// @param intensity    intensity of the light
    /// @param samples      number of times that the light will be sampled at render time
    Light(const Mat4R &xform,
          const Vec3R &color,
          const float &intensity,
          const size_t &samples)
        : mXform(xform)
        , mColor(color)
        , mIntensity(intensity)
        , mSamples(samples) { }

    virtual ~Light() {}

    /// @brief returns the amount of light received at a certain point in space
    virtual void illuminate(const Vec3R &hitPos, Vec3R &lightDir, Vec3R &lightIntensity, double &distance) const = 0;
};


/// @brief Point light specialization
class PointLight : public Light {

private:
    const Vec3R mPos;

public:
    /// @brief Create a point light
    /// @param xform        transformation matrix of the light
    /// @param color        color of the light
    /// @param intensity    intensity of the light
    /// @param pos          the position of the light in space
    PointLight(const Mat4R &xform = Mat4R::identity(),
               const Vec3R &color = Vec3R(1),
               const float &intensity = 1.0,
               const Vec3R &pos = Vec3R(0))
        : Light(xform, color, intensity, 1)
        , mPos(xform.transform(pos)) { }

    /// @brief Compute the light received at a certain point in space
    /// @param hitPos           the position sampled in space
    /// @param lightDir         the normalized direction between the light pos and the hitPos
    /// @param lightIntensity   the light intensity
    /// @param distance         the L2 distance between the light and the hitPos
    void illuminate(const Vec3R &hitPos,
                    Vec3R &lightDir,
                    Vec3R &lightIntensity,
                    double &distance) const
    {
        lightDir = (hitPos - mPos);
        float r2 = lightDir.lengthSqr();
        distance = sqrt(r2);
        lightDir /= distance;
        lightIntensity = mColor * mIntensity / (1.0 + 4.0 * M_PI * r2);
    }

};


/// @brief Distant light specialization
class DistantLight : public Light {

private:
    const Vec3R mDir;

public:
    /// @brief Create a distant light
    /// @param xform        transformation matrix of the light
    /// @param color        color of the light
    /// @param intensity    intensity of the light
    /// @param dir          direction of the light
    DistantLight(const Mat4R &xform = Mat4R::identity(),
                 const Vec3R &color = Vec3R(1),
                 const float &intensity = 1.0,
                 const Vec3R &dir = Vec3R(0, 0, -1))
        : Light(xform, color, intensity, 1)
        , mDir(mXform.transform(dir).unit()) { }

    /// @brief Compute the light received at a certain point in space
    /// @param hitPos           the position sampled in space
    /// @param lightDir         the normalized direction between the light pos and the hitPos
    /// @param lightIntensity   the light intensity
    /// @param distance         the L2 distance between the light and the hitPos
    void illuminate(const Vec3R &hitPos,
                    Vec3R &lightDir,
                    Vec3R &lightIntensity,
                    double &distance) const
    {
        lightDir = mDir;
        lightIntensity = mColor * mIntensity;
        distance = std::numeric_limits<double>::max();
    }

};


/// @brief Environment light specialization
class EnvLight : public Light {

private:
    const std::string mExrFile;
    std::unique_ptr<Distribution2D> mDis2D;
    std::unique_ptr<std::mt19937> mRandGen;
    std::unique_ptr<std::uniform_real_distribution<float>> mRandDist;
    std::unique_ptr<Imf::Array2D<Imf::Rgba>> mPixelBuffer;
    int mWidth;
    int mHeight;

public:
    /// @brief Creates an env light from a provided exr texture map
    /// @param xform        transformation matrix of the light
    /// @param color        color of the light
    /// @param intensity    intensity of the light
    /// @param samples      number of times that the light will be sampled at render time
    /// @param exrFile      texture file to use as illumination
    EnvLight(const Mat4R &xform = Mat4R::identity(),
             const Vec3R &color = Vec3R(1),
             const float &intensity = 1.0,
             const int &samples = 3,
             const std::string &exrFile = "")
        : Light(xform, color, intensity, samples)
        , mExrFile(exrFile)
        , mRandGen(new std::mt19937(4564446))
        , mRandDist(new std::uniform_real_distribution<float>(0.0, 1.0))
        , mWidth(0)
        , mHeight(0)
    {
        mPixelBuffer = std::unique_ptr<Imf::Array2D<Imf::Rgba>>(new Imf::Array2D<Imf::Rgba>(mHeight, mWidth));
        readExr(mExrFile.c_str(), *mPixelBuffer, mWidth, mHeight);

        vector<float> Y(mWidth*mHeight, 0.0f);
        for (int i=0; i<mHeight; i++) {
            float sinTheta = std::sin(M_PI * (float)i/(float)(mHeight-1) );
            for (int j=0; j<mWidth; j++) {
                Y[i*mWidth + j] = 0.2126 * (*mPixelBuffer)[i][j].r
                                + 0.7152 * (*mPixelBuffer)[i][j].g
                                + 0.0722 * (*mPixelBuffer)[i][j].b;
                Y[i*mWidth + j] *= sinTheta;
            }
        }

        mDis2D = std::unique_ptr<Distribution2D>(new Distribution2D(&Y[0], mWidth, mHeight));
    }


    /// @brief Compute the light received at a certain point in space
    /// @param hitPos           the position sampled in space
    /// @param lightDir         the normalized direction between the light pos and the hitPos
    /// @param lightIntensity   the light intensity
    /// @param distance         the L2 distance between the light and the hitPos
    void illuminate(const Vec3R &hitPos,
                    Vec3R &lightDir,
                    Vec3R &lightIntensity,
                    double &distance) const {
        Vec2f rand((*mRandDist)(*mRandGen), (*mRandDist)(*mRandGen));
        float pdf = 0.0f;
        Vec2f samplePos = mDis2D->SampleContinuous(rand, &pdf);

        int col_lo = floor(samplePos[0]*mWidth);
        int row_lo = floor(samplePos[1]*mHeight);
        int col_hi = ceil(samplePos[0]*mWidth);
        int row_hi = ceil(samplePos[1]*mHeight);

        float col_ratio = samplePos[0]*mWidth - (float)col_lo;
        float row_ratio = samplePos[1]*mHeight - (float)row_lo;

        float theta = samplePos[0] * M_PI * 2.0f;
        float phi =  samplePos[1] * M_PI;

        double sinPhi = std::sin(phi);

        Vec3R sampleDir((double)std::cos(theta)*sinPhi,
                        (double)std::cos(phi),
                        (double)std::sin(theta)*sinPhi);
        sampleDir = mXform.transform(sampleDir).unit();

        // perform linear interpolation on the env map
        Vec3R pixelColorLL = Vec3R( (*mPixelBuffer)[row_lo][col_lo].r,
                                    (*mPixelBuffer)[row_lo][col_lo].g,
                                    (*mPixelBuffer)[row_lo][col_lo].b );
        Vec3R pixelColorLH = Vec3R( (*mPixelBuffer)[row_lo][col_hi].r,
                                    (*mPixelBuffer)[row_lo][col_hi].g,
                                    (*mPixelBuffer)[row_lo][col_hi].b );
        Vec3R pixelColorHL = Vec3R( (*mPixelBuffer)[row_hi][col_lo].r,
                                    (*mPixelBuffer)[row_hi][col_lo].g,
                                    (*mPixelBuffer)[row_hi][col_lo].b );
        Vec3R pixelColorHH = Vec3R( (*mPixelBuffer)[row_hi][col_hi].r,
                                    (*mPixelBuffer)[row_hi][col_hi].g,
                                    (*mPixelBuffer)[row_hi][col_hi].b );

        Vec3R imgColor = ((1.0f-row_ratio)*(1.0f-col_ratio)) * pixelColorLL
                       + ((1.0f-row_ratio)*col_ratio) * pixelColorLH
                       + (row_ratio*(1.0f-col_ratio)) * pixelColorHL
                       + (row_ratio*col_ratio) * pixelColorHH;

        lightDir = -sampleDir;
        lightIntensity = imgColor * mColor * mIntensity;
        distance = std::numeric_limits<double>::max();
    }

};

#endif /* SRC_LIGHT_H_ */
