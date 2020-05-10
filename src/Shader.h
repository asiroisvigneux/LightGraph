// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Shader.h
///
/// @author Alexandre Sirois-Vigneux
///
/// @brief This file defines the built-in shader used to render the volumes.


#ifndef SRC_SHADER_H_
#define SRC_SHADER_H_


#include <openvdb/openvdb.h>


using namespace openvdb;
using namespace std;


/// @brief Color ramp object to map temperature to color for emissive volumes
class ColorRamp {
    vector<double> mPositions;
    vector<Vec3R> mColors;
    double mIn;
    double mOut;

public:
    ColorRamp()
        : mPositions(vector<double>())
        , mColors(vector<Vec3R>())
        , mIn(0.0)
        , mOut(1.0) { }

    /// @brief Define the color ramp to map temperature to color
    /// @param positions    a vector of position for the ramp between 0 and 1
    /// @param colors       a vector of color associated with each positions
    /// @param in           remap this temperature value to 0 on the ramp
    /// @param out          remap this temperature value to 1 on the ramp
    ColorRamp(const vector<double> &positions,
              const vector<Vec3R> &colors,
              const double &in,
              const double &out)
        : mPositions(positions)
        , mColors(colors)
        , mIn(in)
        , mOut(out) { }

    virtual ~ColorRamp() {}

    /// @brief  Remap an hitPos to the user defined "in" and "out" point of the
    /// temperature range
    /// @param hitPos   position being sampled
    /// @return         returns the position according to the "in" and "out" values
    inline double remap(const double &hitPos) const {
        return min(max((hitPos-mIn) / (mOut-mIn), 0.0), 1.0);
    }

    /// @brief Sample a color from the color ramp
    /// @param hitPos   the position where the ramp is being sampled
    /// @return         return an RGB color
    Vec3R getColor(const double &hitPos) const {

        // return 0 if no valid colorRamp exist
        if (mPositions.size() == 0) {
            return Vec3R(0);
        }

        assert(mColors.size() == mPositions.size());

        const double hitPosRemap = remap(hitPos);

        if (mPositions.size() == 1) return mColors[0];

        size_t prevIdx = 0;
        size_t nextIdx = 0;

        for (size_t i=0; i<mPositions.size(); i++) {
            if (mPositions[i] <= hitPosRemap) {
                prevIdx = i;
            } else {
                nextIdx = i;
                break;
            }
        }

        const double range = mPositions[nextIdx] - mPositions[prevIdx];
        const double blend = (hitPosRemap - mPositions[prevIdx]) / range;

        return mColors[prevIdx] * (1.0 - blend) + mColors[nextIdx] * blend;
    }
};


/// @brief Built in shader that can be modified by the user inputs. It supports
/// vector values scattering, emission, color ramps for temperature remapping, a
/// decoupled control over the density for primary, shadow and scattering
/// contribution and more.
class Shader {

private:
    double mDensityScale;
    double mDensityMax;
    double mShadowDensityScale;
    double mScatterDensityScale;
    double mScatterDensityMin;

    Vec3R mAbsorption;
    Vec3R mScattering;
    Vec3R mVolumeColor;

    Vec3R mAlbedo;
    Vec3R mExtinction;

    double mEmissionScale;
    ColorRamp *mEmissionColorRamp;

    double mScatterScale;
    double mScatterDensityMaskPower;

public:
    Shader()
        : mDensityScale(1.0)
        , mDensityMax(100.0)
        , mShadowDensityScale(1.0)
        , mScatterDensityScale(0.35)
        , mScatterDensityMin(0.025)
        , mAbsorption(Vec3R(0.1))
        , mScattering(Vec3R(1.5))
        , mVolumeColor(Vec3R(1))
        , mAlbedo((mVolumeColor*mScattering) / (mScattering+mAbsorption))
        , mExtinction(-(mScattering+mAbsorption))
        , mEmissionScale(0.0)
        , mEmissionColorRamp(new ColorRamp())
        , mScatterScale(4.0)
        , mScatterDensityMaskPower(0.25)
    { }

    virtual ~Shader() {
        delete mEmissionColorRamp;
    }

    // setters
    void setDensityScale(double DensityScale) {
        mDensityScale = DensityScale;
    }

    void setDensityMax(double DensityMax) {
        mDensityMax = DensityMax;
    }

    void setShadowDensityScale(double ShadowDensityScale) {
        mShadowDensityScale = ShadowDensityScale;
    }

    void setScatterDensityScale(double ScatterDensityScale) {
        mScatterDensityScale = ScatterDensityScale;
    }

    void setScatterDensityMin(double ScatterDensityMin) {
        mScatterDensityMin = ScatterDensityMin;
    }

    void setAbsorption(Vec3R Absorption) {
        mAbsorption = Absorption;
        recomputeExtinction();
    }

    void setScattering(Vec3R Scattering) {
        mScattering = Scattering;
        recomputeAlbedo();
        recomputeExtinction();
    }

    void setVolumeColor(Vec3R VolumeColor) {
        mVolumeColor = VolumeColor;
        recomputeAlbedo();
    }

    void setAlbedo(Vec3R Albedo) {
        mAlbedo = Albedo;
    }

    void setExtinction(Vec3R Extinction) {
        mExtinction = Extinction;
    }

    void setEmissionScale(double EmissionScale) {
        mEmissionScale = EmissionScale;
    }

    void setEmissionColorRamp(ColorRamp *EmissionColorRamp) {
        mEmissionColorRamp = EmissionColorRamp;
    }

    void setScatterScale(double ScatterScale) {
        mScatterScale = ScatterScale;
    }

    void setScatterDensityMaskPower(double ScatterDensityMaskPower) {
        mScatterDensityMaskPower = ScatterDensityMaskPower;
    }

    void recomputeAlbedo() {
        setAlbedo( (mVolumeColor*mScattering) / (mScattering+mAbsorption) );
    }

    void recomputeExtinction() {
        setExtinction( -(mScattering+mAbsorption) );
    }

    // getters
    double getDensityScale() const {
        return mDensityScale;
    }

    double getDensityMax() const {
        return mDensityMax;
    }

    double getShadowDensityScale() const {
        return mShadowDensityScale;
    }

    double getScatterDensityScale() const {
        return mScatterDensityScale;
    }

    double getScatterDensityMin() const {
        return mScatterDensityMin;
    }

    Vec3R getAbsorption() const {
        return mAbsorption;
    }

    Vec3R getScattering() const {
        return mScattering;
    }

    Vec3R getVolumeColor() const {
        return mVolumeColor;
    }

    Vec3R getAlbedo() const {
        return mAlbedo;
    }

    Vec3R getExtinction() const {
        return mExtinction;
    }

    double getEmissionScale() const {
        return mEmissionScale;
    }

    ColorRamp* getEmissionColorRamp() const {
        return mEmissionColorRamp;
    }

    Vec3R getEmissionColor(const double &temp) const {
        return mEmissionColorRamp->getColor(temp);
    }

    double getScatterScale() const {
        return mScatterScale;
    }

    double getScatterDensityMaskPower() const {
        return mScatterDensityMaskPower;
    }

};

#endif /* SRC_SHADER_H_ */
