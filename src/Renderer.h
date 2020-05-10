// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Renderer.h
///
/// @brief Multithreaded raytracer using a rasterized grid to account for the
/// multiple scattering contribution. Also contains a Film and Perspective Camera
/// class with a full frame sensor size.


#ifndef OPENVDB_TOOLS_RAYTRACER_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_RAYTRACER_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/math/BBox.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Math.h>

#include <openvdb/math/DDA.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/Grid.h>
#include <openvdb/tools/Morphology.h>

#include <openvdb/tools/Interpolation.h>
#include <deque>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <OpenEXR/ImfPixelType.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfFrameBuffer.h>

#include "RandGen.h"
#include "SceneSettings.h"
#include "Light.h"
#include "Shader.h"


using namespace openvdb;


/// @brief Helper function to clamp values to a certain range
/// @param v    value to clamp
/// @param lo   lower bound of the value
/// @param hi   upper bound of the value
/// @return     clamped value
template<class T>
constexpr const T& clamp( const T& v, const T& lo, const T& hi ) {
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}


////////////////////////////////////////////////////////////////////////////////
//////////////                        FILM                         /////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief A simple class that allows for concurrent writes to pixels in an image,
/// background initialization of the image, EXR file output.
/// @note  This class was taken as is from OpenVDB
class Film {
public:
    /// @brief Floating-point RGBA components in the range [0, 1].
    /// @details This is our preferred representation for color processing.
    struct RGBA {
        using ValueT = float;

        RGBA() : r(0), g(0), b(0), a(1) {}
        explicit RGBA(ValueT intensity) : r(intensity), g(intensity), b(intensity), a(1) {}
        RGBA(ValueT _r, ValueT _g, ValueT _b, ValueT _a = static_cast<ValueT>(1.0)):
            r(_r), g(_g), b(_b), a(_a)
        {}
        RGBA(double _r, double _g, double _b, double _a = 1.0)
            : r(static_cast<ValueT>(_r))
            , g(static_cast<ValueT>(_g))
            , b(static_cast<ValueT>(_b))
            , a(static_cast<ValueT>(_a))
        {}

        RGBA  operator* (ValueT scale)  const { return RGBA(r*scale, g*scale, b*scale);}
        RGBA  operator+ (const RGBA& rhs) const { return RGBA(r+rhs.r, g+rhs.g, b+rhs.b);}
        RGBA  operator* (const RGBA& rhs) const { return RGBA(r*rhs.r, g*rhs.g, b*rhs.b);}
        RGBA& operator+=(const RGBA& rhs) { r+=rhs.r; g+=rhs.g; b+=rhs.b; a+=rhs.a; return *this;}

        void over(const RGBA& rhs) {
            const float s = rhs.a*(1.0f-a);
            r = a*r+s*rhs.r;
            g = a*g+s*rhs.g;
            b = a*b+s*rhs.b;
            a = a + s;
        }

        ValueT r, g, b, a;
    };


    Film(size_t width, size_t height)
        : mWidth(width), mHeight(height), mSize(width*height), mPixels(new RGBA[mSize])
    {}

    RGBA& pixel(size_t w, size_t h) {
        assert(w < mWidth);
        assert(h < mHeight);
        return mPixels[w + h*mWidth];
    }

    size_t width()       const { return mWidth; }
    size_t height()      const { return mHeight; }
    const RGBA* pixels() const { return mPixels.get(); }

private:
    size_t mWidth, mHeight, mSize;
    std::unique_ptr<RGBA[]> mPixels;
};


////////////////////////////////////////////////////////////////////////////////
//////////////                PERSPECTIVE CAMERA                   /////////////
////////////////////////////////////////////////////////////////////////////////

/// @note  This class was taken from OpenVDB with small modifications
class PerspectiveCamera {
  public:
    /// @brief Constructor
    /// @param rgbFilm      film (i.e. image) defining the rgba pixel resolution
    /// @param directFilm   film (i.e. image) defining the single scatter aov pixel resolution
    /// @param scatterFilm  film (i.e. image) defining the multiple scatter aov pixel resolution
    /// @param emissionFilm film (i.e. image) defining the emission aov pixel resolution
    /// @param rotation     rotation in degrees of the camera in world space
    ///                     (applied in x, y, z order)
    /// @param translation  translation of the camera in world-space units,
    ///                     applied after rotation
    /// @param focalLength  focal length of the camera in mm
    ///                     (the default of 50mm corresponds to Houdini's default camera)
    /// @param aperture     width in mm of the frame, i.e., the visible field
    ///                     (the default 41.2136 mm corresponds to Houdini's default camera)
    /// @param nearPlane    depth of the near clipping plane in world-space units
    /// @param farPlane     depth of the far clipping plane in world-space units
    /// @details If no rotation or translation is provided, the camera is placed
    /// at (0,0,0) in world space and points in the direction of the negative z axis.
    PerspectiveCamera(Film& rgbFilm,
                      Film& directFilm,
                      Film& scatterFilm,
                      Film& emissionFilm,
                      const Vec3R& rotation    = Vec3R(0.0),
                      const Vec3R& translation = Vec3R(0.0),
                      double focalLength = 50.0,
                      double aperture    = 36,
                      double nearPlane   = 1e-3,
                      double farPlane    = std::numeric_limits<double>::max())
        : mRgbFilm(&rgbFilm)
        , mDirectFilm(&directFilm)
        , mScatterFilm(&scatterFilm)
        , mEmissionFilm(&emissionFilm)
        , mScaleWidth(0.5*aperture/focalLength)
        , mScaleHeight(mScaleWidth * double(rgbFilm.height()) / double(rgbFilm.width()))
    {
        assert(nearPlane > 0 && farPlane > nearPlane);
        mScreenToWorld.accumPostRotation(math::X_AXIS, rotation[0] * M_PI / 180.0);
        mScreenToWorld.accumPostRotation(math::Y_AXIS, rotation[1] * M_PI / 180.0);
        mScreenToWorld.accumPostRotation(math::Z_AXIS, rotation[2] * M_PI / 180.0);
        mScreenToWorld.accumPostTranslation(translation);
        this->initRay(nearPlane, farPlane);
    }

    ~PerspectiveCamera() = default;

    // from index to a reference to the pixel value
    Film::RGBA& pixel(size_t i, size_t j, const string filmName) const {
        if (filmName == "rgb") {
            return mRgbFilm->pixel(i, j);
        } else if (filmName == "direct") {
            return mDirectFilm->pixel(i, j);
        } else if (filmName == "scatter") {
            return mScatterFilm->pixel(i, j);
        } else {
            return mEmissionFilm->pixel(i, j);
        }
    }

    size_t width()  const { return mRgbFilm->width(); }
    size_t height() const { return mRgbFilm->height(); }

    // Rotate the camera so its negative z-axis points at xyz and its
    // y axis is in the plane of the xyz and up vectors. In other
    // words the camera will look at xyz and use up as the
    // horizontal direction.
    void lookAt(const Vec3R& xyz, const Vec3R& up = Vec3R(0.0, 1.0, 0.0)) {
        const Vec3R orig = mScreenToWorld.applyMap(Vec3R(0.0));
        const Vec3R dir  = orig - xyz;
        try {
            Mat4d xform = math::aim<Mat4d>(dir, up);
            xform.postTranslate(orig);
            // AffineMap hold the 4x4 matrix as well and pre-computed
            // inverse and other data to accelerate computation later on
            mScreenToWorld = math::AffineMap(xform);
            this->initRay(mRay.t0(), mRay.t1());
        } catch (...) {}
    }

    Vec3R rasterToScreen(double i, double j, double z) const {
        return Vec3R( (2 * i / double(mRgbFilm->width()) - 1)  * mScaleWidth,
                      (1 - 2 * j / double(mRgbFilm->height())) * mScaleHeight, z );
    }

    /// @brief Return a Ray in world space given the pixel indices and
    /// optional offsets in the range [0,1]. An offset of 0.5 corresponds
    /// to the center of the pixel.
    math::Ray<double> getRay( size_t i,
                              size_t j,
                              RandGen rand,
                              double iOffset = 0.5,
                              double jOffset = 0.5) const
    {
        math::Ray<double> ray(mRay);
        Vec3R dir = rasterToScreen(Real(i) + iOffset + (rand(0)-0.5), Real(j) + jOffset + (rand(0)-0.5), -1.0);
        dir = mScreenToWorld.applyJacobian(dir);
        dir.normalize();
        ray.scaleTimes(1.0/dir.dot(ray.dir()));
        ray.setDir(dir);
        return ray;
    }

private:
    void initRay(double t0, double t1)
    {
        mRay.setTimes(t0, t1);
        mRay.setEye(mScreenToWorld.applyMap(Vec3R(0.0)));
        // the jacobian is the transposed inverse of the 3x3 matrix of
        // mScreenToWorld, this is the correct way to transform direction vectors
        mRay.setDir(mScreenToWorld.applyJacobian(Vec3R(0.0, 0.0, -1.0)));
    }

    Film* mRgbFilm;
    Film* mDirectFilm;
    Film* mScatterFilm;
    Film* mEmissionFilm;
    double mScaleWidth, mScaleHeight;
    math::Ray<double> mRay;
    math::AffineMap mScreenToWorld;
};


////////////////////////////////////////////////////////////////////////////////
//////////////               VOLUME RAY INTERSECTOR                /////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief This class provides the public API for intersecting a ray
/// with a generic (e.g. density) volume.
/// @details Internally it performs the actual hierarchical tree node traversal.
/// @note  This class was taken from OpenVDB with small modifications
template<typename GridT>
class VolumeRayIntersector
{
public:
    static const int NodeLevel = GridT::TreeType::RootNodeType::ChildNodeType::LEVEL;
    using RayT = math::Ray<Real>;

    using GridType = GridT;
    using RayType = RayT;
    using RealType = typename RayT::RealType;
    using RootType = typename GridT::TreeType::RootNodeType;
    using TreeT = tree::Tree<typename RootType::template ValueConverter<bool>::Type>;

    static_assert(NodeLevel >= 0 && NodeLevel < int(TreeT::DEPTH)-1, "NodeLevel out of range");

    /// @brief Grid constructor
    /// @param grid Generic grid to intersect rays against.
    /// @param dilationCount The number of voxel dilations performed
    /// on (a boolean copy of) the input grid. This allows the
    /// intersector to account for the size of interpolation kernels
    /// in client code.
    /// @throw RuntimeError if the voxels of the grid are not uniform
    VolumeRayIntersector(const GridT& grid, int dilationCount = 0) // @suppress("Class members should be properly initialized")
        : mIsMaster(true)
        , mTree(new TreeT(grid.tree(), false, TopologyCopy()))
        , mGrid(&grid)
        , mAccessor(*mTree)
    {
        if (!grid.hasUniformVoxels() ) {
            OPENVDB_THROW(RuntimeError,
                          "VolumeRayIntersector only supports uniform voxels!");
        }

        // Dilate active voxels to better account for the size of interpolation kernels
        tools::dilateVoxels(*mTree, dilationCount);

        mTree->root().evalActiveBoundingBox(mBBox, /*visit individual voxels*/false);

        mBBox.max().offset(1);//padding so the bbox of a node becomes (origin,origin + node_dim)
    }

    /// @brief Shallow copy constructor
    /// @warning This copy constructor creates shallow copies of data
    /// members of the instance passed as the argument. For
    /// performance reasons we are not using shared pointers (their
    /// mutex-lock impairs multi-threading).
    VolumeRayIntersector(const VolumeRayIntersector& other)
        : mIsMaster(false)
        , mTree(other.mTree)//shallow copy
        , mGrid(other.mGrid)//shallow copy
        , mAccessor(*mTree)//initialize new (vs deep copy)
        , mRay(other.mRay)//deep copy
        , mTmax(other.mTmax)//deep copy
        , mBBox(other.mBBox)//deep copy
    { }

    /// @brief Destructor
    ~VolumeRayIntersector() { if (mIsMaster) delete mTree; }

    /// @brief Return @c false if the index ray misses the bbox of the grid.
    /// @param iRay Ray represented in index space.
    /// @warning Call this method (or setWorldRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required.
    inline bool setIndexRay(const RayT& iRay) {
        mRay = iRay;
        const bool hit = mRay.clip(mBBox);
        if (hit) mTmax = mRay.t1();
        return hit;
    }

    /// @brief Return @c false if the world ray misses the bbox of the grid.
    /// @param wRay Ray represented in world space.
    /// @warning Call this method (or setIndexRay) before the ray
    /// traversal starts and use the return value to decide if further
    /// marching is required.
    /// @details Since hit times are computed with respect to the ray
    /// represented in index space of the current grid, it is
    /// recommended that either the client code uses getIndexPos to
    /// compute index position from hit times or alternatively keeps
    /// an instance of the index ray and instead uses setIndexRay to
    /// initialize the ray.
    inline bool hitGridBbox(const RayT& wRay) {
        return this->setIndexRay(wRay.worldToIndex(*mGrid));
    }

    inline typename RayT::TimeSpan march() {
        const typename RayT::TimeSpan t = mHDDA.march(mRay, mAccessor); // mHDDA solves the valid active segments in space
        if (t.t1>0) mRay.setTimes(t.t1 + math::Delta<RealType>::value(), mTmax); // update t0 and t1 for the next query
        return t;
    }

    /// @brief Return true if the ray intersects active values,
    /// i.e. either active voxels or tiles. Only when a hit is
    /// detected are t0 and t1 updated with the corresponding entry
    /// and exit times along the INDEX ray!
    /// @note Note that t0 and t1 are only resolved at the node level
    /// (e.g. a LeafNode with active voxels) as opposed to the individual
    /// active voxels.
    /// @param t0 If the return value > 0 this is the time of the
    /// first hit of an active tile or leaf.
    /// @param t1 If the return value > t0 this is the time of the
    /// first hit (> t0) of an inactive tile or exit point of the
    /// BBOX for the leaf nodes.
    /// @warning t0 and t1 are computed with respect to the ray represented in
    /// index space of the current grid, not world space!
    inline bool march(RealType& t0, RealType& t1) {
        const typename RayT::TimeSpan t = this->march();
        t.get(t0, t1);
        return t.valid(); // check if the distance between t1-t0 > eps
    }

    /// @brief Return the floating-point world position along the
    /// current index ray at the specified time.
    inline Vec3R getWorldPos(RealType time) const {
        return mGrid->indexToWorld(mRay(time));
    }

    /// @brief Return a const reference to the input grid.
    const GridT& grid() const {
        return *mGrid;
    }

private:
    using AccessorT = typename tree::ValueAccessor<const TreeT,/*IsSafe=*/false>;

    const bool      mIsMaster;
    TreeT*          mTree;
    const GridT*    mGrid;
    AccessorT       mAccessor;
    RayT            mRay;
    RealType        mTmax;
    math::CoordBBox mBBox;
    math::VolumeHDDA<TreeT, RayType, NodeLevel> mHDDA;

}; // VolumeRayIntersector


////////////////////////////////////////////////////////////////////////////////
//////////////                   VOLUME RENDER                     /////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief A simple multithreaded volume render that supports single and multiple
/// scattering as well as emission from temperature grids.
template<typename IntersectorT, typename IntersectorU>
class VolumeRender {

public:
    using GridType = typename IntersectorT::GridType;
    using RayType = typename IntersectorT::RayType;
    using ValueType = typename GridType::ValueType;
    using AccessorType = typename GridType::ConstAccessor;
    using SamplerType = tools::GridSampler<AccessorType, tools::BoxSampler>;

    using GridTypeV = typename IntersectorU::GridType;
    using RayTypeV = typename IntersectorU::RayType;
    using ValueTypeV = typename GridTypeV::ValueType;
    using AccessorTypeV = typename GridTypeV::ConstAccessor;
    using SamplerTypeV = tools::GridSampler<AccessorTypeV, tools::BoxSampler>;

    static_assert(std::is_floating_point<ValueType>::value,
        "VolumeRender requires a floating-point-valued grid");

    /// @brief Constructor taking multiple intersectors, a camera and a shader.
    VolumeRender(const IntersectorT& densityInter,
                 PerspectiveCamera& camera,
                 const IntersectorU& scatterInter,
                 const IntersectorT& tempInter,
                 const SceneSettings &opts,
                 const Shader &shader);

    /// @brief Copy constructor which creates a thread-safe clone
    VolumeRender(const VolumeRender& other);

    /// @brief Perform the actual multithreaded volume rendering.
    void render(bool threaded=true) const;

    /// @brief
    void operator()(const tbb::blocked_range<size_t>& range) const;

private:
    AccessorType mAccessor, mTempAccessor;
    AccessorTypeV mScatterAccessor;
    PerspectiveCamera*  mCamera;
    std::unique_ptr<IntersectorT> mPrimary, mShadow, mTemp;
    std::unique_ptr<IntersectorU> mScatter;
    double mProgress;
    const Shader &mShader;
    const SceneSettings &mOpts;
}; // VolumeRender

/// @brief Constructor taking multiple intersectors, a camera and a shader.
/// @param densityInter intersector used to test against the density grid
/// @param camera       camera used to render
/// @param scatterInter intersector used to test against the scatter grid
/// @param tempInter    intersector used to test against the temperature grid
/// @param opts         renderSettings options
/// @param shader       shader used to render the volume
template<typename IntersectorT, typename IntersectorU>
inline VolumeRender<IntersectorT, IntersectorU>::VolumeRender(const IntersectorT& densityInter,
                                                              PerspectiveCamera& camera,
                                                              const IntersectorU& scatterInter,
                                                              const IntersectorT& tempInter,
                                                              const SceneSettings &opts,
                                                              const Shader &shader)
    : mAccessor(densityInter.grid().getConstAccessor()) // @suppress("Symbol is not resolved")
    , mScatterAccessor(scatterInter.grid().getConstAccessor()) // @suppress("Symbol is not resolved")
    , mTempAccessor(tempInter.grid().getConstAccessor()) // @suppress("Symbol is not resolved")
    , mCamera(&camera)
    , mPrimary(new IntersectorT(densityInter))
    , mShadow(new IntersectorT(densityInter))
    , mScatter(new IntersectorU(scatterInter))
    , mTemp(new IntersectorT(tempInter))
    , mProgress(0.0)
    , mShader(shader)
    , mOpts(opts)
    { }

/// @brief Copy constructor which creates a thread-safe clone
/// @param other reference VolumeRender as template
template<typename IntersectorT, typename IntersectorU>
inline VolumeRender<IntersectorT, IntersectorU>::VolumeRender(const VolumeRender& other)
    : mAccessor(other.mAccessor) // @suppress("Symbol is not resolved")
    , mScatterAccessor(other.mScatterAccessor) // @suppress("Symbol is not resolved")
    , mTempAccessor(other.mTempAccessor) // @suppress("Symbol is not resolved")
    , mCamera(other.mCamera)
    , mPrimary(new IntersectorT(*(other.mPrimary)))
    , mShadow(new IntersectorT(*(other.mShadow)))
    , mScatter(new IntersectorU(*(other.mScatter)))
    , mTemp(new IntersectorT(*(other.mTemp)))
    , mProgress(0.0)
    , mShader(other.mShader)
    , mOpts(other.mOpts)
    { }

/// @brief Dispatch ranges of pixel to separate threads for rendering
/// @param threaded boolean enabling multithreaded render
template<typename IntersectorT, typename IntersectorU>
inline void VolumeRender<IntersectorT, IntersectorU>::render(bool threaded) const
{
    tbb::blocked_range<size_t> range(0, mCamera->height());
    threaded ? tbb::parallel_for(range, *this) : (*this)(range);
}

/// @brief the actual rendering of the final image happens here
/// @param range range of pixel assigned to a single thread for rendering
template<typename IntersectorT, typename IntersectorU>
void VolumeRender<IntersectorT, IntersectorU>::operator()(const tbb::blocked_range<size_t>& range) const
{
    // we setup a sampler to query the grid in worldspace
    SamplerType sampler(mAccessor, mShadow->grid().transform());
    SamplerTypeV scatterSampler(mScatterAccessor, mScatter->grid().transform());
    SamplerType tempSampler(mTempAccessor, mTemp->grid().transform());

    // Any variable prefixed with p (or s) means it's associated with a primary
    // (or shadow) ray
    const Vec3R One(1.0);
    const Vec3R invScatter(1.0/mShader.getScattering());

    RandGen rng(1);
    const double jitterAmp = 0.5;

        // loop over image height specified by the range object of TBB
        // (not the full height)
        // it goes a single line at the time
        for (size_t j=range.begin(), je = range.end(); j<je; ++j) {

            // loop over the width of the image
            for (size_t i=0, ie = mCamera->width(); i<ie; ++i) {

                // initialize the pixels to be black to start with
                Film::RGBA& rgbPix = mCamera->pixel(i, j, "rgb");
                Film::RGBA& dirPix = mCamera->pixel(i, j, "direct");
                Film::RGBA& scatPix = mCamera->pixel(i, j, "scatter");
                Film::RGBA& emitPix = mCamera->pixel(i, j, "emission");
                rgbPix.a = rgbPix.r = rgbPix.g = rgbPix.b = 0;
                dirPix.a = dirPix.r = dirPix.g = dirPix.b = 0;
                scatPix.a = scatPix.r = scatPix.g = scatPix.b = 0;
                emitPix.a = emitPix.r = emitPix.g = emitPix.b = 0;

                // init antialias vectors
                Vec3R pTransAA(0.0), pLumiAA(0.0), directLumiAA(0.0), scatterLumiAA(0.0), emitLumiAA(0.0);

                // for each AA samples
                double ratioAA = 1.0/(double)mOpts.samples;
                for (size_t sampleAA=0; sampleAA<mOpts.samples; sampleAA++) {
                    // create a primary ray from the camera
                    RayType pRay = mCamera->getRay(i, j, rng);// Primary ray

                    Vec3R pTrans(1.0);

                    // check if the ray intersects with the bbox of the grid
                    // if it does not skip this iteration
                    // this also sets pRay as the ray member of mPrimary for
                    // marching later on
                    if( !mPrimary->hitGridBbox(pRay)) {
                        pTransAA += pTrans * ratioAA; // ensure black alpha when vdb bbox is missed
                        continue;
                    }

                    // init 2 vector to accumulate primary transmitance and luminausity
                    Vec3R pLumi(0.0), directLumi(0.0), scatterLumi(0.0), emitLumi(0.0),
                                                   tmpDirectLumi(0.0), tmpScatterLumi(0.0), tmpEmitLumi(0.0);

                    // start marching the grid for the primary rays, this will
                    // return false if no more voxels are intersected by the ray
                    // otherwise it updates pT0 and pT1. The march function will
                    // only dispatch valid ranges of active voxels
                    Real pT0, pT1;
                    while (mPrimary->march(pT0, pT1)) {

                        bool pDropout = false;

                        // we march along the ray between pT0 and pT1
                        // to make sure we sample equaly along the ray we use ceil (no samples between increments)
                        double pStep = mOpts.step[0] * (1.0 + (rng(0)-0.5)*jitterAmp);
                        Real pT = mOpts.step[0]*ceil(pT0/mOpts.step[0]);
                        for (; pT <= pT1; pT += pStep) {

                            // get the world position of the ray at time pT
                            Vec3R pPos = mPrimary->getWorldPos(pT);

                            // sample the density at the world position
                            const Real pDensity = min(sampler.wsSample(pPos) * mShader.getDensityScale(),
                                                      mShader.getDensityMax());
                            const Vec3R pScatter = scatterSampler.wsSample(pPos);

                            Vec3R scatter(pScatter * invScatter
                                          * pow(clamp(pDensity, 0.0, 1.0), mShader.getScatterDensityMaskPower())
                                          * mShader.getScatterScale());

                            const Real pTemp = tempSampler.wsSample(pPos);
                            Vec3R emission(mShader.getEmissionColor(pTemp)
                                           * mShader.getEmissionScale());

                            // if the density is too low skip this iteration
                            if (pDensity < mOpts.cutoff) continue;

                            // compute the delta transmitance using Lambert-Beers law (P.176 PVR)
                            const Vec3R dT = math::Exp(mShader.getExtinction() * pDensity * pStep);

                            // loop over each lights
                            for (size_t k=0; k<mOpts.light.size(); k++) {
                                // loop for all light samples
                                Vec3R tmpLumi(0);
                                for (size_t s=0; s<mOpts.light[k]->mSamples; s++) { // @suppress("Field cannot be resolved")

                                    // init the transmitance of the shadow ray
                                    Vec3R sTrans(1.0);

                                    Vec3R lightDir;
                                    Vec3R lightIntensity;
                                    double distance;
                                    mOpts.light[k]->illuminate(pPos, lightDir, lightIntensity, distance); // @suppress("Invalid arguments")

                                    // create a ray in the light direction
                                    RayType sRay(pPos, -lightDir, math::Delta<double>::value(), distance); //Shadow ray

                                    // move the shadow ray in word space and check if it hits the volume bbox
                                    if( !mShadow->hitGridBbox(sRay)) continue;

                                    Real sT0, sT1;
                                    while (mShadow->march(sT0, sT1)) {

                                        bool sDropout = false;

                                        double sStep = mOpts.step[1] * (1.0 + (rng(0)-0.5)*jitterAmp);
                                        Real sT = mOpts.step[1]*ceil(sT0/mOpts.step[1]);
                                        for (; sT <= sT1; sT += sStep) {
                                            const Real sDensity = min(sampler.wsSample(mShadow->getWorldPos(sT))
                                                                      * mShader.getDensityScale()
                                                                      * mShader.getShadowDensityScale(),
                                                                      mShader.getDensityMax());
                                            if (sDensity < mOpts.cutoff) continue;
                                            // accumulate opacity by multiplication
                                            sTrans *= math::Exp(mShader.getExtinction() * sDensity * sStep);
                                            if (sTrans.lengthSqr()<mOpts.cutoff) {
                                                sTrans.setZero(); // set the shadow to full black if passed cutoff
                                                sDropout=true;
                                                break;
                                            } //Terminate sRay
                                            sStep = mOpts.step[1] * (1.0 + (rng(0)-0.5)*jitterAmp);
                                        } // Integration over shadow segment
                                        if (sDropout) break; //Terminate sRay
                                    } // Shadow ray march
                                    tmpLumi += lightIntensity * sTrans;
                                } // light sample loop
                                tmpDirectLumi = mShader.getAlbedo()
                                                * (tmpLumi/(double)mOpts.light[k]->mSamples) // @suppress("Field cannot be resolved")
                                                * pTrans * (One-dT);
                                pLumi += tmpDirectLumi;
                                directLumi += tmpDirectLumi;
                            } // light loop

                            tmpScatterLumi = scatter  * pTrans * (One-dT);
                            tmpEmitLumi = emission  * pTrans * (One-dT);
                            scatterLumi += tmpScatterLumi;
                            emitLumi += tmpEmitLumi;

                            pLumi += tmpScatterLumi + tmpEmitLumi; //
                            pTrans *= dT;
                            if (pTrans.lengthSqr()<mOpts.cutoff) { pDropout = true; break; } // Terminate Ray

                            pStep = mOpts.step[0] * (1.0 + (rng(0)-0.5)*jitterAmp);
                        } // Integration over primary segment
                        if (pDropout) break; // Terminate Ray
                    } // Primary ray march
                    pLumiAA += pLumi * ratioAA;
                    directLumiAA += directLumi * ratioAA;
                    scatterLumiAA += scatterLumi * ratioAA;
                    emitLumiAA += emitLumi * ratioAA;
                    pTransAA += pTrans * ratioAA;
                } // AA samples

                // write the pixel values to the individual films
                rgbPix.r = static_cast<Film::RGBA::ValueT>(pLumiAA[0]);
                rgbPix.g = static_cast<Film::RGBA::ValueT>(pLumiAA[1]);
                rgbPix.b = static_cast<Film::RGBA::ValueT>(pLumiAA[2]);
                rgbPix.a = static_cast<Film::RGBA::ValueT>(1.0f - pTransAA.sum()/3.0f);

                dirPix.r = static_cast<Film::RGBA::ValueT>(directLumiAA[0]);
                dirPix.g = static_cast<Film::RGBA::ValueT>(directLumiAA[1]);
                dirPix.b = static_cast<Film::RGBA::ValueT>(directLumiAA[2]);

                scatPix.r = static_cast<Film::RGBA::ValueT>(scatterLumiAA[0]);
                scatPix.g = static_cast<Film::RGBA::ValueT>(scatterLumiAA[1]);
                scatPix.b = static_cast<Film::RGBA::ValueT>(scatterLumiAA[2]);

                emitPix.r = static_cast<Film::RGBA::ValueT>(emitLumiAA[0]);
                emitPix.g = static_cast<Film::RGBA::ValueT>(emitLumiAA[1]);
                emitPix.b = static_cast<Film::RGBA::ValueT>(emitLumiAA[2]);

         } // Horizontal pixel scan
       } // Vertical pixel scan
}

#endif // OPENVDB_TOOLS_RAYTRACER_HAS_BEEN_INCLUDED
