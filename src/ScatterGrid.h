// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ScatterGrid.h
///
/// @author Alexandre Sirois-Vigneux
///
/// @brief This file is responsible for rasterizing the solved LightGraph point
/// clouds into a low-resolution multiple scattering grid.


#ifndef SRC_SCATTERGRID_H_
#define SRC_SCATTERGRID_H_

//gloabl constants
const float TRUNCATED_GAUSSIAN_TAIL = 0.1;


/// @brief This class handles the rasterization of the multiple scattering to a grid
struct ScatterGrid {
    const openvdb::Vec3fGrid::Ptr &mTargetGrid;
    const openvdb::tools::PointIndexGrid::Ptr &mPointIndexGrid;
    const openvdb::points::PointAttributeVector<openvdb::Vec3R> &mPositionsWrapper;
    const vector<Vec3R> &mPointCloudCombined;
    const vector<Vec3R> &mDiffusePointCloudCombined;
    const size_t &mPtsPerKernel;

    /// @brief Rasterize the LightGraphs point cloud to a low resolution vdb grid
    /// @param targetGrid                   pointer to an empty vdb scatter grid
    /// @param pointIndexGrid               pointer to an empty vdb point grid
    /// @param positionsWrapper             point cloud position wrapper needed for the vdb point grid
    /// @param pointCloudCombined           array that stores the point positions
    /// @param diffusePointCloudCombined    array that stores the multiple scatter values
    /// @param ptsPerKernel                 number of points expected to fit inside the kernel
    ScatterGrid(const openvdb::Vec3fGrid::Ptr &targetGrid,
                    const openvdb::tools::PointIndexGrid::Ptr &pointIndexGrid,
                    const openvdb::points::PointAttributeVector<openvdb::Vec3R> &positionsWrapper,
                    const vector<Vec3R> &pointCloudCombined,
                    const vector<Vec3R> &diffusePointCloudCombined,
                    const size_t &ptsPerKernel)
        : mTargetGrid(targetGrid)
        , mPointIndexGrid(pointIndexGrid)
        , mPositionsWrapper(positionsWrapper)
        , mPointCloudCombined(pointCloudCombined)
        , mDiffusePointCloudCombined(diffusePointCloudCombined)
        , mPtsPerKernel(ptsPerKernel) { }

    /// @brief Rasterize LightGraphs to a scatter grid
    void operator()(const openvdb::Vec3fGrid::ValueOnIter& iter) const;

    /// @brief Create the scatter grid
    static void buildScatterGrid(openvdb::Vec3fGrid::Ptr scatterGrid,
                                 const openvdb::FloatGrid::Ptr grid,
                                 const SceneSettings &opts);

};

/// @brief Iterate over all active voxels and store the estimated multiple
/// scattering using a truncated Gaussian kernel.
/// @param iter the current active voxel
void ScatterGrid::operator()(const openvdb::Vec3fGrid::ValueOnIter& iter) const {

    Coord indexCoord = iter.getCoord();
    openvdb::Vec3R worldPos = mTargetGrid->transform().indexToWorld(indexCoord);

    // Search for points within the box.
    openvdb::tools::PointIndexGrid::ConstAccessor ptAcc = mPointIndexGrid->getConstAccessor(); // @suppress("Invalid arguments")

    const double smallestWeigth = TRUNCATED_GAUSSIAN_TAIL; // truncated gaussian kernel

    const double voxelSize = mPointIndexGrid->transform().voxelSize()[0];
    const double voxelSizeCubed = voxelSize*voxelSize*voxelSize;
    const double radius = pow((3.0*(double)mPtsPerKernel*voxelSizeCubed)/(32.0*M_PI), 1.0/3.0);

    const double sigmaSquared = -(radius*radius)/(2*log(smallestWeigth));

    openvdb::tools::PointIndexIterator<openvdb::tools::PointIndexTree> pointIndexIter;
    pointIndexIter.worldSpaceSearchAndUpdate<
            openvdb::points::PointAttributeVector<openvdb::Vec3R> >(worldPos,
            radius, ptAcc, mPositionsWrapper, mPointIndexGrid->transform());

    Vec3R scatter(0);
    double totalWeigths = 0.0;
    for (; pointIndexIter; ++pointIndexIter) {

        Vec3s diff = worldPos - mPointCloudCombined.at(*pointIndexIter);

        double weigth = math::Exp(-(1.0/(2.0*sigmaSquared)) * diff.lengthSqr());
        totalWeigths += weigth;
        scatter += mDiffusePointCloudCombined.at(*pointIndexIter) * weigth;
    }
    if (totalWeigths > 0.0) {
        scatter /= totalWeigths;
    }

    iter.setValue(scatter);

}

/// @brief Create the scatter grid using the topology of the density grid as a
/// template
/// @param scatterGrid  null pointer to be attached to the scatter grid once created
/// @param grid         density grid provided as a topology ref
/// @param opts         render settings
void ScatterGrid::buildScatterGrid(openvdb::Vec3fGrid::Ptr scatterGrid,
                                   const openvdb::FloatGrid::Ptr grid,
                                   const SceneSettings &opts) {

    const openvdb::Vec3f background(0.0f), setValue(1.0f);
    openvdb::Vec3fTree::Ptr targetTree(new openvdb::Vec3fTree(grid->tree(), background, setValue, openvdb::TopologyCopy()));

    openvdb::Vec3fGrid::Ptr scatterGridHi = openvdb::Vec3fGrid::create(targetTree);

    shared_ptr<openvdb::math::Transform> xformCopy(new openvdb::math::Transform(grid->transform()));
    scatterGridHi->setTransform(xformCopy);

    double scatterGridVoxelSize = scatterGridHi->transform().voxelSize()[0] * (double)opts.scatterGridResFactor;
    scatterGrid->setTransform(openvdb::math::Transform::createLinearTransform(scatterGridVoxelSize));

    // Get the source and target grids' index space to world space transforms.
    const openvdb::math::Transform
        &sourceXform = scatterGridHi->transform(),
        &targetXform = scatterGrid->transform();
    // Compute a source grid to target grid transform.
    // (For this example, we assume that both grids' transforms are linear,
    // so that they can be represented as 4 x 4 matrices.)
    openvdb::Mat4R xform =
        sourceXform.baseMap()->getAffineMap()->getMat4() *
        targetXform.baseMap()->getAffineMap()->getMat4().inverse();
    // Create the transformer.
    openvdb::tools::GridTransformer transformer(xform);

    // Resample using trilinear interpolation.
    transformer.transformGrid<openvdb::tools::BoxSampler, openvdb::Vec3fGrid>(*scatterGridHi, *scatterGrid);

    openvdb::tools::dilateActiveValues(scatterGrid->tree(),
                                       1, /*iterations*/
                                       openvdb::tools::NN_FACE_EDGE_VERTEX,
                                       openvdb::tools::PRESERVE_TILES);

    if (opts.verbose) {
        std::cout << "Multiple Scattering Grid Voxel size: " << scatterGrid->transform().voxelSize() << std::endl;
    }
}


#endif /* SRC_SCATTERGRID_H_ */
