// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file LightGraph.h
///
/// @author Alexandre Sirois-Vigneux
///
/// @brief This file contains the LightGraph class responsible for computing the
/// multiple scattering estimate.


#ifndef SRC_LIGHTGRAPH_H_
#define SRC_LIGHTGRAPH_H_


#include <boost/format.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>

#include "RandGen.h"

// gloabl constants
const uint32_t PC_SIZE_HARD_LIMIT = 1000;
const float CONNECT_SEARCH_RADIUS_MULT = 1.7;
const float PC_DENSITY_CUTOFF = 0.001;
const uint32_t TARGET_PC_SIZE = 750;
const uint32_t PC_GROWTH_NUMBER_OF_ESTIMATE = 100;
const float PC_GROWTH_FUNC_POWER = 0.17;


/// @brief Compute the multiple scattering estimate
class LightGraph {

private:
    vector<float> &mMinDistVec;
    const uint32_t &mScatterCount;
    const openvdb::math::BBox<Vec3R> &mBbox;
    const openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> &mFastSampler;
    const openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> &mTempFastSampler;
    RandGen &mRng;
    const uint32_t &mMaxConnect;
    const openvdb::FloatGrid::Ptr &mGrid;
    const SceneSettings &mOpts;
    vector< vector<Vec3R> >  &mPointCloudGather;
    vector< vector<Vec3R> > &mDiffusePointGather;
    const Shader &mShader;
    const string &mVdbFilename;

public:
    /// @brief Constructor for the LightGraph iterations
    /// @param minDistVec          vector holding the minimum distance between
    ///                            each points of the point cloud (per thread)
    /// @param scatterCount        number of point to thrown in the bbox
    /// @param bbox                bounding box of the density grid being rendered
    /// @param fastSampler         sampler object to query the density grid
    /// @param tempFastSampler     sampler object to query the temperature grid
    /// @param rng                 thread safe random number generator
    /// @param maxConnect          maximum number of connection per points in
    ///                            the created graph
    /// @param grid                density grid being sampled
    /// @param opts                render settings
    /// @param pointCloudGather    vector to store all LightGraph iteration's position
    /// @param diffusePointGather  vector to store all LightGraph iteration's color
    /// @param shader              shader used to render the volume
    /// @param vdbFilename         file path of the input vdb file
    LightGraph(vector<float> &minDistVec,
               const uint32_t &scatterCount,
               const openvdb::math::BBox<Vec3R> &bbox,
               const openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> &fastSampler,
               const openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> &tempFastSampler,
               RandGen &rng,
               const uint32_t &maxConnect,
               const openvdb::FloatGrid::Ptr &grid,
               const SceneSettings &opts,
               vector< vector<Vec3R> > &pointCloudGather,
               vector< vector<Vec3R> > &diffusePointGather,
               const Shader &shader,
               const string &vdbFilename)
        : mMinDistVec(minDistVec)
        , mScatterCount(scatterCount)
        , mBbox(bbox)
        , mFastSampler(fastSampler)
        , mTempFastSampler(tempFastSampler)
        , mRng(rng)
        , mMaxConnect(maxConnect)
        , mGrid(grid)
        , mOpts(opts)
        , mPointCloudGather(pointCloudGather)
        , mDiffusePointGather(diffusePointGather)
        , mShader(shader)
        , mVdbFilename(vdbFilename) { }

    /// @brief Multithreaded creation of LightGraphs
    void operator() ( const tbb::blocked_range<std::size_t> &r ) const;

    /// @brief Dart throwing of points inside the density bounding box
    void scatterPoints(vector<Vec3R> &pointCloud,
                       float &minDist,
                       uint32_t &pcSize,
                       const size_t iter) const;

    /// @brief Creation of the graph based on the scattered points
    void buildGraph(unique_ptr<bool[]> &graph,
                    const uint32_t &pcSize,
                    const vector<Vec3R> &pointCloud,
                    const float &searchRadiusConnectSquared,
                    uint32_t &connectCount) const;

    /// @brief Utility function to print portions of the graph
    void printPartialGraph(const unique_ptr<float[]> &graph,
                           uint32_t graphWidth,
                           uint32_t start,
                           uint32_t end) const;

    /// @brief Ray march each edges of the graph to compute transmitance
    void raymarchConnection(const unique_ptr<bool[]> &graph,
                            vector<Vec3R> &pathGraph,
                            uint32_t pcSize,
                            const vector<Vec3R> &pointCloud) const;

    /// @brief Compute direct lighting per graph vertex
    void computeLighting(uint32_t pcSize,
                         const vector<Vec3R> &pointCloud,
                         unique_ptr<Vec3R[]> &lightPointCloud) const;

    /// @brief Solve the graph with shortest path finding to figure out the
    /// least occluded path between each pair of points in the graph
    void floydAlgo(vector<Vec3R> &graph, const uint32_t graphWidth) const;

    /// @brief Diffuse radiance in the graph according to the solution computed
    /// with shortest path finding
    void diffuseLight(vector<Vec3R> &pathGraph,
                      const unique_ptr<Vec3R[]> &lightPC,
                      vector<Vec3R> &diffusePC,
                      uint32_t pcSize) const;

    /// @brief Output the geometry generated by the LightGraph as ASCII files
    /// for visualization or debug purpose
    void outputDebugGeo(const vector<Vec3R> &pointCloud,
                        const unique_ptr<bool[]> &graph,
                        const vector<Vec3R> &pathGraph,
                        const unique_ptr<Vec3R[]> &lightPointCloud,
                        const vector<Vec3R> &diffusePointCloud,
                        uint32_t pcSize,
                        size_t iter) const;

    /// @brief Create the vdb point grid for fast point cloud lookup when
    /// rasterizing the LightGraphs to the scatter grid
    static openvdb::tools::PointIndexGrid::Ptr
    buildVdbPointGrid(openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper,
                      vector<Vec3R> diffusePointCloudCombined);

};


/// @brief Multithreaded creation of LightGraphs
/// @param r the index of a LightGraph iteration
void LightGraph::operator() ( const tbb::blocked_range<std::size_t> &r ) const {
    uint32_t pcSize = 0;
    uint32_t connectCount = 0;

    // do dart throwing to fill the volume with equally distanced points
    for (size_t i=r.begin(); i<r.end(); ++i) {
        this->scatterPoints(mPointCloudGather[i], mMinDistVec[i], pcSize, i);
        if (mOpts.verbose) {
            cout << "PointCloud Size: " << pcSize << endl;
        }

        // max distance to look for connections in the graph
        const float searchRadiusConnect = mMinDistVec[i] * CONNECT_SEARCH_RADIUS_MULT;
        const float searchRadiusConnectSquared = searchRadiusConnect * searchRadiusConnect;

        unique_ptr<bool[]> graph(new bool[pcSize * pcSize] { 0 });
        this->buildGraph(graph, pcSize, mPointCloudGather[i],
                         searchRadiusConnectSquared, connectCount);
        if (mOpts.verbose) {
            cout << "Connection Count: " << connectCount << endl;
        }

        vector<Vec3R> pathGraph(pcSize*pcSize, Vec3R(0));
        this->raymarchConnection(graph, pathGraph, pcSize, mPointCloudGather[i]);

        this->floydAlgo(pathGraph, pcSize);

        unique_ptr<Vec3R[]> lightPointCloud(new Vec3R[pcSize]);
        this->computeLighting(pcSize, mPointCloudGather[i], lightPointCloud);

        this->diffuseLight(pathGraph, lightPointCloud, mDiffusePointGather[i], pcSize);

        if (mOpts.geoDump) {
            this->outputDebugGeo(mPointCloudGather[i], graph, pathGraph,
                                 lightPointCloud, mDiffusePointGather[i],
                                 pcSize, i);
        }
    }
}


/// @brief Dart throwing of points inside the density bounding box
/// @param pointCloud  array containing the postions of the point scattered in
///                    the density grid
/// @param minDist     minimum distance between each points while performing dart
///                    throwing
/// @param pcSize      size of the point cloud scattered in the density grid
/// @param iter        index of the current LightGraph iteration
void LightGraph::scatterPoints(vector<Vec3R> &pointCloud,
                   float &minDist,
                   uint32_t &pcSize,
                   const size_t iter) const {

    const tbb::tick_count start = tbb::tick_count::now();

    float newPos[3] = {0};
    const Vec3s bboxSize = mBbox.max() - mBbox.min();
    float minDistSquared = minDist*minDist;
    float lastValidminDistSquared = minDistSquared;
    const float cutoff = PC_DENSITY_CUTOFF;

    // adaptative minDist
    const uint32_t estimIter = PC_GROWTH_NUMBER_OF_ESTIMATE;
    const uint32_t batchSize = mScatterCount / estimIter;
    const uint32_t targetPcSize = TARGET_PC_SIZE;

    for (size_t n = 0; n < mScatterCount; ++n) {
        newPos[0] = mRng(iter)*bboxSize[0] + mBbox.min()[0];
        newPos[1] = mRng(iter)*bboxSize[1] + mBbox.min()[1];
        newPos[2] = mRng(iter)*bboxSize[2] + mBbox.min()[2];

        unique_ptr<openvdb::Vec3R> pt(new openvdb::Vec3R(newPos));

        assert(pcSize < PC_SIZE_HARD_LIMIT); // stop the program if pc is too large
        if(pcSize > targetPcSize) {
            if (mOpts.verbose) {
                cout << "Early stop of dart throwing at n = " << n << " / "
                     << mScatterCount << endl;
            }
            break;
        }

        if ((n+1) % batchSize == 0) {
            const uint32_t expectedSize = targetPcSize * pow(((double)n
                                        / (double)mScatterCount), PC_GROWTH_FUNC_POWER);
            const double ratio = pcSize / (double)expectedSize;
            minDistSquared *= ratio;
        }

        // will need to check if we are in density before running this loop
        bool collision = false;
        if (mFastSampler.wsSample(*pt) < cutoff) continue;
        for (size_t i = 0; i < pointCloud.size(); i++) {
            if ((*pt-pointCloud.at(i)).lengthSqr() < minDistSquared) {
                collision = true;
                break;
            }
        }

        if (!collision) {
            pointCloud.push_back(*pt);
            lastValidminDistSquared = minDistSquared;
            pcSize++;
        }

    }

    // update minDist with the adaptive estimate
    minDist = sqrt(lastValidminDistSquared);

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Point Cloud Creation: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Creation of the graph based on the scattered points
/// @param graph                       adjacency matrix of the LightGraph
/// @param pcSize                      size of the point cloud scattered in the
///                                    density grid
/// @param pointCloud                  array containing the postions of the point
///                                    scattered in the density grid
/// @param searchRadiusConnectSquared  maximum search radius squared to create
///                                    connection
/// @param connectCount                keeps count of how many connection are
///                                    created
void LightGraph::buildGraph(unique_ptr<bool[]> &graph,
                const uint32_t &pcSize,
                const vector<Vec3R> &pointCloud,
                const float &searchRadiusConnectSquared,
                uint32_t &connectCount) const {

    const tbb::tick_count start = tbb::tick_count::now();

    uint32_t connectCountArray[pcSize] = { };

    for (size_t i = 0; i < pcSize; i++) {
        for (size_t j = 0; j < pcSize; j++) {
            if (connectCountArray[i] >= mMaxConnect) break;
            if (connectCountArray[j] >= mMaxConnect) continue;
            if (i < j && (pointCloud.at(i)-pointCloud.at(j)).lengthSqr()
                         < searchRadiusConnectSquared) {
                graph[i * pcSize + j] = 1;
                graph[j * pcSize + i] = 1;
                connectCountArray[i]++;
                connectCountArray[j]++;
                connectCount++;
            }
        }
    }

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Graph Creation: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Utility function to print portions of the graph
/// @param graph       adjacency matrix of the LightGraph
/// @param graphWidth  the size of a single dimention of the graph
/// @param start       upper left corner of the sub-matrix to display
/// @param end         lower right corner of the sub-matrix to display
void LightGraph::printPartialGraph(const unique_ptr<float[]> &graph,
                                   uint32_t graphWidth,
                                   uint32_t start,
                                   uint32_t end) const
{
    for (size_t i = start; i < end; i++) {
        for (size_t j = start; j < end; j++) {
            cout << graph[i * graphWidth + j] << " ";
        }
        cout << endl;
    }
}


/// @brief Ray march each edges of the graph to compute transmitance
/// @param graph       adjacency matrix of the LightGraph
/// @param pathGraph   symmetric matrix holding the resulting maximal transmitance
///                    between each pair of vertex
/// @param pcSize      size of the point cloud scattered in the density grid
/// @param pointCloud  array containing the postions of the point scattered in
///                    the density grid
void LightGraph::raymarchConnection(const unique_ptr<bool[]> &graph,
                          vector<Vec3R> &pathGraph,
                          uint32_t pcSize,
                          const vector<Vec3R> &pointCloud) const
{
    const tbb::tick_count start = tbb::tick_count::now();

    using RayType = typename VolumeRayIntersector<openvdb::FloatGrid>::RayType;

    unique_ptr<VolumeRayIntersector<openvdb::FloatGrid> >
    mShadow(new VolumeRayIntersector<openvdb::FloatGrid>(*mGrid));

    for (size_t i = 0; i < pcSize; i++) {
        for (size_t j = 0; j < pcSize; j++) {
            if (graph[i * pcSize + j]) {
                if (i < j) {

                    const Vec3R startPos = pointCloud.at(i);
                    Vec3R dir = pointCloud.at(j)-startPos;
                    float dist = dir.length();
                    dir.normalize();

                    RayType sRay(startPos, dir);//Shadow ray
                    sRay.setMaxTime(dist);

                    Vec3R sTrans(1.0);

                    // move the shadow ray in word space and check if it hits the volume bbox
                    if( !mShadow->hitGridBbox(sRay) ) continue;

                    Real sT0, sT1;
                    while (mShadow->march(sT0, sT1)) {


                        bool sDropout = false;

                        Real sT = mOpts.step[1]*ceil(sT0/mOpts.step[1]);
                        for (; sT <= sT1; sT+= mOpts.step[1]) {
                            const Real sDensity = min(mFastSampler.wsSample(mShadow->getWorldPos(sT))
                                                      * mShader.getDensityScale()
                                                      * mShader.getShadowDensityScale()
                                                      * mShader.getScatterDensityScale(),
                                                      mShader.getDensityMax());
                            if (sDensity < mOpts.cutoff && mShader.getScatterDensityMin()
                                                           < mOpts.cutoff) continue;
                            // accumulate opacity by multiplication
                            sTrans *= math::Exp(mShader.getExtinction()
                                                * max(sDensity, mShader.getScatterDensityMin())
                                                * mOpts.step[1]);
                            if (sTrans.lengthSqr() < mOpts.cutoff) {
                                sTrans.setZero(); // set the shadow to full black if passed cutoff
                                sDropout=true;
                                break;
                            } //Terminate sRay
                        } //Integration over shadow segment
                        if (sDropout) break; //Terminate sRay
                    } // Shadow ray march

                    pathGraph[i * pcSize + j] = sTrans;
                    pathGraph[j * pcSize + i] = pathGraph[i * pcSize + j];
                }
            }
        }
    }

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Raymarch Graph Edges: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Compute direct lighting per graph vertex
/// @param pcSize           size of the point cloud scattered in the density grid
/// @param pointCloud       array containing the positions of the point scattered
///                         in the density grid
/// @param lightPointCloud  array containing the radiance of each vertex of the
///                         graph from direct lighting
void LightGraph::computeLighting(uint32_t pcSize,
                     const vector<Vec3R> &pointCloud,
                     unique_ptr<Vec3R[]> &lightPointCloud) const {

    const tbb::tick_count start = tbb::tick_count::now();

    using RayType = typename VolumeRayIntersector<openvdb::FloatGrid>::RayType;

    unique_ptr<VolumeRayIntersector<openvdb::FloatGrid> >
    mShadow(new VolumeRayIntersector<openvdb::FloatGrid>(*mGrid));

    for (size_t i = 0; i < pcSize; i++) {
        lightPointCloud[i] = Vec3R(0);

        const Vec3R startPos = pointCloud.at(i);

        const Real pTemp = mTempFastSampler.wsSample(startPos);
        Vec3R emission(mShader.getEmissionColor(pTemp) * mShader.getEmissionScale());

        for (size_t j = 0; j < mOpts.light.size(); j++) {
            Vec3R tmpLumi(0);

            uint64_t lightSamples = mOpts.light[j]->mSamples; // @suppress("Field cannot be resolved")
            if (lightSamples > 1) {
                lightSamples *= 10; // over sample env lights to ensure stability
            }

            for (size_t s=0; s<lightSamples; s++) { // @suppress("Field cannot be resolved")

                    Vec3R lightDir;
                    Vec3R lightIntensity;
                    double distance;
                    mOpts.light[j]->illuminate(startPos, lightDir, lightIntensity, distance); // @suppress("Invalid arguments")

                    RayType sRay(startPos, -lightDir, math::Delta<double>::value(), distance);//Shadow ray

                    Vec3R sTrans(1.0);

                    // move the shadow ray in word space and check if it hits the volume bbox
                    if( !mShadow->hitGridBbox(sRay) ) continue;

                    Real sT0, sT1;
                    while (mShadow->march(sT0, sT1)) {


                        bool sDropout = false;

                        Real sT = mOpts.step[1]*ceil(sT0/mOpts.step[1]);
                        for (; sT <= sT1; sT+= mOpts.step[1]) {
                            const Real sDensity = min(mFastSampler.wsSample(mShadow->getWorldPos(sT))
                                                      * mShader.getDensityScale()
                                                      * mShader.getShadowDensityScale()
                                                      * mShader.getScatterDensityScale(),
                                                      mShader.getDensityMax());
                            if (sDensity < mOpts.cutoff) continue;
                            // accumulate opacity by multiplication
                            sTrans *= math::Exp(mShader.getExtinction() * sDensity * mOpts.step[1]);
                            if (sTrans.lengthSqr() < mOpts.cutoff) {
                                sTrans.setZero(); // set the shadow to full black if passed cutoff
                                sDropout=true;
                                break;
                            } //Terminate sRay
                        } //Integration over shadow segment
                        if (sDropout) break; //Terminate sRay
                    } // Shadow ray march
                    tmpLumi += lightIntensity * sTrans;
            } // light sample loop
            lightPointCloud[i] += mShader.getAlbedo() * (tmpLumi/(double)lightSamples); // @suppress("Field cannot be resolved")
        } // light loop
        lightPointCloud[i] += emission;
    }

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Vertex Lighting Calculation: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Solve the graph with shortest path finding to figure out the
/// least occluded path between each pair of points in the graph
/// @param graph       adjacency matrix of the LightGraph
/// @param graphWidth  the size of a single dimension of the graph
void LightGraph::floydAlgo(vector<Vec3R> &graph, const uint32_t graphWidth) const {

    const tbb::tick_count start = tbb::tick_count::now();

    for (size_t k = 0; k < graphWidth; k++) {
        for (size_t i = 0; i < graphWidth; i++) {
            if (graph[i*graphWidth+k].lengthSqr() > 0) {
                for (size_t j = 0; j < graphWidth; j++) {
                    if (graph[k*graphWidth+j].lengthSqr() > 0) {
                        graph[i*graphWidth+j] = maxComponent(graph[i*graphWidth+j],
                                                             graph[i*graphWidth+k]
                                                             * graph[k*graphWidth+j]);
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < graphWidth; i++) {
        for (size_t j = 0; j < graphWidth; j++) {
            if (i == j) graph[i*graphWidth+j] = Vec3R(1.0);
        }
    }

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Shortest Path Finding: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Diffuse radiance in the graph according to the solution computed
/// with shortest path finding
/// @param pathGraph  symmetric matrix holding the resulting maximal transmitance
///                   between each pair of vertex
/// @param lightPC    array containing the radiance of each vertex of the graph
///                   from direct lighting
/// @param diffusePC  array containing the radiance of each vertex of the graph
///                   after diffusion
/// @param pcSize     size of the point cloud scattered in the density grid
void LightGraph::diffuseLight(vector<Vec3R> &pathGraph,
                  const unique_ptr<Vec3R[]> &lightPC,
                  vector<Vec3R> &diffusePC,
                  uint32_t pcSize) const {

    const tbb::tick_count start = tbb::tick_count::now();

    // make the radiance graph energy conservative
    for (size_t j = 0; j < pcSize; j++) {
        Vec3R accum = Vec3R(0);
        for (size_t i = 0; i < pcSize; i++) {
            accum += pathGraph[i * pcSize + j];
        }
        accum = maxComponent(accum, Vec3R(1));
        for (size_t i = 0; i < pcSize; i++) {
            pathGraph[i * pcSize + j] /= accum;
        }
    }

    diffusePC = vector<Vec3R>(pcSize, Vec3R(0));

    for (size_t i = 0; i < pcSize; i++) {
        for (size_t j = 0; j < pcSize; j++) {
            diffusePC[i] += lightPC[j] * pathGraph[i * pcSize + j];
        }
    }

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Radiance Diffusion: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Output the geometry generated by the LightGraph as ASCII files
/// for visualization or debug purpose
/// @param pointCloud         array containing the positions of the point
///                           scattered in the density grid
/// @param graph              adjacency matrix of the LightGraph
/// @param pathGraph          symmetric matrix holding the resulting maximal
///                           transmitance between each pair of vertex
/// @param lightPointCloud    array containing the radiance of each vertex of
///                           the graph from direct lighting
/// @param diffusePointCloud  array containing the radiance of each vertex of
///                           the graph after diffusion
/// @param pcSize             size of the point cloud scattered in the density
///                           grid
/// @param iter               index of the current LightGraph iteration
void LightGraph::outputDebugGeo(const vector<Vec3R> &pointCloud,
                    const unique_ptr<bool[]> &graph,
                    const vector<Vec3R> &pathGraph,
                    const unique_ptr<Vec3R[]> &lightPointCloud,
                    const vector<Vec3R> &diffusePointCloud,
                    uint32_t pcSize,
                    size_t iter) const {

    const tbb::tick_count start = tbb::tick_count::now();

    boost::filesystem::path vdbPath(mVdbFilename);
    const string vdbFileName = vdbPath.stem().string();
    string iterString = (boost::format("%04i") % iter).str();

    // create the geoDump dir if it does not exist
    boost::filesystem::path dir("geoDump");
    if(boost::filesystem::create_directory(dir)) {
        std::cout << "Success" << "\n";
    }

    boost::format pcFmt = boost::format("geoDump/%1%_%2%_%3%.txt")
                          % vdbFileName % "pc" % iterString;
    ofstream pcFile;
    pcFile.open(pcFmt.str());
    for (size_t i = 0; i < pcSize; i++) {
        pcFile << boost::format("%1%,%2%,%3%\n")
                  % pointCloud[i][0] % pointCloud[i][1] % pointCloud[i][2];
    }
    pcFile.close();

    boost::format edgeFmt = boost::format("geoDump/%1%_%2%_%3%.txt")
                            % vdbFileName % "edge" % iterString;
    ofstream edgeFile;
    edgeFile.open(edgeFmt.str());
    for (size_t i = 0; i < pcSize; i++) {
        for (size_t j = 0; j < pcSize; j++) {
            if (i < j && graph[i * pcSize + j]) {
                edgeFile << boost::format("%1%,%2%\n")
                            % i % j;
            }
        }
    }
    edgeFile.close();

    boost::format marchingFmt = boost::format("geoDump/%1%_%2%_%3%.txt")
                                % vdbFileName % "marching" % iterString;
    ofstream marchingFile;
    marchingFile.open(marchingFmt.str());
    size_t primNum = 0;
    for (size_t i = 0; i < pcSize; i++) {
        for (size_t j = 0; j < pcSize; j++) {
            if (i < j && graph[i * pcSize + j]) {
                marchingFile << boost::format("%1%,%2%,%3%\n")
                                % pathGraph[i * pcSize + j][0]
                                % pathGraph[i * pcSize + j][1]
                                % pathGraph[i * pcSize + j][2];
                primNum++;
            }
        }
    }
    marchingFile.close();

    boost::format lightFmt = boost::format("geoDump/%1%_%2%_%3%.txt")
                             % vdbFileName % "light" % iterString;
    ofstream lightFile;
    lightFile.open(lightFmt.str());
    for (size_t i = 0; i < pcSize; i++) {
        lightFile << boost::format("%1%,%2%,%3%\n")
                     % lightPointCloud[i][0]
                     % lightPointCloud[i][1]
                     % lightPointCloud[i][2];
    }
    lightFile.close();

    boost::format diffuseFmt = boost::format("geoDump/%1%_%2%_%3%.txt")
                               % vdbFileName % "diffuse" % iterString;
    ofstream diffuseFile;
    diffuseFile.open(diffuseFmt.str());
    for (size_t i = 0; i < pcSize; i++) {
        diffuseFile << boost::format("%1%,%2%,%3%\n")
                       % diffusePointCloud[i][0]
                       % diffusePointCloud[i][1]
                       % diffusePointCloud[i][2];
    }
    diffuseFile.close();

    if (mOpts.verbose) {
        const tbb::tick_count end = tbb::tick_count::now();
        std::cout << "Geo Dump to Disk: " << (end - start).seconds()
                  << " sec" << std::endl;
    }

}


/// @brief Create the vdb point grid for fast point cloud lookup when
/// rasterizing the LightGraphs to the scatter grid
/// @param positionsWrapper             position of the aggregated points from
///                                     the LightGraph's iterations
/// @param diffusePointCloudCombined    diffuse radiance of the aggregated points
///                                     from the LightGraph's iterations
/// @return returns a vdb point grid that can be used as an acceleration structure
openvdb::tools::PointIndexGrid::Ptr
LightGraph::buildVdbPointGrid(openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper,
                              vector<Vec3R> diffusePointCloudCombined) {

    // This method computes a voxel-size to match the number of
    // points / voxel requested. Although it won't be exact, it typically offers
    // a good balance of memory against performance.
    int pointsPerVoxel = 8;
    float voxelSize = openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel);

    // Create a transform using this voxel-size.
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(voxelSize);

    // Create a PointIndexGrid. This can be done automatically on creation of
    // the grid, however as this index grid is required for the position and
    // radius attributes, we create one we can use for both attribute creation.
    openvdb::tools::PointIndexGrid::Ptr pointIndexGrid =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(positionsWrapper, *transform);

    // Create a PointDataGrid containing these four points and using the point
    // index grid. This requires the positions wrapper.
    openvdb::points::PointDataGrid::Ptr ptGrid =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
            openvdb::points::PointDataGrid>(*pointIndexGrid, positionsWrapper, *transform);

    // Append a "scatter" attribute to the grid to hold the radius.
    // This attribute storage uses a unit range codec to reduce the memory
    // storage requirements down from 4-bytes to just 1-byte per value. This is
    // only possible because accuracy of the radius is not that important to us
    // and the values are always within unit range (0.0 => 1.0).
    // Note that this attribute type is not registered by default so needs to be
    // explicitly registered.
    using Codec = openvdb::points::FixedPointCodec</*1-byte=*/false, openvdb::points::UnitRange>;
    openvdb::points::TypedAttributeArray<Vec3R, Codec>::registerType();
    openvdb::NamePair scatterAttribute = openvdb::points::TypedAttributeArray<Vec3R, Codec>::attributeType();
    openvdb::points::appendAttribute(ptGrid->tree(), "scatter", scatterAttribute);

    // Create a wrapper around the radius vector.
    openvdb::points::PointAttributeVector<Vec3R> scatterWrapper(diffusePointCloudCombined);

    // Populate the "scatter" attribute on the points
    openvdb::points::populateAttribute<openvdb::points::PointDataTree,
        openvdb::tools::PointIndexTree, openvdb::points::PointAttributeVector<Vec3R>>(
                ptGrid->tree(), pointIndexGrid->tree(), "scatter", scatterWrapper);

    // Set the name of the grid
    ptGrid->setName("Points");

    return pointIndexGrid;
}


#endif /* SRC_LIGHTGRAPH_H_ */
