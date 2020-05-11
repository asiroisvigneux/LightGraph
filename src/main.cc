// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file main.cc
///
/// @brief LightGraph volume ray tracer for OpenVDB volumes. This file mainly
/// handles user inputs.


#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <assert.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/format.hpp>
#include <boost/format/group.hpp>

#include <openvdb/PlatformConfig.h>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfPixelType.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/tick_count.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/PointIndexGrid.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>

#include "Renderer.h"
#include "LightGraph.h"
#include "SceneSettings.h"
#include "Light.h"
#include "Shader.h"
#include "Sampling.h"
#include "Image.h"
#include "RandGen.h"
#include "ScatterGrid.h"


using namespace std;

// global constants
const float MIN_DIST_MULT = 0.13;
const uint32_t PC_SCATTER_COUNT = 1000000;
const uint32_t GRAPH_MAX_CONNECT = 12;

/// @brief Create a rotation matrix about the Y axis provided a rotation
/// amount in degrees
/// @param rotAmt   amount of rotation in degrees
/// @return         returns a 4x4 matrix containing a 3x3 rotation matrix
openvdb::math::Mat4d yRotXform(const double rotAmt) {
    return openvdb::math::rotation<openvdb::math::Mat4d>(openvdb::math::Y_AXIS, ((double)rotAmt/180.0) * M_PI);
}

/// @brief Handles the scene creation and triggers the actual pixel render as
/// well as the output to exr files
/// @param densityGrid  density grid to render
/// @param scatterGrid  scatter grid to render
/// @param tempGrid     temperature grid to render
/// @param imgFilename  output file path for the exr image file
/// @param opts         render options
/// @param shader       shader to apply to the volume being rendered
void render(const openvdb::FloatGrid& densityGrid,
            const openvdb::Vec3fGrid& scatterGrid,
            const openvdb::FloatGrid& tempGrid,
            const std::string& imgFilename,
            const SceneSettings& opts,
            const Shader &shader)
{
    using namespace openvdb;

    // setup the film to store the pixels with aovs
    Film rgbFilm(opts.width, opts.height);
    Film directFilm(opts.width, opts.height);
    Film scatterFilm(opts.width, opts.height);
    Film emissionFilm(opts.width, opts.height);

    // setup the camera
    std::unique_ptr<PerspectiveCamera> camera;
    if (boost::starts_with(opts.camera, "persp")) {
        openvdb::math::Mat4d xform = yRotXform(opts.camYRotate);
        camera.reset(new PerspectiveCamera(rgbFilm, directFilm, scatterFilm, emissionFilm,
                                           opts.camRotate+Vec3R(0,opts.camYRotate,0),
                                           opts.camTranslate*xform,
                                           opts.focal, opts.aperture, opts.znear, opts.zfar));
    } else {
        OPENVDB_THROW(ValueError,
            "expected perspective or orthographic camera, got \"" << opts.camera << "\"");
    }

    const tbb::tick_count start = tbb::tick_count::now();

    // define the intersector
    VolumeRayIntersector<openvdb::FloatGrid> intersector(densityGrid);
    VolumeRayIntersector<openvdb::Vec3fGrid> scatterIntersector(scatterGrid);
    VolumeRayIntersector<openvdb::FloatGrid> tempIntersector(tempGrid);

    // define the renderer (from the intersector and the camera)
    VolumeRender<VolumeRayIntersector<openvdb::FloatGrid>, // @suppress("Type cannot be resolved") // @suppress("Ambiguous problem")
                 VolumeRayIntersector<openvdb::Vec3fGrid> >
        renderer(intersector, *camera, scatterIntersector, tempIntersector, opts, shader);

    // START RENDERING!
    renderer.render(opts.threads != 1); // @suppress("Method cannot be resolved")

    // print rendertime to the console
    const tbb::tick_count end = tbb::tick_count::now();
    std::cout << "Render: " << (end - start).seconds() << " sec" << std::endl;

    if (boost::iends_with(imgFilename, ".exr")) {
        // Save as EXR (slow, but small file size).
        saveEXR(imgFilename, rgbFilm, directFilm, scatterFilm, emissionFilm, opts);
        cout << "Writting to Disk: " << imgFilename << "\n\n";
    } else {
        OPENVDB_THROW(ValueError, "unsupported image file format (" + imgFilename + ")");
    }
}

/// @brief Print minimalist help to see what commands are supported by the
/// program and what kind of input is expected
void printHelp() {
    cout << "-vdbIn                    : Input vdb file to read and render" << endl;
    cout << "                            format: string" << endl;
    cout << "                            ex: /path/to/volume.vdb" << endl;
    cout << "-exrOut                   : Output exr file to write to disk" << endl;
    cout << "                            format: string" << endl;
    cout << "                            ex: /path/to/image.exr" << endl;
    cout << "-camPos                   : Camera position" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-camRot                   : Camera rotation" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-camYRot                  : World centered camera Y rotation (turntable)" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-resWidth                 : Output image resolution in width" << endl;
    cout << "                            format: int" << endl;
    cout << "-resHeight                : Output image resolution in height" << endl;
    cout << "                            format: int" << endl;
    cout << "-focal                    : Camera focal length in mm (full frame sensor)" << endl;
    cout << "                            format: float" << endl;
    cout << "-samples                  : Number of antialising samples per pixel" << endl;
    cout << "                            format: int" << endl;
    cout << "-primaryStep              : Number of voxel width per primary sample in the volume" << endl;
    cout << "                            format: int" << endl;
    cout << "-shadowStep               : Number of voxel width per shadow sample in the volume" << endl;
    cout << "                            format: int" << endl;
    cout << "-cutoff                   : Density threshold, values lower are ignored" << endl;
    cout << "                            format: float" << endl;
    cout << "-light                    : Add a light to the scene" << endl;
    cout << "                            format-1: string:float:float,float,float:float:float,float,float" << endl;
    cout << "                            info: lightType:yRot:lightColor:lightIntensity:lightDirection" << endl;
    cout << "                            ex: dir:0:1,1,1:0.7:0.3,0.3,0.3" << endl;
    cout << "                            format-2: string:float:float,float,float:float:int,string" << endl;
    cout << "                            info: lightType:yRot:lightColor:lightIntensity:lightSamples,envMapPath" << endl;
    cout << "                            ex: env:0:1,1,1:1:10,/path/to/envMap.exr" << endl;
    cout << "-iter                     : Number of LightGraph iteration" << endl;
    cout << "                            format: int" << endl;
    cout << "-scatterGridResFactor     : Factor by which the scatter grid is scaled down" << endl;
    cout << "                            format: int" << endl;
    cout << "-ptsPerKernel             : Number of points used per kernel for a single iteration" << endl;
    cout << "                            format: int" << endl;
    cout << "-densityScale             : Density multiplier" << endl;
    cout << "                            format: float" << endl;
    cout << "-densityMax               : Make density values lower or equal to this value" << endl;
    cout << "                            format: float" << endl;
    cout << "-shadowDensityScale       : Density multiplier for the shadows only" << endl;
    cout << "                            format: float" << endl;
    cout << "-scatterDensityScale      : Density multiplier for the scatter only" << endl;
    cout << "                            format: float" << endl;
    cout << "-scatterDensityMin        : Make density values greater or equal to this value for scatter only" << endl;
    cout << "                            format: float" << endl;
    cout << "-absorption               : Shader absorption parameter" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-scattering               : Shader scattering parameter" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-volumeColor              : Shader volume color" << endl;
    cout << "                            format: float,float,float" << endl;
    cout << "-emissionScale            : Shader emission multiplier" << endl;
    cout << "                            format: float" << endl;
    cout << "-scatterScale             : Shader scattering multiplier" << endl;
    cout << "                            format: float" << endl;
    cout << "-scatterDensityMaskPower  : Power applied to the density field before masking the scatter field" << endl;
    cout << "                            format: float" << endl;
    cout << "-tempColorRamp            : Define a color ramp to remap temperature values to emission colors" << endl;
    cout << "                            format: float,float,...,float/float,float,float:float,float,float:...:float,float,float/float/float" << endl;
    cout << "                            info: posTicks/colorTicks/minTempRemap/maxTempRemap" << endl;
    cout << "                            ex: 0.0,0.33,0.66,1.0/0,0,0:1.0,0.076,0.0:1.0,0.322,0.0:1.0,0.45,0.15/0.35/1.5" << endl;
    cout << "-v                        : Enable verbose output" << endl;
    cout << "-geoDump                  : Write LightGraph geometry to disk as ASCII files" << endl;
    cout << "-h                        : Print this help" << endl;
    cout << "-help                     : Print this help" << endl;
    cout << "--help                    : Print this help" << endl;
}

/// @brief  Parse a user provided string to create a Vec3R object
/// @param  str string to parse for Vec3R extraction
/// @return returns a Vec3R object
Vec3d strToVec(string str) {
    vector<double> result;
    stringstream strStream(str);

    while(strStream.good()) {
       string substr;
       getline(strStream, substr, ',');
       result.push_back(stod(substr));
    }

    if (result.size()==1) {
        return Vec3R(result[0]);
    }

    return Vec3R(result[0], result[1], result[2]);
}

/// @brief  Parse a user provided string to create a light object
/// @param  str string to parse for light extraction
/// @return returns a light object
Light *parseLight(string str) {
    vector<string> colonSplit;
    stringstream strStream(str);

    while(strStream.good()) {
       string substr;
       getline(strStream, substr, ':');
       colonSplit.push_back(substr);
    }

    if (colonSplit[0] == "env") {
        double yRotAmt = stod(colonSplit[1]);
        Vec3d color = strToVec(colonSplit[2]);
        double intensity = stod(colonSplit[3]);
        int samples = stoi(colonSplit[4]);
        string exrFile = colonSplit[5];

        openvdb::math::Mat4d rotXform = yRotXform(yRotAmt);
        Light *envLight = new EnvLight(rotXform, color, intensity, samples, exrFile);

        return envLight;
    } else if (colonSplit[0] == "dir") {
        double yRotAmt = stod(colonSplit[1]);
        Vec3d color = strToVec(colonSplit[2]);
        double intensity = stod(colonSplit[3]);
        Vec3d rot = strToVec(colonSplit[4]);

        math::AffineMap orientXform;
        orientXform.accumPostRotation(math::X_AXIS, rot[0] * M_PI / 180.0);
        orientXform.accumPostRotation(math::Y_AXIS, rot[1] * M_PI / 180.0);
        orientXform.accumPostRotation(math::Z_AXIS, rot[2] * M_PI / 180.0);

        openvdb::math::Mat4d rotXform = orientXform.getMat4()*yRotXform(yRotAmt);
        Light *dirLight = new DistantLight(rotXform, color, intensity);

        return dirLight;
    } else {
        cout << colonSplit[0] << " is not a valid light type!" << endl;
        return nullptr;
    }
}

/// @brief  Parse a user provided string to remap temperature values into a
/// color ramp
/// @param  str string to parse for color ramp extraction
/// @return returns a color ramp object
ColorRamp* parseColorRamp(string str) {
    vector<string> slashSplit;
    stringstream strStream(str);

    while(strStream.good()) {
       string substr;
       getline(strStream, substr, '/');
       slashSplit.push_back(substr);
    }

    assert(slashSplit.size()==4 && "This is not a valid color ramp argument");

    const double inRemap = stod(slashSplit[2]);
    const double outRemap = stod(slashSplit[3]);

    // parse the pos
    vector<double> posVector;
    stringstream strStreamPos(slashSplit[0]);

    while(strStreamPos.good()) {
       string substr;
       getline(strStreamPos, substr, ',');
       posVector.push_back(stod(substr));
    }

    // parse the color
    vector<Vec3R> colorVector;
    stringstream strStreamColor(slashSplit[1]);

    while(strStreamColor.good()) {
       string substr;
       getline(strStreamColor, substr, ':');
       colorVector.push_back(strToVec(substr));
    }

    assert(posVector.size() == colorVector.size() && "Those two vectors should be the same size");

    ColorRamp* colorRamp = new ColorRamp(posVector, colorVector, inRemap, outRemap);
    return colorRamp;
}

/// @brief Helper struct to match user input with actual options of the program
struct ArgParse {
    int mArgc;
    char** mArgv;

    ArgParse(int argc, char* argv[]): mArgc(argc), mArgv(argv) {}

    /// @brief Check for a match between user provided string and actual options
    /// @param idx  index of the word in the user provided command chain
    /// @param name name of the potential option name
    /// @return     returns true if there is a match
    bool check(int idx, const std::string& name) const
    {
        if (mArgv[idx] == name) {
            return true;
        }
        return false;
    }
};


int
main(int argc, char * argv[]) {

    int retcode = EXIT_SUCCESS;

    SceneSettings opts;
    Shader shader;
    std::string vdbFilename, imgFilename, gridName;


    ////////////////////////////////////////////////////////////////////////////
    ///////////////                 PARSE INPUT                   //////////////
    ////////////////////////////////////////////////////////////////////////////


    ArgParse parser(argc, argv);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] == '-') {
            // I/O
            if (parser.check(i, "-vdbIn")) {
                ++i;
                vdbFilename = argv[i];
            } else if (parser.check(i, "-exrOut")) {
                ++i;
                imgFilename = argv[i];

            // scene
            } else if (parser.check(i, "-camPos")) {
                ++i;
                opts.camTranslate = strToVec(argv[i]);
            } else if (parser.check(i, "-camRot")) {
                ++i;
                opts.camRotate = strToVec(argv[i]);
            } else if (parser.check(i, "-camYRot")) {
                ++i;
                opts.camYRotate = stod(argv[i]);
            } else if (parser.check(i, "-resWidth")) {
                ++i;
                opts.width = uint64_t(atoi(argv[i]));
            } else if (parser.check(i, "-resHeight")) {
                ++i;
                opts.height = uint64_t(atoi(argv[i]));
            } else if (parser.check(i, "-focal")) {
                ++i;
                opts.focal = float(atof(argv[i]));
            } else if (parser.check(i, "-samples")) {
                ++i;
                opts.samples = (std::max(0, atoi(argv[i])));
            } else if (parser.check(i, "-primaryStep")) {
                ++i;
                opts.step[0] = atof(argv[i]);
            } else if (parser.check(i, "-shadowStep")) {
                ++i;
                opts.step[1] = atof(argv[i]);
            } else if (parser.check(i, "-cutoff")) {
                ++i;
                opts.cutoff = atof(argv[i]);
            } else if (parser.check(i, "-light")) {
                ++i;
                Light *lightPtr = parseLight(argv[i]);
                if (lightPtr != nullptr) {
                    opts.light.push_back(std::unique_ptr<Light>(lightPtr)); // @suppress("Symbol is not resolved")
                }

            // scattering
            } else if (parser.check(i, "-iter")) {
                ++i;
                opts.iter = size_t(atoi(argv[i]));
            } else if (parser.check(i, "-scatterGridResFactor")) {
                ++i;
                opts.scatterGridResFactor = size_t(atoi(argv[i]));
            } else if (parser.check(i, "-ptsPerKernel")) {
                ++i;
                opts.ptsPerKernel = size_t(atoi(argv[i]));

            // shader
            } else if (parser.check(i, "-densityScale")) {
                ++i;
                shader.setDensityScale(stod(argv[i]));
            } else if (parser.check(i, "-densityMax")) {
                ++i;
                shader.setDensityMax(stod(argv[i]));
            } else if (parser.check(i, "-shadowDensityScale")) {
                ++i;
                shader.setShadowDensityScale(stod(argv[i]));
            } else if (parser.check(i, "-scatterDensityScale")) {
                ++i;
                shader.setScatterDensityScale(stod(argv[i]));
            } else if (parser.check(i, "-scatterDensityMin")) {
                ++i;
                shader.setScatterDensityMin(stod(argv[i]));
            } else if (parser.check(i, "-absorption")) {
                ++i;
                shader.setAbsorption(strToVec(argv[i]));
            } else if (parser.check(i, "-scattering")) {
                ++i;
                shader.setScattering(strToVec(argv[i]));
            } else if (parser.check(i, "-volumeColor")) {
                ++i;
                shader.setVolumeColor(strToVec(argv[i]));
            } else if (parser.check(i, "-emissionScale")) {
                ++i;
                shader.setEmissionScale(stod(argv[i]));
            } else if (parser.check(i, "-scatterScale")) {
                ++i;
                shader.setScatterScale(stod(argv[i]));
            } else if (parser.check(i, "-scatterDensityMaskPower")) {
                ++i;
                shader.setScatterDensityMaskPower(stod(argv[i]));
            } else if (parser.check(i, "-tempColorRamp")) {
                ++i;
                shader.setEmissionColorRamp(parseColorRamp(argv[i]));

            // others
            } else if (arg == "-v") {
                opts.verbose = true;
            } else if (arg == "-geoDump") {
                opts.geoDump = true;
            } else if (arg == "-h" || arg == "-help" || arg == "--help") {
                printHelp();
                return 0;
            } else {
                cout << arg << " is not a valid option" << endl;
                return 1;
            }
        }
    }

    assert(!vdbFilename.empty() && "No vdb file provided");

    if (imgFilename.empty()) {
        imgFilename = vdbFilename.substr(0, vdbFilename.find_last_of('.')) + ".exr";
    }


    ////////////////////////////////////////////////////////////////////////////
    ///////////////               LOAD INPUT GRIDS                //////////////
    ////////////////////////////////////////////////////////////////////////////


    cout << "Reading in: " << vdbFilename << endl;

    openvdb::initialize();

    // density grid
    openvdb::io::File file(vdbFilename);
    file.open();
    const openvdb::FloatGrid::Ptr
    grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid("density"));
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> fastSampler(*grid);

    // temperature grid
    openvdb::FloatGrid::Ptr tempGrid = nullptr;
    if(file.hasGrid("temperature")) {
        tempGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid("temperature"));
    } else {
        tempGrid =  openvdb::gridPtrCast<openvdb::FloatGrid>(openvdb::FloatGrid::create(0.0));
    }
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> tempFastSampler(*tempGrid);


    ////////////////////////////////////////////////////////////////////////////
    ////////////           COMPUTE LIGHTGRAPH ITERATIONS             ///////////
    ////////////////////////////////////////////////////////////////////////////


    const tbb::tick_count start1 = tbb::tick_count::now();

    openvdb::math::BBox<Vec3R> bbox = grid->transform().indexToWorld(grid->evalActiveVoxelBoundingBox());

    const Vec3s bboxSize = bbox.max() - bbox.min();
    const float avgLength = (bboxSize[0]+bboxSize[1]+bboxSize[2]) / 3.0f;

    // make the minDist relative to the bbox of the vdb
    float minDist = MIN_DIST_MULT * avgLength;
    const uint32_t scatterCount = PC_SCATTER_COUNT;
    const uint32_t maxConnect = GRAPH_MAX_CONNECT;

    RandGen rng(opts.iter);

    vector<Vec3R> pointCloudCombined;
    vector<Vec3R> diffusePointCloudCombined;

    vector< vector<Vec3R> > pointCloudGather(opts.iter, vector<Vec3R>(0, Vec3s(0)));
    vector< vector<Vec3R> > diffusePointGather(opts.iter, vector<Vec3R>(0, Vec3R(0)));

    // split minDist to one instance per iter (thread safe)
    vector<float> minDistVec;
    for (size_t i=0; i<opts.iter; i++) minDistVec.push_back(minDist);

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, opts.iter), // @suppress("Invalid arguments")
                      LightGraph(minDistVec, scatterCount, bbox, fastSampler,
                                 tempFastSampler, rng, maxConnect, grid, opts,
                                 pointCloudGather, diffusePointGather, shader,
                                 vdbFilename));

    // flatten vectors after multithreading computation
    for (size_t i = 0; i<pointCloudGather.size(); i++) {

        pointCloudCombined.insert(end(pointCloudCombined),
                                  begin(pointCloudGather[i]),
                                  end(pointCloudGather[i]));

        diffusePointCloudCombined.insert(end(diffusePointCloudCombined),
                                         begin(diffusePointGather[i]),
                                         end(diffusePointGather[i]));
    }

    const tbb::tick_count end1 = tbb::tick_count::now();
    std::cout << "LightGraphs Calculation: " << (end1 - start1).seconds() << " sec" << std::endl;

    if (opts.verbose) {
        cout << "Point Cloud size: " << pointCloudCombined.size() << endl;
    }


    ////////////////////////////////////////////////////////////////////////////
    ////////////           RASTERIZE LIGHTGRAPH TO GRID              ///////////
    ////////////////////////////////////////////////////////////////////////////


    const tbb::tick_count start = tbb::tick_count::now();

    if (opts.verbose) {
        std::cout << "Input Density Grid Voxel size: " << grid->transform().voxelSize() << std::endl;
    }

    openvdb::Vec3fGrid::Ptr scatterGrid = openvdb::Vec3fGrid::create(Vec3f(0));
    ScatterGrid::buildScatterGrid(scatterGrid, grid, opts);

    openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(pointCloudCombined);

    openvdb::tools::PointIndexGrid::Ptr
    pointIndexGrid = LightGraph::buildVdbPointGrid(positionsWrapper,
                                                   diffusePointCloudCombined);

    // Apply the functor to all active values.
    openvdb::tools::foreach(scatterGrid->beginValueOn(), ScatterGrid(scatterGrid, // @suppress("Invalid arguments")
                                                                     pointIndexGrid,
                                                                     positionsWrapper,
                                                                     pointCloudCombined,
                                                                     diffusePointCloudCombined,
                                                                     opts.ptsPerKernel*opts.iter));

    const tbb::tick_count end = tbb::tick_count::now();
    std::cout << "Rasterize LightGraphs to Grid: " << (end - start).seconds() << " sec" << std::endl;


    ////////////////////////////////////////////////////////////////////////////
    ////////////                 START RENDER LOOP                   ///////////
    ////////////////////////////////////////////////////////////////////////////


    try {
        tbb::task_scheduler_init schedulerInit( tbb::task_scheduler_init::automatic );

        if (grid) {
            render(*grid, *scatterGrid, *tempGrid, imgFilename, opts, shader);
        }
    } catch (std::exception& e) {
        OPENVDB_LOG_FATAL(e.what());
        retcode = EXIT_FAILURE;
    } catch (...) {
        OPENVDB_LOG_FATAL("Exception caught (unexpected type)");
    }

    return retcode;
}
