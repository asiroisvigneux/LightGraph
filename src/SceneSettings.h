// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file SceneSettings.h
///
/// @brief This file holds a struct that contains the scene options used for
/// rendering.


#ifndef SRC_SCENESETTINGS_H_
#define SRC_SCENESETTINGS_H_


#include <string>

#include <openvdb/openvdb.h>

#include "Light.h"


struct SceneSettings {
    std::string camera;
    float aperture, focal, znear, zfar;
    openvdb::Vec3d camRotate;
    openvdb::Vec3d camTranslate;
    double camYRotate;
    uint64_t samples;
    std::vector<std::unique_ptr<Light>> light;
    double cutoff;
    openvdb::Vec2d step;
    uint64_t width, height;
    std::string compression;
    size_t iter;
    size_t scatterGridResFactor;
    size_t ptsPerKernel;
    int threads;
    bool verbose;
    bool geoDump;

    SceneSettings():
        camera("perspective"),
        aperture(36.0f),
        focal(50.0f),
        znear(1.0e-3f),
        zfar(std::numeric_limits<float>::max()),
        camRotate(0.0),
        camTranslate(0.0),
        camYRotate(0.0),
        samples(1),
        cutoff(0.005),
        step(1.0, 3.0),
        width(960),
        height(540),
        compression("zip"),
        iter(10),
        scatterGridResFactor(4),
        ptsPerKernel(3),
        threads(0),
        verbose(false),
        geoDump(false)
    {}

};

#endif /* SRC_SCENESETTINGS_H_ */
