cmake_minimum_required (VERSION 3.3)

project (LightGraph)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

list(APPEND CMAKE_MODULE_PATH "/opt/openvdb/lib64/cmake/OpenVDB")

find_package(OpenVDB REQUIRED)
find_package(OpenEXR REQUIRED)

add_executable(main src/main.cc)

target_link_libraries(main -lboost_filesystem -lboost_system OpenVDB::openvdb OpenEXR::IlmImf)