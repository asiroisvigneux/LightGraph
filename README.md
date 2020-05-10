# LightGraph
#### Efficient Multiple Scattering in Participating Media using Shortest Path Finding

<div align="center">
  <br>
  <a href="https://www.youtube.com/watch?v=BEbdy-7v5cE"><img src="https://img.youtube.com/vi/BEbdy-7v5cE/0.jpg" alt="IMAGE ALT TEXT"></a>
  <br><br>
  <a href="https://drive.google.com/file/d/1gPweA_EjzZ-LnGEMgiLaPmhjSYhiB6nC/view?usp=sharing">Technical Paper</a>
  | <a href="https://www.youtube.com/watch?v=BEbdy-7v5cE">YouTube Video</a>
  <br><br>
</div>

**LightGraph** is an efficient way of estimating multiple scattering in
discrete high resolution heterogeneous participating media. The approach is based on stochastically generated graphs that estimate how light propagates through the volume using shortest path finding. This new method tries to provide a way of achieving high quality photorealistic multiple scattering effect at a fraction of the computational cost of commonly used techniques in visual effects. The goal is not to be physically accurate nor it is to run in real-time, but to be a fast and reliable solution to allow quick turnarounds from a practical standpoint. The code is built on top of [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) using their raytracer implementation as a base.

## Main Features

* Flexible command line scene description
* Multiple light types supported (directional, point, environment)
* Environment light supports Exr maps with importance sampling
* Render OpenVDB files with multiple scattering within seconds
* Fully multithreaded with *Intel TBB*
* Outputs AOVs for single scattering, multiple scattering and emission
* Can output the LightGraph geometry as ASCII files
* Camera samples support (anti-aliasing)
* Shader with color ramp to remap temperature values as emission
* Spectrally varying scattering coefficient (wavelength dependency)

## Building the Project

**LightGraph** is built on top of [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb), which means that building the program will first require to build and install [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb). The instructions regarding this can be found in their [build documentation](https://www.openvdb.org/documentation/doxygen/build.html). It sould be straight forward to compile **LightGraph** after that.

## Rendering

The **scene** directory contains example scene descriptions that can run once the project is compiled. Those scenes uses the same settings as the images rendered in the [video](https://www.youtube.com/watch?v=BEbdy-7v5cE) above. The vdb for the cloud scene can be downloaded [here](https://www.technology.disneyanimation.com/clouds).

## Acknowledgments

* [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb) - Open Volume DataBase
* [PBRT](https://github.com/mmp/pbrt-v3/) - Physically Based Rendering
* [PVR](https://github.com/pvrbook/pvr) - Production Volume Rendering
