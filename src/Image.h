// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file Image.h
///
/// @brief This file contains a utility function to save the rendered image to
/// disk as exr.


#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_


/// @brief Write the rendered image to disk as an exr file.
/// @param fname        file path of the rendered image
/// @param rgbFilm      film object holding the rgba values of the rendered image
/// @param directFilm   film object holding the single scattering values of the rendered image
/// @param scatterFilm  film object holding the multiple scattering values of the rendered image
/// @param emissionFilm film object holding the emission values of the rendered image
/// @param opts         render settings
void saveEXR(const std::string& fname,
             const Film& rgbFilm,
             const Film& directFilm,
             const Film& scatterFilm,
             const Film& emissionFilm,
             const SceneSettings& opts)
{
    using RGBA = Film::RGBA;

    std::string filename = fname;
    if (!boost::iends_with(filename, ".exr")) filename += ".exr";

    int threads = (opts.threads == 0 ? 8 : opts.threads);
    Imf::setGlobalThreadCount(threads);

    Imf::Header header(int(rgbFilm.width()), int(rgbFilm.height()));
    if (opts.compression == "none") {
        header.compression() = Imf::NO_COMPRESSION;
    } else if (opts.compression == "rle") {
        header.compression() = Imf::RLE_COMPRESSION;
    } else if (opts.compression == "zip") {
        header.compression() = Imf::ZIP_COMPRESSION;
    } else {
        OPENVDB_THROW(openvdb::ValueError,
            "expected none, rle or zip compression, got \"" << opts.compression << "\"");
    }
    header.channels().insert("R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("B", Imf::Channel(Imf::FLOAT));
    header.channels().insert("A", Imf::Channel(Imf::FLOAT));

    header.channels().insert("direct.R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("direct.G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("direct.B", Imf::Channel(Imf::FLOAT));

    header.channels().insert("scatter.R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("scatter.G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("scatter.B", Imf::Channel(Imf::FLOAT));

    header.channels().insert("emission.R", Imf::Channel(Imf::FLOAT));
    header.channels().insert("emission.G", Imf::Channel(Imf::FLOAT));
    header.channels().insert("emission.B", Imf::Channel(Imf::FLOAT));

    const uint64_t pixelBytes = sizeof(RGBA), rowBytes = pixelBytes * rgbFilm.width();
    RGBA& rgbPix = const_cast<RGBA*>(rgbFilm.pixels())[0];
    RGBA& dirPix = const_cast<RGBA*>(directFilm.pixels())[0];
    RGBA& scatPix = const_cast<RGBA*>(scatterFilm.pixels())[0];
    RGBA& emitPix = const_cast<RGBA*>(emissionFilm.pixels())[0];

    Imf::FrameBuffer framebuffer;
    framebuffer.insert("R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&rgbPix.r), pixelBytes, rowBytes));
    framebuffer.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&rgbPix.g), pixelBytes, rowBytes));
    framebuffer.insert("B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&rgbPix.b), pixelBytes, rowBytes));
    framebuffer.insert("A", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&rgbPix.a), pixelBytes, rowBytes));

    framebuffer.insert("direct.R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dirPix.r), pixelBytes, rowBytes));
    framebuffer.insert("direct.G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dirPix.g), pixelBytes, rowBytes));
    framebuffer.insert("direct.B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&dirPix.b), pixelBytes, rowBytes));

    framebuffer.insert("scatter.R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&scatPix.r), pixelBytes, rowBytes));
    framebuffer.insert("scatter.G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&scatPix.g), pixelBytes, rowBytes));
    framebuffer.insert("scatter.B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&scatPix.b), pixelBytes, rowBytes));

    framebuffer.insert("emission.R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&emitPix.r), pixelBytes, rowBytes));
    framebuffer.insert("emission.G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&emitPix.g), pixelBytes, rowBytes));
    framebuffer.insert("emission.B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char*>(&emitPix.b), pixelBytes, rowBytes));

    Imf::OutputFile imgFile(filename.c_str(), header);
    imgFile.setFrameBuffer(framebuffer);
    imgFile.writePixels(int(rgbFilm.height()));

}


#endif /* SRC_IMAGE_H_ */
