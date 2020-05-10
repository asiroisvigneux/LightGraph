// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file RandGen.h
///
/// @brief This file defines a thread safe pseudo-random number generator that
/// will spawn a single instance of the generator per iterations so the results
/// stays deterministic.


#ifndef SRC_RANDGEN_H_
#define SRC_RANDGEN_H_


/// @brief Create a thread safe pseudo-random number generator
class RandGen {

public:
    typedef std::mt19937 Engine;

    /// @brief Create multiple instances of random number generator
    /// @param iter number of instance to spawn
    RandGen(const size_t &iter)
        : real_uni_dist_(0.0f, 1.0f)
        , engines()
        , mIter(iter)
        {
            for (size_t seed = 0; seed < mIter; ++seed) {
                engines.push_back(Engine(seed+854123));
            }
        }

    /// @brief Sample from the instances numbered "id"
    /// @param id the index of the generator to call
    /// @return   returns a random number between 0 and 1
    float operator()(const size_t &id) {
         return real_uni_dist_(engines[id]);
    }

    private:
        std::uniform_real_distribution<float> real_uni_dist_;
        std::vector<Engine> engines;
        const size_t mIter;
};

#endif /* SRC_RANDGEN_H_ */
