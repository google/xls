// Copyright 2021 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLS_NOC_SIMULATION_RANDOM_NUMBER_INTERFACE_H_
#define XLS_NOC_SIMULATION_RANDOM_NUMBER_INTERFACE_H_

#include <random>

// This file contains classes used manage and obtain random numbers
// from different distributions.

namespace xls::noc {

// Interface to generate random outcomes from a seeded random number generator.
class RandomNumberInterface {
 public:
  RandomNumberInterface() {
    // Default to a deterministic seed if nothing else is done.
    random_engine_.seed(0);
  }

  // Sets seed of random number generator.
  void SetSeed(int16_t seed) { random_engine_.seed(seed); }

  // Set a random seed to the random number generator.
  void SetRandomSeed() { SetSeed(std::random_device()()); }

  // These distributions have no internal state.

  // Returns True with probability p.
  bool BernoilliDistribution(double p) {
    return std::bernoulli_distribution(p)(random_engine_);
  }

  // Return number of trails until success.
  //  - sucess at each trail is with probability p
  //  - mean is 1/p
  int64_t GeometricDistribution(double p) {
    return std::geometric_distribution<int64_t>(p)(random_engine_);
  }

  // Returns next inter-arrival time.
  //   https://ieeexplore.ieee.org/document/9153030
  //
  //  - average arrival time is 1/lambda
  //  - probability of a burst is burst_prob
  int64_t GeneralizedGeometric(double lambda, double burst_prob) {
    bool burst = BernoilliDistribution(burst_prob);
    if (burst) {
      return 0;
    }

    double geo_p = lambda * (1.0 - burst_prob);
    return 1 + GeometricDistribution(geo_p);
  }

 private:
  // Random engine with a single int as state.
  std::minstd_rand random_engine_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_RANDOM_NUMBER_INTERFACE_H_
