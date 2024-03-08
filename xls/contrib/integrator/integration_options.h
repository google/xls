// Copyright 2020 Google LLC
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
// A abstract class for an algorithm to integrate multiple functions.

#ifndef XLS_CONTRIB_INTEGRATOR_INTEGRATION_OPTIONS_H_
#define XLS_CONTRIB_INTEGRATOR_INTEGRATION_OPTIONS_H_

#include <iostream>

namespace xls {

class IntegrationOptions {
 public:
  // Used to specify different integration algorithms.
  enum class Algorithm {
    kBasicIntegrationAlgorithm,
  };

  // Which algorithm to use to merge functions.
  IntegrationOptions& algorithm(Algorithm value) {
    algorithm_ = value;
    return *this;
  }
  Algorithm algorithm() const { return algorithm_; }

  // Whether we can program individual muxes with unique select signals
  // or if we can configure the entire graph to match one of the input
  // functions using a single select signal.
  IntegrationOptions& unique_select_signal_per_mux(bool value) {
    unique_select_signal_per_mux_ = value;
    return *this;
  }
  bool unique_select_signal_per_mux() const {
    return unique_select_signal_per_mux_;
  }

 private:
  bool unique_select_signal_per_mux_ = false;
  Algorithm algorithm_ = Algorithm::kBasicIntegrationAlgorithm;
};

// Convert IntegrationOptions::Algorithm to human-readable text.
std::ostream& operator<<(std::ostream& os,
                         const IntegrationOptions::Algorithm& alg);

}  // namespace xls

#endif  // XLS_CONTRIB_INTEGRATOR_INTEGRATION_OPTIONS_H_
