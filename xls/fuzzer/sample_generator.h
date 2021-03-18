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

#ifndef XLS_FUZZER_SAMPLE_GENERATOR_H_
#define XLS_FUZZER_SAMPLE_GENERATOR_H_

#include <random>

#include "xls/dslx/concrete_type.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"

namespace xls {

// Holds RNG state and provides wrapper helper functions for useful
// distributions.
class RngState {
 public:
  explicit RngState(std::mt19937 rng) : rng_(std::move(rng)) {}

  // Returns a random double value in the range [0.0, 1.0).
  double RandomDouble();

  // Returns a random integer in the range [0, limit).
  int64_t RandRange(int64_t limit);

  // Returns a triangular distribution biased towards the zero side, with limit
  // as the exclusive limit.
  int64_t RandRangeBiasedTowardsZero(int64_t limit);

  std::mt19937& rng() { return rng_; }

 private:
  std::mt19937 rng_;
};

// Returns randomly generated values of the given types.
absl::StatusOr<std::vector<dslx::InterpValue>> GenerateArguments(
    absl::Span<const dslx::ConcreteType* const> arg_types, RngState* rng);

// Generates and returns a random Sample with the given options.
absl::StatusOr<Sample> GenerateSample(const dslx::AstGeneratorOptions& options,
                                      int64_t calls_per_sample,
                                      const SampleOptions& default_options,
                                      RngState* rng);

}  // namespace xls

#endif  // XLS_FUZZER_SAMPLE_GENERATOR_H_
