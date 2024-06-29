// Copyright 2020 The XLS Authors
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
//
// Sample package definitions, generally used for testing aspects of the
// compiler.

#ifndef XLS_EXAMPLES_SAMPLE_PACKAGES_H_
#define XLS_EXAMPLES_SAMPLE_PACKAGES_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

namespace xls {
namespace sample_packages {

// Builds a right-rotate function inside of a package and returns the (owning)
// package and a pointer to the rrot function.
std::pair<std::unique_ptr<Package>, Function*> BuildRrot32();

// Builds a right-rotate function inside of a package and returns the (owning)
// package and a pointer to the rrot function. This version uses a constant
// value to specify how far to shift.
std::pair<std::unique_ptr<Package>, Function*> BuildRrot8Fixed();

// Builds an absolute-value function inside of a package and returns the
// (owning) package and a pointer to the abs function.
std::pair<std::unique_ptr<Package>, Function*> BuildAbs32();

// Builds a function that performs a simple concatenation with a 0b1 value.
//
//    fn concat_with_1(x: bits[31]) -> bits[32] {
//      concat(x, bits[1]:1)
//    }
std::pair<std::unique_ptr<Package>, Function*> BuildConcatWith1();

// Builds a function that sign extends the 2-bit operand to 32-bits.
//
//    fn concat_with_1(x: bits[2]) -> bits[32] {
//      sign_ext(x, bits[32])
//    }
std::pair<std::unique_ptr<Package>, Function*> BuildSignExtendTo32();

// As above, but for zero extension.
std::pair<std::unique_ptr<Package>, Function*> BuildZeroExtendTo32();

// Builds a function that accumulates the ivar with a given trip_count, as a
// sample function that contains a counted loop.
//
//    fn [TRIP_COUNT: u32] accum() -> u32 {
//      let result = for i, accum in range(TRIP_COUNT) {
//        accum + i
//      }(u32:0) in
//      result
//    }
std::pair<std::unique_ptr<Package>, Function*> BuildAccumulateIvar(
    int64_t trip_count, int64_t bit_count);

// Builds a function containing 2 independent counted loops.
// The results of the loops are added, to make sure they
// are not dead-code eliminated. In regards to scheduling
// the loops should run in parallel.
//
// Another goal is to prepare for control generation and shared FSMs.
std::pair<std::unique_ptr<Package>, Function*> BuildTwoLoops(
    bool same_trip_count, bool dependent_loops);

// Build up a simple function with a map().
std::pair<std::unique_ptr<Package>, Function*> BuildSimpleMap(
    int element_count);

// Returns a package built from the IR of one of the dslx_tests in the examples
// directory. 'name' should be the name of the dslx_test target (e.g.,
// "sha256"). The optional 'optimize' indicates whether to return the IR after
// optimizations.
absl::StatusOr<std::unique_ptr<Package>> GetBenchmark(std::string_view name,
                                                      bool optimized);

// Returns the names of all benchmarks.
absl::StatusOr<std::vector<std::string>> GetBenchmarkNames();

}  // namespace sample_packages
}  // namespace xls

#endif  // XLS_EXAMPLES_SAMPLE_PACKAGES_H_
