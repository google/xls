// Copyright 2023 The XLS Authors
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

#ifndef XLS_FUZZER_CPP_SAMPLE_RUNNER_H_
#define XLS_FUZZER_CPP_SAMPLE_RUNNER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/dslx/interp_value.h"

namespace xls {

using ArgsBatch = std::vector<std::vector<dslx::InterpValue>>;

// Compares a set of results for equality.
//
// Each entry in the map is a sequence of Values generated from some source
// (e.g., interpreting the optimized IR). Each sequence of Values is compared
// for equality.
//
// Args:
//   results: Map of result values.
//   args_batch: Optional (may be null) batch of arguments used to produce the
//    given results. Batch should be the same length as the number of results
//    for any given value in "results".
absl::Status CompareResultsFunction(
    const absl::flat_hash_map<std::string, absl::Span<const dslx::InterpValue>>&
        results,
    const ArgsBatch* maybe_args_batch);

}  // namespace xls

#endif  // XLS_FUZZER_CPP_SAMPLE_RUNNER_H_
