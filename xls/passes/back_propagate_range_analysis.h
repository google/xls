// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_BACK_PROPAGATE_RANGE_ANALYSIS_H_
#define XLS_PASSES_BACK_PROPAGATE_RANGE_ANALYSIS_H_


#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/passes/range_query_engine.h"

namespace xls {

// Helper to analyze a function by back-propagating range information.
//
// Returns the data extracted from analyzing the given computation.
absl::StatusOr<absl::flat_hash_map<Node*, RangeData>> PropagateGivensBackwards(
    const RangeQueryEngine& engine, Node* node, RangeData given);

absl::StatusOr<absl::flat_hash_map<Node*, RangeData>> PropagateGivensBackwards(
    const RangeQueryEngine& engine, Node* node, const Bits& given);

}  // namespace xls

#endif  // XLS_PASSES_BACK_PROPAGATE_RANGE_ANALYSIS_H_
