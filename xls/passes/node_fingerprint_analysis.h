// Copyright 2026 The XLS Authors
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

#ifndef XLS_PASSES_NODE_FINGERPRINT_ANALYSIS_H_
#define XLS_PASSES_NODE_FINGERPRINT_ANALYSIS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

// An analysis that computes a structural fingerprint for each node.
// This fingerprint is intended to be stable across optimization passes that
// rename nodes or change their IDs, as long as the underlying expression tree
// remains identical.
class NodeFingerprintAnalysis : public LazyNodeData<uint64_t> {
 public:
  NodeFingerprintAnalysis()
      : LazyNodeData<uint64_t>(DagCacheInvalidateDirection::kInvalidatesUsers) {
  }

  uint64_t GetFingerprint(Node* node) const { return *GetInfo(node); }

 protected:
  uint64_t ComputeInfo(
      Node* node,
      absl::Span<const uint64_t* const> operand_fingerprints) const override;

  absl::Status MergeWithGiven(uint64_t& info,
                              const uint64_t& given) const override {
    return absl::InternalError("Cannot merge fingerprints");
  }
};

}  // namespace xls

#endif  // XLS_PASSES_NODE_FINGERPRINT_ANALYSIS_H_
