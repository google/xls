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

#ifndef XLS_TOOLS_NODE_COVERAGE_UTILS_H_
#define XLS_TOOLS_NODE_COVERAGE_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/observer.h"
#include "xls/tools/node_coverage_stats.pb.h"

namespace xls {

class CoverageEvalObserver final : public EvaluationObserver,
                                   public RuntimeObserver {
 public:
  explicit CoverageEvalObserver(std::optional<JitRuntime*> jit = std::nullopt)
      : jit_(jit) {}
  void NodeEvaluated(Node* n, const Value& v) override;
  std::optional<RuntimeObserver*> AsRawObserver() override {
    if (jit_) {
      return this;
    }
    return std::nullopt;
  }
  void RecordNodeValue(int64_t node_ptr, const uint8_t* data) override;

  // Prepare for proto conversion.
  absl::Status Finalize();

  absl::StatusOr<NodeCoverageStatsProto> proto() const;
  void SetPaused(bool v) { paused_ = v; }

 private:
  absl::flat_hash_map<Node*, LeafTypeTree<InlineBitmap>> coverage_;
  absl::flat_hash_map<Node*, std::vector<uint8_t>> raw_coverage_;
  std::optional<JitRuntime*> jit_;
  bool paused_ = false;
};

class ScopedRecordNodeCoverage {
 public:
  ScopedRecordNodeCoverage(std::optional<std::string> binproto,
                           std::optional<std::string> txtproto,

                           std::optional<JitRuntime*> jit = std::nullopt)
      : binproto_(std::move(binproto)),
        txtproto_(std::move(txtproto)),
        obs_(jit) {}
  ~ScopedRecordNodeCoverage();
  std::optional<EvaluationObserver*> observer() {
    if (binproto_ || txtproto_) {
      return &obs_;
    }
    return std::nullopt;
  }

  // Set to true to pause collection.
  void SetPaused(bool paused) { obs_.SetPaused(paused); }

 private:
  std::optional<std::string> binproto_;
  std::optional<std::string> txtproto_;
  CoverageEvalObserver obs_;
};

}  // namespace xls

#endif  // XLS_TOOLS_NODE_COVERAGE_UTILS_H_
