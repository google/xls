// Copyright 2025 The XLS Authors
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

#ifndef XLS_PASSES_AREA_ACCUMULATED_ANALYSIS_H_
#define XLS_PASSES_AREA_ACCUMULATED_ANALYSIS_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

class AreaAccumulatedAnalysis : public LazyNodeData<double> {
 public:
  explicit AreaAccumulatedAnalysis(const AreaEstimator* area_estimator)
      : LazyNodeData<double>(DagCacheInvalidateDirection::kInvalidatesUsers),
        area_estimator_(area_estimator),
        area_through_to_node_() {
    CHECK(area_estimator_ != nullptr);
  }

  double GetAreaThroughToNode(Node* accumulated_to_node) const;

 protected:
  double ComputeInfo(
      Node* node, absl::Span<const double* const> operand_infos) const override;

  absl::Status MergeWithGiven(double& info, const double& given) const override;

 private:
  const AreaEstimator* area_estimator_;

  mutable absl::flat_hash_map<Node*, double> area_through_to_node_;

  void ForgetAccumulatedAreaDependingOn(Node* node);

 public:
  void NodeDeleted(Node* node) override;
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override;
  void OperandRemoved(Node* node, Node* old_operand) override;
  void OperandAdded(Node* node) override;
};

}  // namespace xls

#endif  // XLS_PASSES_AREA_ACCUMULATED_ANALYSIS_H_
