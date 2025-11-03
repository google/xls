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

#ifndef XLS_PASSES_CRITICAL_PATH_SLACK_ANALYSIS_H_
#define XLS_PASSES_CRITICAL_PATH_SLACK_ANALYSIS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

class CriticalPathSlackAnalysis : public LazyNodeData<int64_t> {
 public:
  CriticalPathSlackAnalysis(
      const CriticalPathDelayAnalysis* critical_path_delay_analysis);

  int64_t SlackFromCriticalPath(Node* node) const;

  // It is necessary to recompute all slacks whenever nodes are modified
  // because the critical path may have changed.
  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override;
  void OperandRemoved(Node* node, Node* old_operand) override;
  void OperandAdded(Node* node) override;

 protected:
  int64_t ComputeInfo(
      Node* node, absl::Span<const int64_t* const> user_infos) const override;

  absl::Status MergeWithGiven(int64_t& info,
                              const int64_t& given) const override;

  // Propagate from users to operands
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }

 private:
  const CriticalPathDelayAnalysis* critical_path_delay_analysis_;
};

}  // namespace xls

#endif  // XLS_PASSES_CRITICAL_PATH_SLACK_ANALYSIS_H_
