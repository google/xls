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

#include "xls/passes/node_fingerprint_analysis.h"

#include <cstdint>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"

namespace xls {

uint64_t NodeFingerprintAnalysis::ComputeInfo(
    Node* node, absl::Span<const uint64_t* const> operand_fingerprints) const {
  std::vector<uint64_t> operands;
  operands.reserve(operand_fingerprints.size());
  for (const uint64_t* f : operand_fingerprints) {
    operands.push_back(*f);
  }

  Op op = node->op();
  Type* type = node->GetType();

  if (node->Is<Literal>()) {
    return absl::HashOf(op, type, operands, node->As<Literal>()->value());
  }
  if (node->Is<BitSlice>()) {
    return absl::HashOf(op, type, operands, node->As<BitSlice>()->start());
  }
  if (node->Is<TupleIndex>()) {
    return absl::HashOf(op, type, operands, node->As<TupleIndex>()->index());
  }
  if (node->Is<ChannelNode>()) {
    return absl::HashOf(op, type, operands,
                        node->As<ChannelNode>()->channel_name());
  }
  if (node->Is<Param>()) {
    absl::StatusOr<int64_t> param_index =
        node->function_base()->GetParamIndex(node->As<Param>());
    CHECK_OK(param_index);
    return absl::HashOf(op, type, operands, *param_index);
  }

  return absl::HashOf(op, type, operands);
}

}  // namespace xls
