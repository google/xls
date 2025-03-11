// Copyright 2022 The XLS Authors
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

#include "xls/codegen/bdd_io_analysis.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {

namespace {

// Restrict BDD analysis to a subset of nodes.
//
// Currently limited to those cheap to analyze using BDDs plus
// compare ops.
bool UseNodeInBddEngine(const Node* node) {
  if (std::all_of(node->operands().begin(), node->operands().end(),
                  IsSingleBitType) &&
      IsSingleBitType(node)) {
    return true;
  }
  return (node->Is<NaryOp>() || node->Is<UnOp>() || node->Is<BitSlice>() ||
          node->Is<ExtendOp>() || node->Is<Concat>() ||
          node->Is<BitwiseReductionOp>() || node->Is<Literal>()) ||
         node->Is<CompareOp>();
}

}  // namespace

absl::StatusOr<bool> AreStreamingOutputsMutuallyExclusive(Proc* proc) {
  // Find all send nodes associated with streaming channels.
  int64_t streaming_send_count = 0;
  std::vector<Node*> send_predicates;

  for (Node* node : proc->nodes()) {
    if (!node->Is<Send>()) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(ChannelRef channel, node->As<Send>()->GetChannelRef());
    if (ChannelRefKind(channel) != ChannelKind::kStreaming) {
      continue;
    }

    Send* send = node->As<Send>();
    ++streaming_send_count;

    if (send->predicate().has_value()) {
      Node* predicate = send->predicate().value();
      send_predicates.push_back(predicate);
    }
  }

  // If there is only <=1 streaming send node, outputs are mutually exclusive
  if (streaming_send_count <= 1) {
    return true;
  }

  // If there > 1 streaming send node and not all have predicates, then
  // make an assumption that the streaming channels are not exclusive.
  // TODO(tedhong): 2022-02-12 - Refine this to perform a less
  // pessimistic assumption.
  if (streaming_send_count != send_predicates.size()) {
    return false;
  }

  // Use BDD query engine to determine predicates are such that
  // if one is true, the rest are false.
  BddQueryEngine query_engine(BddQueryEngine::kDefaultPathLimit,
                              UseNodeInBddEngine);
  XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());

  return query_engine.AtMostOneNodeTrue(send_predicates);
}

}  // namespace xls
