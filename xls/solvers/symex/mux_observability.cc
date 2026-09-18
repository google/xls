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

#include "xls/solvers/symex/mux_observability.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"

namespace xls::solvers::symex {
namespace {

// The multiplexer kinds that `Z3EncodingVisitor` encodes as opaque SSA
// variables. `one_hot_sel` is excluded because the encoder does not support it;
// treating it as a branch point here would make this analysis disagree with the
// encoding it reasons about.
bool IsMux(const Node* node) {
  return node->OpIn({Op::kSel, Op::kPrioritySel});
}

}  // namespace

absl::StatusOr<MuxObservability> MuxObservability::Create(Function* fn) {
  XLS_RET_CHECK_NE(fn, nullptr);
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_nodes, TopoSort(fn));

  MuxObservability observability;
  for (Node* node : topo_nodes) {
    if (IsMux(node)) {
      observability.muxes_.push_back(node);
    }
  }
  absl::c_reverse(observability.muxes_);

  // The return value is observable on every path.
  //
  // TODO: Side-effecting operations such as `assert` and `trace` make their
  // operands observable too, even when nothing flows to the return value.
  // Seeding them here is unnecessary for now because `Z3EncodingVisitor`
  // cannot translate those operations, so such functions are rejected before
  // this analysis runs.
  observability.MarkObservable(fn->return_value());

  // Root observability holds on every path. Discarding the trail makes these
  // markings permanent, so no later `Rollback` can retract them.
  observability.undo_trail_.clear();

  return observability;
}

int64_t MuxObservability::MarkArmChosen(Node* mux, Node* arm) {
  std::optional<GenericSelect> select = GenericSelect::TryFrom(mux);
  CHECK(select.has_value()) << "Not a multiplexer: " << mux->ToString();

  const int64_t token = TrailSize();
  // Only the selector and the arm actually taken become observable; the arms
  // that were not chosen stay don't cares under this decision.
  MarkObservable(select->selector());
  MarkObservable(arm);
  return token;
}

void MuxObservability::Rollback(int64_t token) {
  CHECK_GE(token, 0);
  CHECK_LE(token, TrailSize());
  while (TrailSize() > token) {
    observable_.erase(undo_trail_.back());
    undo_trail_.pop_back();
  }
}

void MuxObservability::MarkObservable(Node* node) {
  for (Node* mux : BoundaryMuxes(node)) {
    if (observable_.insert(mux).second) {
      undo_trail_.push_back(mux);
    }
  }
}

const std::vector<Node*>& MuxObservability::BoundaryMuxes(Node* node) {
  auto [it, inserted] = boundary_cache_.try_emplace(node);
  if (!inserted) {
    return it->second;
  }

  std::vector<Node*> boundary;
  if (IsMux(node)) {
    boundary.push_back(node);
    it->second = std::move(boundary);
    return it->second;
  }

  // Walk the fan-in cone of `node`, stopping at each multiplexer. Multiplexers
  // are opaque variables in the Z3 encoding, so nothing behind one is mentioned
  // by `node`'s expression.
  absl::flat_hash_set<Node*> visited;
  std::vector<Node*> worklist(node->operands().begin(), node->operands().end());
  while (!worklist.empty()) {
    Node* current = worklist.back();
    worklist.pop_back();
    if (!visited.insert(current).second) {
      continue;
    }
    if (IsMux(current)) {
      boundary.push_back(current);
      continue;
    }
    worklist.insert(worklist.end(), current->operands().begin(),
                    current->operands().end());
  }

  it->second = std::move(boundary);
  return it->second;
}

}  // namespace xls::solvers::symex
