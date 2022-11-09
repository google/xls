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

#include "xls/passes/token_dependency_pass.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/passes/token_provenance_analysis.h"

namespace xls {

absl::StatusOr<bool> TokenDependencyPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  using NodeRelation = absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>;

  auto relation_to_string = [](const NodeRelation& relation) {
    std::vector<std::string> lines;
    for (const auto& [n, s] : relation) {
      lines.push_back(absl::StrFormat("  %s : {%s}", n->GetName(),
                                      absl::StrJoin(s, ", ", NodeFormatter)));
    }
    return absl::StrJoin(lines, "\n");
  };

  // This is a relation over the set of side-effecting nodes (and AfterAlls) in
  // the program. If a token consumed by node A was produced by node B (ignoring
  // intervening tuples or identity functions), then there will be an edge from
  // B to A in this relation.
  NodeRelation token_deps;
  {
    XLS_ASSIGN_OR_RETURN(TokenProvenance provenance,
                         TokenProvenanceAnalysis(f));
    for (const auto& [node, ignore] : provenance) {
      for (Node* child : node->operands()) {
        if (!provenance.contains(child)) {
          continue;
        }
        for (Node* element : provenance.at(child).elements()) {
          if (element != nullptr) {
            token_deps[element].insert(node);
          }
        }
      }
    }
  }
  XLS_VLOG(3) << "Token deps:";
  XLS_VLOG_LINES(3, relation_to_string(token_deps));

  // This is a relation over the set of nodes in the program. If a value was
  // consumed by node A was produced by node B, then there will be an edge from
  // B to A in this relation.
  NodeRelation data_deps;
  {
    for (Node* node : f->nodes()) {
      for (Node* child : node->operands()) {
        data_deps[child].insert(node);
      }
    }
  }

  XLS_VLOG(3) << "Data deps:";
  XLS_VLOG_LINES(3, relation_to_string(data_deps));

  // The transitive closure of the token dependency relation.
  NodeRelation token_deps_closure = TransitiveClosure<Node*>(token_deps);
  XLS_VLOG(3) << "Token deps closure:";
  XLS_VLOG_LINES(3, relation_to_string(token_deps_closure));

  auto dfs = [&](Node* root,
                 const NodeRelation& dag) -> absl::flat_hash_set<Node*> {
    std::vector<Node*> stack;
    stack.push_back(root);
    absl::flat_hash_set<Node*> discovered;
    while (!stack.empty()) {
      Node* popped = stack.back();
      stack.pop_back();
      if (!discovered.contains(popped)) {
        discovered.insert(popped);
        if (dag.contains(popped)) {
          for (Node* child : dag.at(popped)) {
            stack.push_back(child);
          }
        }
      }
    }
    return discovered;
  };

  // A relation mapping each effectful node to the set of receives that it is
  // data-dependent on, but not token-dependent on.
  NodeRelation io_to_receive;

  auto is_side_effecting_token_op = [](Node* n) {
    return OpIsSideEffecting(n->op()) && TypeHasToken(n->GetType());
  };
  for (Node* a : f->nodes()) {
    if (!is_side_effecting_token_op(a)) {
      continue;
    }
    if (a->GetType()->IsToken()) {
      continue;
    }
    if (a->op() != Op::kReceive) {
      return absl::InternalError(
          "Can't handle token-and-data producing ops other than receive yet");
    }
    for (Node* b : dfs(a, data_deps)) {
      if (a == b) {
        continue;
      }
      if (!is_side_effecting_token_op(b)) {
        continue;
      }

      if (token_deps_closure.at(a).contains(b)) {
        continue;
      }

      io_to_receive[b].insert(a);
    }
  }

  XLS_VLOG(3) << "IO to receive:";
  XLS_VLOG_LINES(3, relation_to_string(io_to_receive));

  // A relation similar to `io_to_receive`, except that only the earliest
  // effectful nodes are included. For example, if `io_to_receive` contains
  // three keys `A`, `B`, and `C`, and `C` is transitively token-dependent
  // on `B`, then `minimal_io_to_receive` will only contain `A` and `B`.
  NodeRelation minimal_io_to_receive = io_to_receive;
  for (const auto& [io, receive] : io_to_receive) {
    for (Node* downstream_of_io : token_deps_closure.at(io)) {
      if (downstream_of_io != io) {
        minimal_io_to_receive.erase(downstream_of_io);
      }
    }
  }
  XLS_VLOG(3) << "Minimal IO to receive:";
  XLS_VLOG_LINES(3, relation_to_string(minimal_io_to_receive));

  bool changed = false;

  // Before touching the IR create a determistic sort of the keys of the
  // relation.
  std::vector<Node*> minimal_io_to_receive_keys;
  minimal_io_to_receive_keys.reserve(minimal_io_to_receive.size());
  for (const auto& [io, _] : minimal_io_to_receive) {
    minimal_io_to_receive_keys.push_back(io);
  }
  SortByNodeId(&minimal_io_to_receive_keys);

  for (Node* io : minimal_io_to_receive_keys) {
    for (Node* receive : SetToSortedVector(minimal_io_to_receive.at(io))) {
      for (Node* input : io->operands()) {
        // We're making the assumption that any token-typed input to an
        // effectful operation must be a proper token input.
        if (input->GetType()->IsToken()) {
          XLS_ASSIGN_OR_RETURN(
              Node * receive_token,
              f->MakeNode<TupleIndex>(SourceInfo(), receive, 0));
          XLS_ASSIGN_OR_RETURN(
              Node * new_token,
              f->MakeNode<AfterAll>(SourceInfo(),
                                    std::vector<Node*>{receive_token, input}));
          changed |= io->ReplaceOperand(input, new_token);
        }
      }
    }
  }

  return changed;
}

}  // namespace xls
