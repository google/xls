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

#include "xls/dev_tools/extract_segment.h"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/node_dependency_analysis.h"

namespace xls {

namespace {

absl::StatusOr<BValue> ExtractSegmentInto(FunctionBuilder& dest,
                                          FunctionBase* full,
                                          absl::Span<Node* const> source_nodes,
                                          absl::Span<Node* const> sink_nodes,
                                          bool next_nodes_are_tuples) {
  // Get node dependency information.
  std::optional<NodeDependencyAnalysis> forward_analysis;
  std::optional<NodeDependencyAnalysis> backward_analysis;
  std::vector<DependencyBitmap> forward_bitmaps;
  std::vector<DependencyBitmap> backward_bitmaps;
  if (!source_nodes.empty()) {
    forward_analysis =
        NodeDependencyAnalysis::ForwardDependents(full, source_nodes);
    for (auto n : source_nodes) {
      XLS_ASSIGN_OR_RETURN(auto dep, forward_analysis->GetDependents(n));
      forward_bitmaps.push_back(dep);
    }
  }
  if (!sink_nodes.empty()) {
    backward_analysis =
        NodeDependencyAnalysis::BackwardDependents(full, sink_nodes);
    for (auto n : sink_nodes) {
      XLS_ASSIGN_OR_RETURN(auto dep, backward_analysis->GetDependents(n));
      backward_bitmaps.push_back(dep);
    }
  }
  auto is_forward_dep = [&](Node* n) -> bool {
    return forward_bitmaps.empty() ||
           absl::c_any_of(forward_bitmaps, [&](DependencyBitmap db) {
             return *db.IsDependent(n);
           });
  };
  auto is_backward_dep = [&](Node* n) -> bool {
    return backward_bitmaps.empty() ||
           absl::c_any_of(backward_bitmaps, [&](DependencyBitmap db) {
             return *db.IsDependent(n);
           });
  };
  auto is_dep = [&](Node* n) -> bool {
    return is_forward_dep(n) && is_backward_dep(n);
  };
  auto is_used_by_dep = [&](Node* n) -> bool {
    const auto& users = n->users();
    return absl::c_any_of(users, [&](Node* u) { return is_dep(u); });
  };
  Package* dest_pkg = dest.function()->package();
  std::vector<Node*> outputs;
  absl::flat_hash_map<Node*, Node*> old_to_new;
  for (Node* n : TopoSort(full)) {
    if (!is_dep(n)) {
      if (is_used_by_dep(n)) {
        // Add a param/literal
        if (n->Is<Literal>()) {
          old_to_new[n] =
              dest.Literal(n->As<Literal>()->value(), n->loc(), n->GetName())
                  .node();
        } else {
          old_to_new[n] =
              dest.Param(
                      absl::StrFormat("param_for_%s_id%d", n->GetName(),
                                      n->id()),
                      dest_pkg->MapTypeFromOtherPackage(n->GetType()).value(),
                      n->loc())
                  .node();
        }
      }
      continue;
    }
    // Specific node types to handle specially.
    if (n->Is<Invoke>() || n->Is<Map>() || n->Is<CountedFor>() ||
        n->Is<DynamicCountedFor>()) {
      return absl::UnimplementedError("Subrountine calls not supported");
    }
    if (n->Is<Receive>()) {
      // Replace token with bits[1].
      std::vector<Type*> types;
      types.push_back(dest_pkg->GetBitsType(1));
      for (Type* other_type :
           n->GetType()->AsTupleOrDie()->element_types().subspan(1)) {
        XLS_ASSIGN_OR_RETURN(auto type,
                             dest_pkg->MapTypeFromOtherPackage(other_type));
        types.push_back(type);
      }
      old_to_new[n] = dest.Param(absl::StrFormat("param_wrapper_for_receive_%s",
                                                 n->GetName()),
                                 dest_pkg->GetTupleType(types), n->loc())
                          .node();
    } else if (n->Is<RegisterRead>()) {
      XLS_ASSIGN_OR_RETURN(auto type,
                           dest_pkg->MapTypeFromOtherPackage(n->GetType()));
      old_to_new[n] =
          dest.Param(absl::StrFormat("param_wrapper_for_reg_read_%s",
                                     n->GetName()),
                     type, n->loc())
              .node();
    } else if (n->Is<Send>()) {
      outputs.push_back(old_to_new.at(n->As<Send>()->data()));
      old_to_new[n] =
          dest.Literal(UBits(0, 1), n->loc(),
                       absl::StrFormat("marker_for_%s", n->GetName()))
              .node();
    } else if (n->Is<AfterAll>() || n->Is<Assert>() || n->Is<Cover>()) {
      // Do nothing.
      old_to_new[n] =
          dest.Literal(UBits(0, 1), n->loc(),
                       absl::StrFormat("marker_for_%s", n->GetName()))
              .node();
    } else if (n->Is<RegisterWrite>()) {
      outputs.push_back(old_to_new.at(n->As<RegisterWrite>()->data()));
      // Do nothing.
    } else if (n->Is<StateRead>()) {
      XLS_ASSIGN_OR_RETURN(auto type,
                           dest_pkg->MapTypeFromOtherPackage(n->GetType()));
      old_to_new[n] = dest.Param(n->As<StateRead>()->state_element()->name(),
                                 type, n->loc())
                          .node();
    } else if (n->Is<Next>() && next_nodes_are_tuples) {
      std::vector<BValue> new_ops;
      for (Node* op : n->operands()) {
        XLS_RET_CHECK(old_to_new.contains(op));
        new_ops.push_back(BValue(old_to_new.at(op), &dest));
      }
      old_to_new[n] = dest.Tuple(new_ops, n->loc(), n->GetName()).node();
      outputs.push_back(old_to_new[n]);
    } else {
      std::vector<Node*> new_ops;
      for (Node* op : n->operands()) {
        XLS_RET_CHECK(old_to_new.contains(op));
        new_ops.push_back(old_to_new.at(op));
      }
      XLS_ASSIGN_OR_RETURN(auto node,
                           n->CloneInNewFunction(new_ops, dest.function()));
      old_to_new[n] = node;
    }
  }
  std::vector<BValue> res;
  if (sink_nodes.empty()) {
    for (auto* n : outputs) {
      res.push_back(BValue(n, &dest));
    }
    if (full->IsFunction()) {
      res.push_back(
          BValue(old_to_new[full->AsFunctionOrDie()->return_value()], &dest));
    }
  } else {
    for (auto* n : sink_nodes) {
      res.push_back(BValue(old_to_new[n], &dest));
    }
  }
  if (res.size() == 1) {
    return res.front();
  }
  return dest.Tuple(res);
}

}  // namespace

absl::StatusOr<std::unique_ptr<Package>> ExtractSegmentInNewPackage(
    FunctionBase* full, absl::Span<Node* const> source_nodes,
    absl::Span<Node* const> sink_nodes, std::string_view extracted_package_name,
    std::string_view extracted_function_name, bool next_nodes_are_tuples) {
  std::unique_ptr<Package> pkg =
      std::make_unique<Package>(extracted_package_name);
  FunctionBuilder fb(extracted_function_name, pkg.get());
  XLS_ASSIGN_OR_RETURN(auto res,
                       ExtractSegmentInto(fb, full, source_nodes, sink_nodes,
                                          next_nodes_are_tuples));
  XLS_RETURN_IF_ERROR(fb.SetAsTop());
  XLS_RETURN_IF_ERROR(fb.BuildWithReturnValue(res).status());
  return pkg;
}

absl::StatusOr<Function*> ExtractSegment(
    FunctionBase* full, absl::Span<Node* const> source_nodes,
    absl::Span<Node* const> sink_nodes,
    std::string_view extracted_function_name, bool next_nodes_are_tuples) {
  FunctionBuilder fb(extracted_function_name, full->package());
  XLS_ASSIGN_OR_RETURN(auto res,
                       ExtractSegmentInto(fb, full, source_nodes, sink_nodes,
                                          next_nodes_are_tuples));
  return fb.BuildWithReturnValue(res);
}
}  // namespace xls
