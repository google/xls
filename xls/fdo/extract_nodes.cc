// Copyright 2023 The XLS Authors
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

#include "xls/fdo/extract_nodes.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"

namespace xls {

absl::StatusOr<std::unique_ptr<Package>> ExtractNodes(
    const absl::flat_hash_set<Node*>& nodes, std::string_view top_module_name,
    bool return_all_liveouts) {
  XLS_RET_CHECK(!nodes.empty());
  FunctionBase* f = (*nodes.begin())->function_base();
  XLS_RET_CHECK(std::all_of(nodes.begin(), nodes.end(), [&](Node* node) {
    return node->function_base() == f;
  }));

  std::vector<Node*> topo_sorted_nodes;
  absl::flat_hash_set<Node*> filtered_nodes;
  topo_sorted_nodes.reserve(nodes.size());

  for (Node* node : TopoSort(f)) {
    if (nodes.contains(node)) {
      Type* ntype = node->GetType();
      if (ntype->GetFlatBitCount() > 0) {
        // Unexpected case (it seems tuples w/ tokens have been dissolved).
        // It probably *would* work correctly, but if this is encountered,
        // verify that handling is correct (that data paths are timed correctly)
        XLS_RET_CHECK(!TypeHasToken(ntype))
            << "Unexpected node with a type that contains a token"
            << node->ToString();

        topo_sorted_nodes.emplace_back(node);
        filtered_nodes.insert(node);
      } else {
        // Currently support FDO only post-opt;
        // only tokens should have zero bitwidth
        XLS_RET_CHECK(ntype->IsToken())
            << "Not expecting non-token type with zero bits"
            << node->ToString();
        // skip (don't include token-producing nodes)
      }
    }
  }

  // Here, we create a temporary package for holding the temporary function. The
  // rationale is, in many cases, we want to run this method concurrently for
  // multiple different set of nodes. As a result, working on the same package
  // will cause data racing that will crash the compiler very soon.
  auto tmp_package = std::make_unique<xls::Package>(top_module_name);
  auto tmp_f = std::make_unique<Function>(top_module_name, tmp_package.get());

  absl::flat_hash_map<Node*, Node*> node_map;
  std::vector<Node*> live_out;
  for (Node* node : topo_sorted_nodes) {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      if (node_map.contains(operand)) {
        new_operands.push_back(node_map.at(operand));
      } else {
        // As the temporary package doesn't own any types from the original
        // package, we need this one more step to map the original type to a new
        // one owned by our temporary package.
        XLS_ASSIGN_OR_RETURN(
            Type * operand_type,
            tmp_package->MapTypeFromOtherPackage(operand->GetType()));
        Node* new_param = tmp_f->AddNode(std::make_unique<Param>(
            operand->loc(), operand_type, operand->GetName(), tmp_f.get()));
        node_map[operand] = new_param;
        new_operands.push_back(new_param);
      }
    }
    // Hack to support viewing procs as functions.
    Node* new_node;
    if (node->Is<Send>() || node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(
          Type * node_type,
          tmp_package->MapTypeFromOtherPackage(node->GetType()));
      new_node = tmp_f->AddNode(std::make_unique<xls::Param>(
          node->loc(), node_type, node->GetName(), tmp_f.get()));
    } else {
      XLS_ASSIGN_OR_RETURN(new_node,
                           node->CloneInNewFunction(new_operands, tmp_f.get()));
    }
    // Collect the live-out of the set of nodes.
    node_map[node] = new_node;
    if (tmp_f->HasImplicitUse(node)) {
      live_out.push_back(new_node);
    } else if (return_all_liveouts) {
      if (std::any_of(node->users().begin(), node->users().end(), [&](Node* u) {
            return !filtered_nodes.contains(u) || u->Is<Send>();
          })) {
        live_out.push_back(new_node);
      }
    } else {
      // Return live-outs that only have external users if return_all_liveouts
      // is not set.
      if (std::all_of(node->users().begin(), node->users().end(), [&](Node* u) {
            return !filtered_nodes.contains(u) || u->Is<Send>();
          })) {
        live_out.push_back(new_node);
      }
    }
  }

  // If the set of nodes don't include the function output, create a final tuple
  // which gathers all live-outs. The tuple will be the return value of the new
  // function. Otherwise, just use the mapped function output.
  auto src_function = static_cast<Function*>(f);
  if (node_map.contains(src_function->return_value())) {
    XLS_RETURN_IF_ERROR(
        tmp_f->set_return_value(node_map[src_function->return_value()]));
  } else {
    if (live_out.size() == 1) {
      XLS_RETURN_IF_ERROR(tmp_f->set_return_value(live_out.front()));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * return_tuple,
                           tmp_f->MakeNode<Tuple>(SourceInfo(), live_out));
      XLS_RETURN_IF_ERROR(tmp_f->set_return_value(return_tuple));
    }
  }
  tmp_package->AddFunction(std::move(tmp_f));
  return std::move(tmp_package);
}

}  // namespace xls
