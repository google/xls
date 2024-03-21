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

#include "xls/passes/label_recovery_pass.h"

#include <optional>
#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

absl::StatusOr<bool> RecoverLabels(FunctionBase* f) {
  absl::flat_hash_map<std::string,
                      absl::InlinedVector<std::variant<Cover*, Assert*>, 1>>
      original_label_to_nodes;
  for (Node* node : f->nodes()) {
    switch (node->op()) {
      case Op::kCover: {
        Cover* c = node->As<Cover>();
        if (std::optional<std::string> original_label = c->original_label()) {
          original_label_to_nodes[original_label.value()].push_back(c);
        }
        break;
      }
      case Op::kAssert: {
        Assert* a = node->As<Assert>();
        if (std::optional<std::string> original_label = a->original_label()) {
          original_label_to_nodes[original_label.value()].push_back(a);
        }
        break;
      }
      default:
        break;
    }
  }

  bool changed_any = false;
  for (auto& [original_label_key, nodes] : original_label_to_nodes) {
    VLOG(10) << absl::StreamFormat("original label `%s` had %d nodes",
                                   original_label_key, nodes.size());
    // If there's only one node that had this original label, there were no
    // collisions due to inlining, and we can rename it back.
    //
    // TODO(leary): 2023-08-23 Even when there are collisions we can find the
    // minimal inlining symbol that will distinguish the labels to make them
    // shorter, but we don't do that yet thinking this may capture most of the
    // benefit.
    if (nodes.size() == 1) {
      changed_any = true;
      // Workaround lack of structured binding capture in C++17.
      const std::string& original_label = original_label_key;
      absl::visit(
          Visitor{
              [original_label](Cover* n) { n->set_label(original_label); },
              [original_label](Assert* n) { n->set_label(original_label); },
          },
          nodes.at(0));
    }
  }

  return changed_any;
}

}  // namespace

absl::StatusOr<bool> LabelRecoveryPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  return RecoverLabels(f);
}

REGISTER_OPT_PASS(LabelRecoveryPass);

}  // namespace xls
