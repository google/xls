// Copyright 2020 The XLS Authors
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

#include "xls/ir/function.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"

namespace xls {

absl::Status Function::set_return_value(Node* n) {
  XLS_RET_CHECK_EQ(n->function_base(), this) << absl::StreamFormat(
      "Return value node %s is not in this function %s (is in function %s)",
      n->GetName(), name(), n->function_base()->name());
  Node* old_return_value = return_value_;
  return_value_ = n;
  for (ChangeListener* listener : change_listeners_) {
    listener->ReturnValueChanged(this, old_return_value);
  }
  return absl::OkStatus();
}

FunctionType* Function::GetType() {
  std::vector<Type*> arg_types;
  for (Param* param : params()) {
    arg_types.push_back(param->GetType());
  }
  CHECK(return_value() != nullptr);
  return package_->GetFunctionType(arg_types, return_value()->GetType());
}
std::string Function::DumpIr() const {
  return DumpIrWithAnnotations(
      [](Node* n) -> std::optional<std::string> { return std::nullopt; });
}

std::string Function::DumpIrWithAnnotations(
    const std::function<std::optional<std::string>(Node*)>& annotate) const {
  std::string res;

  absl::StrAppend(&res, "fn " + name() + "(");
  absl::StrAppend(
      &res,
      absl::StrJoin(params_, ", ", [&](std::string* s, Param* const param) {
        std::optional<std::string> annotation = annotate(param);
        absl::StrAppendFormat(
            s, "%s: %s id=%d%s", param->name(), param->GetType()->ToString(),
            param->id(),
            annotation ? absl::StrCat(" (", *annotation, ")") : "");
      }));
  absl::StrAppend(&res, ") -> ");

  if (return_value() != nullptr) {
    absl::StrAppend(&res, return_value()->GetType()->ToString());
  }
  absl::StrAppend(&res, " {\n");

  for (Node* node : TopoSort(const_cast<Function*>(this))) {
    if (node->op() == Op::kParam && node == return_value()) {
      absl::StrAppendFormat(&res, "  ret %s\n", node->ToString());
      continue;
    }
    if (node->op() == Op::kParam) {
      continue;  // Already accounted for in the signature.
    }
    std::optional<std::string> annotation = annotate(node);
    absl::StrAppend(
        &res, "  ", node == return_value() ? "ret " : "", node->ToString(),
        annotation ? absl::StrCat(" (", *annotation, ")") : "", "\n");
  }

  absl::StrAppend(&res, "}\n");
  return res;
}

absl::StatusOr<Function*> Function::Clone(
    std::string_view new_name, Package* target_package,
    const absl::flat_hash_map<const Function*, Function*>& call_remapping)
    const {
  absl::flat_hash_map<Node*, Node*> original_to_clone;
  if (target_package == nullptr) {
    target_package = package();
  }
  Function* cloned_function = target_package->AddFunction(
      std::make_unique<Function>(new_name, target_package));
  cloned_function->SetForeignFunctionData(foreign_function_);

  // Clone parameters over first to maintain order.
  for (Param* param : (const_cast<Function*>(this))->params()) {
    XLS_ASSIGN_OR_RETURN(original_to_clone[param],
                         param->CloneInNewFunction({}, cloned_function));
  }
  for (Node* node : TopoSort(const_cast<Function*>(this))) {
    if (node->Is<Param>()) {  // Params were already copied.
      continue;
    }
    std::vector<Node*> cloned_operands;
    for (Node* operand : node->operands()) {
      cloned_operands.push_back(original_to_clone.at(operand));
    }

    switch (node->op()) {
      // Remap CountedFor body.
      case Op::kCountedFor: {
        CountedFor* src = node->As<CountedFor>();
        Function* body = call_remapping.contains(src->body())
                             ? call_remapping.at(src->body())
                             : src->body();
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_function->MakeNodeWithName<CountedFor>(
                src->loc(), cloned_operands[0],
                absl::Span<Node*>(cloned_operands).subspan(1),
                src->trip_count(), src->stride(), body, src->GetName()));
        break;
      }
      // Remap Map to_apply.
      case Op::kMap: {
        Map* src = node->As<Map>();
        Function* to_apply = call_remapping.contains(src->to_apply())
                                 ? call_remapping.at(src->to_apply())
                                 : src->to_apply();
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_function->MakeNodeWithName<Map>(
                src->loc(), cloned_operands[0], to_apply, src->GetName()));
        break;
      }
      // Remap Invoke to_apply.
      case Op::kInvoke: {
        Invoke* src = node->As<Invoke>();
        Function* to_apply = call_remapping.contains(src->to_apply())
                                 ? call_remapping.at(src->to_apply())
                                 : src->to_apply();
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            cloned_function->MakeNodeWithName<Invoke>(
                src->loc(), cloned_operands, to_apply, src->GetName()));
        break;
      }
      // Default clone.
      default: {
        XLS_ASSIGN_OR_RETURN(
            original_to_clone[node],
            node->CloneInNewFunction(cloned_operands, cloned_function));
        break;
      }
    }
  }
  XLS_RETURN_IF_ERROR(
      cloned_function->set_return_value(original_to_clone.at(return_value())));
  return cloned_function;
}

// Helper function for IsDefinitelyEqualTo. Recursively compares 'node' and
// 'other_node' and their operands using Node::IsDefinitelyEqualTo.
// 'matched_pairs' is used to memoize the result of the comparison.
static bool IsEqualRecurse(
    const Node* node, const Node* other_node,
    absl::flat_hash_map<const Node*, const Node*>* matched_pairs) {
  auto it = matched_pairs->find(node);
  if (it != matched_pairs->end()) {
    return it->second == other_node;
  }

  if (!node->IsDefinitelyEqualTo(other_node)) {
    VLOG(2) << absl::StrFormat("Function %s != %s: node %s != %s",
                               node->function_base()->name(),
                               other_node->function_base()->name(),
                               node->GetName(), other_node->GetName());
    return false;
  }

  for (int64_t i = 0; i < node->operand_count(); ++i) {
    if (!IsEqualRecurse(node->operand(i), other_node->operand(i),
                        matched_pairs)) {
      return false;
    }
  }
  (*matched_pairs)[node] = other_node;
  return true;
}

bool Function::IsDefinitelyEqualTo(const Function* other) const {
  if (this == other) {
    VLOG(2) << absl::StrFormat("Function %s == %s: same pointer", name(),
                               other->name());
    return true;
  }

  // Must have the types of parameters in the same order.
  if (params().size() != other->params().size()) {
    VLOG(2) << absl::StrFormat(
        "Function %s != %s: different number of parameters (%d vs %d)", name(),
        other->name(), params().size(), other->params().size());
    return false;
  }

  absl::flat_hash_map<const Node*, const Node*> matched_pairs;
  for (int64_t i = 0; i < params().size(); ++i) {
    // All we care about is the type (not the name) of the parameter so don't
    // use Param::IsDefinitelyEqualTo.
    if (!param(i)->GetType()->IsEqualTo(other->param(i)->GetType())) {
      VLOG(2) << absl::StrFormat(
          "Function %s != %s: type of parameter %d not the same (%s vs %s)",
          name(), other->name(), i, param(i)->GetType()->ToString(),
          other->param(i)->GetType()->ToString());
      return false;
    }
    matched_pairs[param(i)] = other->param(i);
  }

  bool result =
      IsEqualRecurse(return_value(), other->return_value(), &matched_pairs);
  if (result) {
    VLOG(2) << absl::StrFormat("Function %s is equal to %s", name(),
                               other->name());
  }
  return result;
}
absl::Status Function::InternalRebuildSideTables() {
  // We can't actually rebuild much of anything since the order of the params is
  // only held in the side table. We can still check for correctness at least.
  // TODO(allight): We should ideally be able to do this.
  XLS_RET_CHECK(next_values_.empty());
  XLS_RET_CHECK(next_values_by_state_read_.empty());

  for (Param* p : params_) {
    XLS_RET_CHECK(p->function_base() == this)
        << "param from different function: " << p;
  }
  return absl::OkStatus();
}

}  // namespace xls
