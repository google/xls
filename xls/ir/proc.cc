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

#include "xls/ir/proc.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"

namespace xls {

std::string Proc::DumpIr(bool recursive) const {
  // TODO(meheff): Remove recursive argument. Recursively dumping multiple
  // functions should be a method at the Package level, not the function/proc
  // level.
  XLS_CHECK(!recursive);

  std::string res = absl::StrFormat(
      "proc %s(%s: %s, %s: %s, init=%s) {\n", name(), TokenParam()->GetName(),
      TokenParam()->GetType()->ToString(), StateParam()->GetName(),
      StateParam()->GetType()->ToString(), InitValue().ToHumanString());

  for (Node* node : TopoSort(const_cast<Proc*>(this))) {
    if (node->op() == Op::kParam) {
      continue;
    }
    absl::StrAppend(&res, "  ", node->ToString(), "\n");
  }
  absl::StrAppend(&res, "  next (", NextToken()->GetName(), ", ",
                  NextState()->GetName(), ")\n");

  absl::StrAppend(&res, "}\n");
  return res;
}

absl::Status Proc::SetNextToken(Node* next) {
  if (!next->GetType()->IsToken()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot set next token to \"%s\", expected token type but has type %s",
        next->GetName(), next->GetType()->ToString()));
  }
  next_token_ = next;
  return absl::OkStatus();
}

absl::Status Proc::SetNextState(Node* next) {
  if (next->GetType() != StateType()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot set next state to \"%s\"; type %s does not match "
        "proc state type %s",
        next->GetName(), next->GetType()->ToString(), StateType()->ToString()));
  }
  next_state_ = next;
  return absl::OkStatus();
}

absl::Status Proc::ReplaceState(absl::string_view state_param_name,
                                Node* next_state) {
  Param* old_state_param = StateParam();
  if (!old_state_param->users().empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Existing state param \"%s\" still has uses",
                        old_state_param->GetName()));
  }

  // Add a new state param node.
  XLS_ASSIGN_OR_RETURN(
      state_param_,
      MakeNodeWithName<Param>(/*loc=*/absl::nullopt, state_param_name,
                              next_state->GetType()));
  next_state_ = next_state;

  XLS_RET_CHECK(!HasImplicitUse(old_state_param));
  XLS_RETURN_IF_ERROR(RemoveNode(old_state_param, /*remove_param_ok=*/true));
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Proc::Port>> Proc::GetPorts() const {
  std::vector<Port> ports;
  for (Node* node : nodes()) {
    // Port channels should only have send/receive nodes, not
    // send_if/receive_if.
    if (node->Is<Receive>() || node->Is<Send>()) {
      int64_t channel_id = node->Is<Send>() ? node->As<Send>()->channel_id()
                                            : node->As<Receive>()->channel_id();
      XLS_ASSIGN_OR_RETURN(Channel * channel,
                           package()->GetChannel(channel_id));
      if (channel->IsPort()) {
        ports.push_back(Port{down_cast<PortChannel*>(channel),
                             node->Is<Receive>() ? PortDirection::kInput
                                                 : PortDirection::kOutput,
                             node});
      }
    }
  }
  std::sort(ports.begin(), ports.end(), [](const Port& a, const Port& b) {
    // The IR verifier verifies that either no or all ports have positions.
    if (a.channel->GetPosition().has_value() &&
        b.channel->GetPosition().has_value()) {
      return a.channel->GetPosition() < b.channel->GetPosition();
    }
    return a.channel->id() < b.channel->id();
  });
  return std::move(ports);
}

}  // namespace xls
