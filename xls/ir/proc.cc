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
        "Cannot set next token to %s, expected token type but has type %s",
        next->GetName(), next->GetType()->ToString()));
  }
  next_token_ = next;
  return absl::OkStatus();
}

absl::Status Proc::SetNextState(Node* next) {
  if (next->GetType() != StateType()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot set next state to %s; type %s does not match "
        "proc state type %s",
        next->GetName(), next->GetType()->ToString(), StateType()->ToString()));
  }
  next_state_ = next;
  return absl::OkStatus();
}

}  // namespace xls
