// Copyright 2020 Google LLC
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
#include "xls/ir/node_iterator.h"

namespace xls {

absl::Status Proc::set_return_value(Node* n) {
  XLS_RET_CHECK_EQ(n->GetType(), ReturnType()) << absl::StreamFormat(
      "Cannot set proc return value to node %s. Proc state type is %s, but "
      "node has type %s",
      n->GetName(), ReturnType()->ToString(), n->GetType()->ToString());
  return Function::set_return_value(n);
}

std::string Proc::DumpIr(bool recursive) const {
  // TODO(meheff): Remove recursive argument. Recursively dumping multiple
  // functions should be a method at the Package level, not the function/proc
  // level.
  XLS_CHECK(!recursive);

  std::string res = absl::StrFormat(
      "proc %s(%s: %s, %s: %s, init=%s) {\n", name(), StateParam()->GetName(),
      StateParam()->GetType()->ToString(), TokenParam()->GetName(),
      TokenParam()->GetType()->ToString(), InitValue().ToHumanString());

  for (Node* node : TopoSort(const_cast<Proc*>(this))) {
    if (node->op() == Op::kParam) {
      // A param can never be the return value of a proc because of the type
      // restrictions and should never be printed (in functions parameters can
      // be return values and are printed in this case).
      XLS_CHECK(node != return_value())
          << "A parameter cannot be the return value of a proc: "
          << node->GetName();
      continue;
    }
    absl::StrAppend(&res, "  ", node == return_value() ? "ret " : "",
                    node->ToString(), "\n");
  }

  absl::StrAppend(&res, "}\n");
  return res;
}

}  // namespace xls
