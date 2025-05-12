// Copyright 2025 The XLS Authors
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

#include "xls/contrib/xlscc/node_manipulation.h"

#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"

namespace xlscc {

const xls::Node* RemoveIdentities(const xls::Node* node) {
  if (node == nullptr) {
    return node;
  }
  while (node->op() == xls::Op::kIdentity) {
    node = node->operand(xls::UnOp::kArgOperand);
  }
  return node;
}

bool NodesEquivalentWithContinuations(const xls::Node* a, const xls::Node* b) {
  return RemoveIdentities(a) == RemoveIdentities(b);
}

}  // namespace xlscc
