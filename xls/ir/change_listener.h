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

#ifndef XLS_IR_CHANGE_LISTENER_H_
#define XLS_IR_CHANGE_LISTENER_H_

#include <cstdint>

#include "absl/types/span.h"

namespace xls {

class Function;
class Node;
class Proc;

// A pure-virtual interface for listening to changes to XLS IR.
class ChangeListener {
 public:
  virtual ~ChangeListener() = default;

  virtual void NodeAdded(Node* node) {}
  virtual void NodeDeleted(Node* node) {}

  virtual void OperandChanged(Node* node, Node* old_operand,
                              absl::Span<const int64_t> operand_nos) {}
  inline void OperandChanged(Node* node, Node* old_operand,
                             int64_t operand_no) {
    OperandChanged(node, old_operand, absl::MakeConstSpan({operand_no}));
  }

  // Called when an instance of `old_operand` was removed from `node`'s operand
  // list, and no other instances remain.
  virtual void OperandRemoved(Node* node, Node* old_operand) {}

  // Called when a new operand is added to a node (currently necessarily the
  // last operand in the list).
  virtual void OperandAdded(Node* node) {}

  virtual void ReturnValueChanged(Function* function_base,
                                  Node* old_return_value) {}
  virtual void NextStateElementChanged(Proc* proc, int64_t state_index,
                                       Node* old_next_state_element) {}
};

}  // namespace xls

#endif  // XLS_IR_CHANGE_LISTENER_H_
