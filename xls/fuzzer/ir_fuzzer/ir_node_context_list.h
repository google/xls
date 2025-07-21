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

#ifndef XLS_FUZZER_IR_FUZZER_IR_NODE_CONTEXT_LIST_H_
#define XLS_FUZZER_IR_FUZZER_IR_NODE_CONTEXT_LIST_H_

#include <cstdint>
#include <vector>

#include "xls/ir/function_builder.h"

namespace xls {

// Specifies the type of context list.
enum ContextListType {
  COMBINED_LIST = 0,
  BITS_LIST = 1,
};

// Maintains several lists of IR nodes to act as context for the use as operands
// by future IR nodes. There is one combined list and an individual list for
// each type to allow for easier retrieval of specific types of IR node
// operands.
class IrNodeContextList {
 public:
  BValue GetElementAt(int64_t list_idx,
                      ContextListType list_type = COMBINED_LIST) const;
  int64_t GetListSize(ContextListType list_type = COMBINED_LIST) const;
  bool IsEmpty(ContextListType list_type = COMBINED_LIST) const;

  void AppendElement(BValue element);

 private:
  // Vector of context lists, where the first is a combined list and the rest
  // are for individual types. Use of a ContextListType enum to specify which
  // list to use.
  std::vector<BValue> combined_context_list_;
  std::vector<BValue> bits_context_list_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_NODE_CONTEXT_LIST_H_
