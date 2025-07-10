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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/function_builder.h"

namespace xls {

class IrNodeContextList {
 public:
  IrNodeContextList() {
    // There are 4 separate context lists. The first list contains all of the IR
    // nodes acting as a mono list. The last 3 lists represent IR nodes of
    // specific types.
    context_list_.resize(4);
  }

  BValue GetElementAt(int64_t list_idx, TypeCase type_case = UNSET) const;
  BValue GetLastElement(TypeCase type_case = UNSET) const;
  int64_t GetListSize(TypeCase type_case = UNSET) const;
  bool IsEmpty(TypeCase type_case = UNSET) const;

  void AppendElement(BValue element);

 private:
  std::vector<std::vector<BValue>> context_list_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_NODE_CONTEXT_LIST_H_
