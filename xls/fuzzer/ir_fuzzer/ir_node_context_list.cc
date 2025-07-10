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

#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"

#include <cstdint>

#include "absl/log/check.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/function_builder.h"

namespace xls {

BValue IrNodeContextList::GetElementAt(int64_t list_idx,
                                       TypeCase type_case) const {
  return context_list_[type_case][list_idx % context_list_[type_case].size()];
}

BValue IrNodeContextList::GetLastElement(TypeCase type_case) const {
  CHECK(!context_list_[type_case].empty());
  return context_list_[type_case].back();
}

int64_t IrNodeContextList::GetListSize(TypeCase type_case) const {
  return context_list_[type_case].size();
}

bool IrNodeContextList::IsEmpty(TypeCase type_case) const {
  return context_list_[type_case].empty();
}

void IrNodeContextList::AppendElement(BValue element) {
  context_list_[TypeCase::UNSET].push_back(element);
  Type* type = element.GetType();
  if (type->IsBits()) {
    context_list_[TypeCase::BITS].push_back(element);
  } else if (type->IsTuple()) {
    context_list_[TypeCase::TUPLE].push_back(element);
  } else if (type->IsArray()) {
    context_list_[TypeCase::ARRAY].push_back(element);
  }
}

}  // namespace xls
