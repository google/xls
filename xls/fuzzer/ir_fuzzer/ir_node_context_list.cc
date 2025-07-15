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
#include "xls/ir/function_builder.h"

namespace xls {

// Retrieves an element by its index, where the specific context list is
// specified by the ContextListType.
BValue IrNodeContextList::GetElementAt(int64_t list_idx,
                                       ContextListType list_type) {
  // Use of modulus to ensure the requested operand is within the bounds of the
  // context list.
  return context_list_[list_type][list_idx % context_list_[list_type].size()];
}

BValue IrNodeContextList::GetLastElement(ContextListType list_type) {
  CHECK(!context_list_[list_type].empty());
  return context_list_[list_type].back();
}

int64_t IrNodeContextList::GetListSize(ContextListType list_type) {
  return context_list_[list_type].size();
}

bool IrNodeContextList::IsEmpty(ContextListType list_type) {
  return context_list_[list_type].empty();
}

// Adds the element to the combined list and the individual list for its type.
void IrNodeContextList::AppendElement(BValue element) {
  context_list_[ContextListType::COMBINED_LIST].push_back(element);
  Type* type = element.GetType();
  if (type->IsBits()) {
    context_list_[ContextListType::BITS_LIST].push_back(element);
  } else if (type->IsTuple()) {
    context_list_[ContextListType::TUPLE_LIST].push_back(element);
  } else if (type->IsArray()) {
    context_list_[ContextListType::ARRAY_LIST].push_back(element);
  }
}

}  // namespace xls
