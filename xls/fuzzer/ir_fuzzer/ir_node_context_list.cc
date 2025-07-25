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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Retrieves an element by its index, where the specific context list is
// specified by the ContextListType.
// TODO: Use a default value if the context list is empty.
BValue IrNodeContextList::GetElementAt(int64_t list_idx,
                                       ContextListType list_type) const {
  switch (list_type) {
    case ContextListType::COMBINED_LIST:
      if (combined_context_list_.empty()) {
        return DefaultValue(p_, fb_);
      }
      // Use of modulus to ensure the requested operand is within the bounds of
      // the context list.
      return combined_context_list_[list_idx % combined_context_list_.size()];
    case ContextListType::BITS_LIST:
      if (bits_context_list_.empty()) {
        return DefaultValue(p_, fb_, TypeCase::BITS_CASE);
      }
      return bits_context_list_[list_idx % bits_context_list_.size()];
    case ContextListType::TUPLE_LIST:
      return tuple_context_list_[list_idx % tuple_context_list_.size()];
    case ContextListType::ARRAY_LIST:
      return array_context_list_[list_idx % array_context_list_.size()];
  }
}

int64_t IrNodeContextList::GetListSize(ContextListType list_type) const {
  switch (list_type) {
    case ContextListType::COMBINED_LIST:
      return combined_context_list_.size();
    case ContextListType::BITS_LIST:
      return bits_context_list_.size();
    case ContextListType::TUPLE_LIST:
      return tuple_context_list_.size();
    case ContextListType::ARRAY_LIST:
      return array_context_list_.size();
  }
}

bool IrNodeContextList::IsEmpty(ContextListType list_type) const {
  switch (list_type) {
    case ContextListType::COMBINED_LIST:
      return combined_context_list_.empty();
    case ContextListType::BITS_LIST:
      return bits_context_list_.empty();
    case ContextListType::TUPLE_LIST:
      return tuple_context_list_.empty();
    case ContextListType::ARRAY_LIST:
      return array_context_list_.empty();
  }
}

// Adds the element to the combined list and the individual list for its type.
void IrNodeContextList::AppendElement(BValue element) {
  combined_context_list_.push_back(element);
  Type* type = element.GetType();
  if (type->IsBits()) {
    bits_context_list_.push_back(element);
  } else if (type->IsTuple()) {
    tuple_context_list_.push_back(element);
  } else if (type->IsArray()) {
    array_context_list_.push_back(element);
  }
}

}  // namespace xls
