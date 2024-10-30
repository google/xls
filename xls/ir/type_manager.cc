// Copyright 2024 The XLS Authors
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

#include "xls/ir/type_manager.h"

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

TypeManager::TypeManager() {
  token_type_ = std::make_unique<TokenType>();
  owned_types_.insert(token_type_.get());
}
BitsType* TypeManager::GetBitsType(int64_t bit_count) {
  if (bit_count_to_type_.find(bit_count) != bit_count_to_type_.end()) {
    return &bit_count_to_type_.at(bit_count);
  }
  auto it = bit_count_to_type_.emplace(bit_count, BitsType(bit_count));
  BitsType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

ArrayType* TypeManager::GetArrayType(int64_t size, Type* element_type) {
  ArrayKey key{size, element_type};
  if (array_types_.find(key) != array_types_.end()) {
    return &array_types_.at(key);
  }
  CHECK(IsOwnedType(element_type))
      << "Type is not owned by package: " << *element_type;
  auto it = array_types_.emplace(key, ArrayType(size, element_type));
  ArrayType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

TupleType* TypeManager::GetTupleType(absl::Span<Type* const> element_types) {
  TypeVec key(element_types.begin(), element_types.end());
  if (tuple_types_.find(key) != tuple_types_.end()) {
    return &tuple_types_.at(key);
  }
  for (const Type* element_type : element_types) {
    CHECK(IsOwnedType(element_type))
        << "Type is not owned by package: " << *element_type;
  }
  auto it = tuple_types_.emplace(key, TupleType(element_types));
  TupleType* new_type = &(it.first->second);
  owned_types_.insert(new_type);
  return new_type;
}

TokenType* TypeManager::GetTokenType() { return token_type_.get(); }

absl::StatusOr<Type*> TypeManager::MapTypeFromOtherArena(
    Type* other_arena_type) {
  // TypeManager already owns this type.
  if (IsOwnedType(other_arena_type)) {
    return other_arena_type;
  }

  if (other_arena_type->IsBits()) {
    const BitsType* bits = other_arena_type->AsBitsOrDie();
    return GetBitsType(bits->bit_count());
  }
  if (other_arena_type->IsArray()) {
    const ArrayType* array = other_arena_type->AsArrayOrDie();
    XLS_ASSIGN_OR_RETURN(Type * elem_type,
                         MapTypeFromOtherArena(array->element_type()));
    return GetArrayType(array->size(), elem_type);
  }
  if (other_arena_type->IsTuple()) {
    const TupleType* tuple = other_arena_type->AsTupleOrDie();
    std::vector<Type*> member_types;
    member_types.reserve(tuple->size());
    for (auto* elem_type : tuple->element_types()) {
      XLS_ASSIGN_OR_RETURN(Type * new_elem_type,
                           MapTypeFromOtherArena(elem_type));
      member_types.push_back(new_elem_type);
    }
    return GetTupleType(member_types);
  }
  if (other_arena_type->IsToken()) {
    return GetTokenType();
  }
  return absl::InternalError("Unsupported type.");
}

FunctionType* TypeManager::GetFunctionType(absl::Span<Type* const> args_types,
                                           Type* return_type) {
  std::string key = FunctionType(args_types, return_type).ToString();
  if (function_types_.find(key) != function_types_.end()) {
    return &function_types_.at(key);
  }
  for (Type* t : args_types) {
    CHECK(IsOwnedType(t)) << "Parameter type is not owned by package: "
                          << t->ToString();
  }
  auto it = function_types_.emplace(key, FunctionType(args_types, return_type));
  FunctionType* new_type = &(it.first->second);
  owned_function_types_.insert(new_type);
  return new_type;
}

absl::StatusOr<Type*> TypeManager::GetTypeFromProto(const TypeProto& proto) {
  if (!proto.has_type_enum()) {
    return absl::InvalidArgumentError("Missing type_enum field in TypeProto.");
  }
  if (proto.type_enum() == TypeProto::BITS) {
    if (!proto.has_bit_count() || proto.bit_count() < 0) {
      return absl::InvalidArgumentError(
          "Missing or invalid bit_count field in TypeProto.");
    }
    return GetBitsType(proto.bit_count());
  }
  if (proto.type_enum() == TypeProto::TUPLE) {
    std::vector<Type*> elements;
    for (const TypeProto& element_proto : proto.tuple_elements()) {
      XLS_ASSIGN_OR_RETURN(Type * element, GetTypeFromProto(element_proto));
      elements.push_back(element);
    }
    return GetTupleType(elements);
  }
  if (proto.type_enum() == TypeProto::ARRAY) {
    if (!proto.has_array_size() || proto.array_size() < 0) {
      return absl::InvalidArgumentError(
          "Missing or invalid array_size field in TypeProto.");
    }
    if (!proto.has_array_element()) {
      return absl::InvalidArgumentError(
          "Missing array_element field in TypeProto.");
    }
    XLS_ASSIGN_OR_RETURN(Type * element_type,
                         GetTypeFromProto(proto.array_element()));
    return GetArrayType(proto.array_size(), element_type);
  }
  if (proto.type_enum() == TypeProto::TOKEN) {
    return GetTokenType();
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid type_enum value in TypeProto: %d", proto.type_enum()));
}

absl::StatusOr<FunctionType*> TypeManager::GetFunctionTypeFromProto(
    const FunctionTypeProto& proto) {
  std::vector<Type*> param_types;
  for (const TypeProto& param_proto : proto.parameters()) {
    XLS_ASSIGN_OR_RETURN(Type * param_type, GetTypeFromProto(param_proto));
    param_types.push_back(param_type);
  }
  if (!proto.has_return_type()) {
    return absl::InvalidArgumentError(
        "Missing return_type field in FunctionTypeProto.");
  }
  XLS_ASSIGN_OR_RETURN(Type * return_type,
                       GetTypeFromProto(proto.return_type()));
  return GetFunctionType(param_types, return_type);
}

Type* TypeManager::GetTypeForValue(const Value& value) {
  switch (value.kind()) {
    case ValueKind::kBits:
      return GetBitsType(value.bits().bit_count());
    case ValueKind::kTuple: {
      std::vector<Type*> element_types;
      for (const Value& element_value : value.elements()) {
        element_types.push_back(GetTypeForValue(element_value));
      }
      return GetTupleType(element_types);
    }
    case ValueKind::kArray: {
      // No element type can be inferred for 0-element arrays.
      // TODO(google/xls#917): Remove this check when empty arrays are
      // supported.
      CHECK(!value.empty());
      return GetArrayType(value.size(), GetTypeForValue(value.elements()[0]));
    }
    case ValueKind::kToken:
      return GetTokenType();
    case ValueKind::kInvalid:
      break;
  }
  LOG(FATAL) << "Invalid value for type extraction.";
}

static_assert(std::is_move_constructible_v<TypeManager>);
static_assert(std::is_move_assignable_v<TypeManager>);

}  // namespace xls
