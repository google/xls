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

#ifndef XLS_IR_TYPE_MANAGER_H_
#define XLS_IR_TYPE_MANAGER_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class TypeManager {
 public:
  explicit TypeManager();
  TypeManager(TypeManager&&) = delete;
  TypeManager& operator=(TypeManager&&) = delete;
  TypeManager(const TypeManager&) = delete;
  TypeManager& operator=(const TypeManager&) = delete;
  // Returns whether the given type is one of the types owned by this package.
  bool IsOwnedType(const Type* type) const {
    return owned_types_.find(type) != owned_types_.end();
  }
  bool IsOwnedFunctionType(const FunctionType* function_type) const {
    return owned_function_types_.find(function_type) !=
           owned_function_types_.end();
  }

  BitsType* GetBitsType(int64_t bit_count);
  ArrayType* GetArrayType(int64_t size, Type* element_type);
  TupleType* GetTupleType(absl::Span<Type* const> element_types);
  TokenType* GetTokenType();
  FunctionType* GetFunctionType(absl::Span<Type* const> args_types,
                                Type* return_type);

  // Returns a pointer to a type owned by this arena that is of the same
  // type as 'other_arena_type', which may be owned by another arena.
  absl::StatusOr<Type*> MapTypeFromOtherArena(Type* other_arena_type);
  // Creates and returned an owned type constructed from the given proto.
  absl::StatusOr<Type*> GetTypeFromProto(const TypeProto& proto);
  absl::StatusOr<FunctionType*> GetFunctionTypeFromProto(
      const FunctionTypeProto& proto);

  Type* GetTypeForValue(const Value& value);

 private:
  // Set of owned types in this package.
  absl::flat_hash_set<const Type*> owned_types_;

  // Set of owned function types in this package.
  absl::flat_hash_set<const FunctionType*> owned_function_types_;

  // Mapping from bit count to the owned "bits" type with that many bits. Use
  // node_hash_map for pointer stability.
  absl::node_hash_map<int64_t, BitsType> bit_count_to_type_;

  // Mapping from the size and element type of an array type to the owned
  // ArrayType. Use node_hash_map for pointer stability.
  using ArrayKey = std::pair<int64_t, const Type*>;
  absl::node_hash_map<ArrayKey, ArrayType> array_types_;

  // Mapping from elements to the owned tuple type.
  //
  // Uses node_hash_map for pointer stability.
  using TypeVec = absl::InlinedVector<const Type*, 4>;
  absl::node_hash_map<TypeVec, TupleType> tuple_types_;

  // Owned token type.
  TokenType token_type_;

  // Mapping from Type:ToString to the owned function type. Use
  // node_hash_map for pointer stability.
  absl::node_hash_map<std::string, FunctionType> function_types_;
};

}  // namespace xls

#endif  // XLS_IR_TYPE_MANAGER_H_
