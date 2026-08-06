// Copyright 2020 The XLS Authors
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

#include "xls/ir/type.h"

#include <cstdint>
#include <iterator>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

std::string TypeKindToString(TypeKind type_kind) {
  switch (type_kind) {
    case TypeKind::kTuple:
      return "tuple";
    case TypeKind::kBits:
      return "bits";
    case TypeKind::kArray:
      return "array";
    case TypeKind::kToken:
      return "token";
  }
  return absl::StrFormat("<invalid TypeKind %d>", static_cast<int>(type_kind));
}

std::ostream& operator<<(std::ostream& os, TypeKind type_kind) {
  os << TypeKindToString(type_kind);
  return os;
}

absl::StatusOr<BitsType*> Type::AsBits() {
  if (IsBits()) {
    return AsBitsOrDie();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type is not 'bits': ", *this));
}

absl::StatusOr<ArrayType*> Type::AsArray() {
  if (IsArray()) {
    return AsArrayOrDie();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type is not an array: ", *this));
}

absl::StatusOr<TupleType*> Type::AsTuple() {
  if (IsTuple()) {
    return AsTupleOrDie();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type is not a tuple: ", *this));
}

std::pair<Type*, int64_t> Type::GetSubtypeAndOffset(
    absl::Span<const int64_t> tree_index) {
  Type* cur = this;
  int64_t offset = 0;
  for (int64_t idx : tree_index) {
    if (cur->IsArray()) {
      ArrayType* array_type = cur->AsArrayOrDie();
      CHECK_GE(idx, 0);
      CHECK_LT(idx, array_type->size())
          << "Index out of bounds: [" << absl::StrJoin(tree_index, ", ")
          << "] vs " << *array_type;
      Type* elem_type = array_type->element_type();
      offset += idx * elem_type->leaf_count();
      cur = elem_type;
    } else if (cur->IsTuple()) {
      TupleType* tuple_type = cur->AsTupleOrDie();
      CHECK_GE(idx, 0);
      CHECK_LT(idx, tuple_type->size())
          << "Index out of bounds: [" << absl::StrJoin(tree_index, ", ")
          << "] vs " << *tuple_type;
      offset += tuple_type->member_leaf_offset(idx);
      cur = tuple_type->element_type(idx);
    } else {
      LOG(FATAL) << "Type is not indexable: " << *cur;
    }
  }
  return {cur, offset};
}

TypeProto BitsType::ToProto() const {
  TypeProto proto;
  proto.set_type_enum(TypeProto::BITS);
  proto.set_bit_count(bit_count());
  return proto;
}

bool BitsType::IsEqualTo(const Type* other) const {
  if (this == other) {
    return true;
  }
  return other->IsBits() && bit_count() == other->AsBitsOrDie()->bit_count();
}

TypeProto TupleType::ToProto() const {
  TypeProto proto;
  proto.set_type_enum(TypeProto::TUPLE);
  for (Type* element : element_types()) {
    *proto.add_tuple_elements() = element->ToProto();
  }
  return proto;
}

bool TupleType::IsEqualTo(const Type* other) const {
  if (this == other) {
    return true;
  }
  if (!other->IsTuple()) {
    return false;
  }
  const TupleType* other_tuple = other->AsTupleOrDie();
  if (size() != other_tuple->size()) {
    return false;
  }
  for (int64_t i = 0; i < size(); ++i) {
    if (!element_type(i)->IsEqualTo(other_tuple->element_type(i))) {
      return false;
    }
  }
  return true;
}

TypeProto ArrayType::ToProto() const {
  TypeProto proto;
  proto.set_type_enum(TypeProto::ARRAY);
  proto.set_array_size(size());
  *proto.mutable_array_element() = element_type()->ToProto();
  return proto;
}

bool ArrayType::IsEqualTo(const Type* other) const {
  if (this == other) {
    return true;
  }
  if (!other->IsArray()) {
    return false;
  }
  const ArrayType* other_array = other->AsArrayOrDie();
  return size() == other_array->size() &&
         element_type()->IsEqualTo(other_array->element_type());
}

TypeProto TokenType::ToProto() const {
  TypeProto proto;
  proto.set_type_enum(TypeProto::TOKEN);
  return proto;
}

bool TokenType::IsEqualTo(const Type* other) const {
  if (this == other) {
    return true;
  }
  return other->IsToken();
}

std::ostream& operator<<(std::ostream& os, const Type& type) {
  os << type.ToString();
  return os;
}

std::string TupleType::ToString() const {
  std::vector<std::string> pieces;
  pieces.reserve(members_.size());
  for (Type* member : members_) {
    pieces.push_back(member->ToString());
  }
  return absl::StrCat("(", absl::StrJoin(pieces, ", "), ")");
}

BitsType::BitsType(int64_t bit_count) : Type(TypeKind::kBits) {
  CHECK_GE(bit_count, 0);
  bit_count_ = bit_count;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

BitsType::BitsType(const BitsType& other) : Type(TypeKind::kBits) {
  bit_count_ = other.bit_count_;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

BitsType::BitsType(BitsType&& other) : Type(TypeKind::kBits) {
  bit_count_ = other.bit_count_;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

BitsType& BitsType::operator=(const BitsType& other) {
  bit_count_ = other.bit_count_;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
  return *this;
}

BitsType& BitsType::operator=(BitsType&& other) {
  bit_count_ = other.bit_count_;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
  return *this;
}

TokenType::TokenType() : Type(TypeKind::kToken) {
  bit_count_ = 0;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

TokenType::TokenType(const TokenType& other) : Type(TypeKind::kToken) {
  bit_count_ = 0;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

TokenType::TokenType(TokenType&& other) : Type(TypeKind::kToken) {
  bit_count_ = 0;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
}

TokenType& TokenType::operator=(const TokenType& other) {
  bit_count_ = 0;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
  return *this;
}

TokenType& TokenType::operator=(TokenType&& other) {
  bit_count_ = 0;
  leaf_types_ = {this};
  tree_index_vectors_ = {{}};
  return *this;
}

TupleType::TupleType(absl::Span<Type* const> members)
    : Type(TypeKind::kTuple), members_(members.begin(), members.end()) {
  bit_count_ = 0;
  int64_t leaf_count = absl::c_accumulate(
      members, 0,
      [](int64_t sum, Type* member) { return sum + member->leaf_count(); });
  leaf_types_.reserve(leaf_count);
  tree_index_vectors_.reserve(leaf_count);
  member_leaf_offsets_.reserve(members.size());
  for (const auto& [i, member_type] : iter::enumerate(members)) {
    member_leaf_offsets_.push_back(leaf_types_.size());
    absl::c_copy(member_type->leaf_types(), std::back_inserter(leaf_types_));
    for (int64_t j = 0; j < member_type->leaf_count(); ++j) {
      std::vector<int64_t>& tree_index = tree_index_vectors_.emplace_back();
      tree_index.reserve(member_type->tree_index_vectors()[j].size() + 1);
      tree_index.push_back(i);
      absl::c_copy(member_type->tree_index_vectors()[j],
                   std::back_inserter(tree_index));
    }
    bit_count_ += member_type->GetFlatBitCount();
  }
}

ArrayType::ArrayType(int64_t size, Type* element_type)
    : Type(TypeKind::kArray), size_(size), element_type_(element_type) {
  bit_count_ = size * element_type->GetFlatBitCount();

  leaf_types_.reserve(size * element_type->leaf_count());
  tree_index_vectors_.reserve(size * element_type->leaf_count());
  for (int64_t i = 0; i < size; ++i) {
    absl::c_copy(element_type->leaf_types(), std::back_inserter(leaf_types_));
    for (int64_t j = 0; j < element_type->leaf_count(); ++j) {
      std::vector<int64_t>& tree_index = tree_index_vectors_.emplace_back();
      tree_index.reserve(element_type->tree_index_vectors()[j].size() + 1);
      tree_index.push_back(i);
      absl::c_copy(element_type->tree_index_vectors()[j],
                   std::back_inserter(tree_index));
    }
  }
}

std::string BitsType::ToString() const {
  return absl::StrFormat("bits[%d]", bit_count());
}

std::string ArrayType::ToString() const {
  return absl::StrFormat("%s[%d]", element_type()->ToString(), size());
}

std::string TokenType::ToString() const { return absl::StrFormat("token"); }

FunctionTypeProto FunctionType::ToProto() const {
  FunctionTypeProto proto;
  for (Type* parameter : parameters()) {
    *proto.add_parameters() = parameter->ToProto();
  }
  *proto.mutable_return_type() = return_type()->ToProto();
  return proto;
}

bool FunctionType::IsEqualTo(const FunctionType* other) const {
  if (this == other) {
    return true;
  }
  if (!return_type()->IsEqualTo(other->return_type())) {
    return false;
  }
  if (parameter_count() != other->parameter_count()) {
    return false;
  }
  for (int64_t i = 0; i < parameter_count(); ++i) {
    if (!parameter_type(i)->IsEqualTo(other->parameter_type(i))) {
      return false;
    }
  }
  return true;
}

std::string FunctionType::ToString() const {
  std::vector<std::string> pieces;
  for (Type* parameter : parameters()) {
    pieces.push_back(parameter->ToString());
  }
  return absl::StrCat("(", absl::StrJoin(pieces, ", "), ") -> ",
                      return_type()->ToString());
}

std::ostream& operator<<(std::ostream& os, const Type* type) {
  os << (type == nullptr ? std::string("<nullptr Type*>") : type->ToString());
  return os;
}

absl::StatusOr<Type*> GetIndexedElementType(Type* type_to_index,
                                            int64_t index_size) {
  Type* indexed_element_type = type_to_index;
  for (int64_t i = 0; i < index_size; ++i) {
    if (!indexed_element_type->IsArray()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Index has more elements (%d) than type %s has "
                          "array dimensions (%d)",
                          index_size, type_to_index->ToString(),
                          GetArrayDimensionCount(type_to_index)));
    }
    indexed_element_type = indexed_element_type->AsArrayOrDie()->element_type();
  }
  return indexed_element_type;
}

int64_t GetArrayDimensionCount(Type* type) {
  int64_t count = 0;
  while (type->IsArray()) {
    count++;
    type = type->AsArrayOrDie()->element_type();
  }
  return count;
}

// Returns true if the given type is a token type or has a token type as an
// subelement.
bool TypeHasToken(Type* type) {
  if (type->IsToken()) {
    return true;
  }
  if (type->IsArray()) {
    return TypeHasToken(type->AsArrayOrDie()->element_type());
  }
  if (type->IsTuple()) {
    for (Type* element_type : type->AsTupleOrDie()->element_types()) {
      if (TypeHasToken(element_type)) {
        return true;
      }
    }
  }
  return false;
}

absl::StatusOr<Type*> InstantiationType::GetOutputPortType(
    std::string_view name) const {
  XLS_RET_CHECK(output_types_.contains(name));
  return output_types_.at(name);
}
absl::StatusOr<Type*> InstantiationType::GetInputPortType(
    std::string_view name) const {
  XLS_RET_CHECK(input_types_.contains(name));
  return input_types_.at(name);
}

}  // namespace xls
