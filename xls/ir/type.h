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

#ifndef XLS_IR_TYPE_H_
#define XLS_IR_TYPE_H_

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

class BitsType;
class TupleType;
class ArrayType;
class TokenType;
class FunctionType;

enum class TypeKind { kTuple, kBits, kArray, kToken };

std::string TypeKindToString(TypeKind kind);
std::ostream& operator<<(std::ostream& os, TypeKind type_kind);

// Abstract base class for types represented in the IR.
//
// The abstract type base can be checked-downconverted via the As*() methods
// below.
class Type {
 public:
  virtual ~Type() = default;

  // Creates a Type object from the given proto.
  static absl::StatusOr<std::unique_ptr<Type>> FromProto(
      const TypeProto& proto);

  // Returns a proto representation of the type.
  virtual TypeProto ToProto() const = 0;

  TypeKind kind() const { return kind_; }

  // Returns true if this type and 'other' represent the same type.
  virtual bool IsEqualTo(const Type* other) const = 0;

  bool IsBits() const { return kind_ == TypeKind::kBits; }
  BitsType* AsBitsOrDie();
  const BitsType* AsBitsOrDie() const;
  absl::StatusOr<BitsType*> AsBits();

  bool IsTuple() const { return kind_ == TypeKind::kTuple; }
  TupleType* AsTupleOrDie();
  const TupleType* AsTupleOrDie() const;

  bool IsArray() const { return kind_ == TypeKind::kArray; }
  ArrayType* AsArrayOrDie();
  const ArrayType* AsArrayOrDie() const;
  absl::StatusOr<ArrayType*> AsArray();

  bool IsToken() const { return kind_ == TypeKind::kToken; }
  TokenType* AsTokenOrDie();
  const TokenType* AsTokenOrDie() const;

  // Returns the count of bits required to represent the underlying type; e.g.
  // for tuples this will be the sum of the bit count from all its members, for
  // a "bits" type it will be the count of bits.
  virtual int64_t GetFlatBitCount() const = 0;

  // Returns the number of leaf Bits types in this object.
  virtual int64_t leaf_count() const = 0;

  virtual std::string ToString() const = 0;

 protected:
  explicit Type(TypeKind kind) : kind_(kind) {}

 private:
  TypeKind kind_;
};

std::ostream& operator<<(std::ostream& os, const Type& type);
std::ostream& operator<<(std::ostream& os, const Type* type);

// Represents a type that is vector of bits with a fixed bit_count().
class BitsType : public Type {
 public:
  explicit BitsType(int64_t bit_count);
  ~BitsType() override {}
  int64_t bit_count() const { return bit_count_; }

  TypeProto ToProto() const override;
  bool IsEqualTo(const Type* other) const override;
  int64_t GetFlatBitCount() const override { return bit_count(); }

  int64_t leaf_count() const override { return 1; }

  // Returns a string like "bits[32]".
  std::string ToString() const override;

 private:
  int64_t bit_count_;
};

// Represents a type that is a tuple (of values of other potentially different
// types).
//
// Note that tuples can be empty.
class TupleType : public Type {
 public:
  explicit TupleType(absl::Span<Type* const> members)
      : Type(TypeKind::kTuple), members_(members.begin(), members.end()) {
    leaf_count_ = 0;
    for (Type* t : members) {
      leaf_count_ += t->leaf_count();
    }
  }
  ~TupleType() override {}
  std::string ToString() const override;

  TypeProto ToProto() const override;
  bool IsEqualTo(const Type* other) const override;

  // Returns the number of elements of the tuple.
  int64_t size() const { return members_.size(); }

  // Returns the type of the given element.
  Type* element_type(int64_t index) const { return members_.at(index); }

  // Returns the element types of the tuple.
  absl::Span<Type* const> element_types() const { return members_; }

  int64_t leaf_count() const override { return leaf_count_; }

  int64_t GetFlatBitCount() const override {
    int64_t total = 0;
    for (const Type* type : members_) {
      total += type->GetFlatBitCount();
    }
    return total;
  }

 private:
  int64_t leaf_count_;
  std::vector<Type*> members_;
};

// Represents a type that is a one-dimensional array of identical types.
//
// Note that arrays can be empty.
class ArrayType : public Type {
 public:
  explicit ArrayType(int64_t size, Type* element_type)
      : Type(TypeKind::kArray), size_(size), element_type_(element_type) {}
  ~ArrayType() override {}
  std::string ToString() const override;

  TypeProto ToProto() const override;
  bool IsEqualTo(const Type* other) const override;

  Type* element_type() const { return element_type_; }
  int64_t size() const { return size_; }

  int64_t GetFlatBitCount() const override {
    return element_type_->GetFlatBitCount() * size_;
  }

  int64_t leaf_count() const override {
    return size_ * element_type()->leaf_count();
  }

 private:
  int64_t size_;
  Type* element_type_;
};

// Represents a token type used for ordering channel accesses.
class TokenType : public Type {
 public:
  explicit TokenType() : Type(TypeKind::kToken) {}
  ~TokenType() override {}
  std::string ToString() const override;

  TypeProto ToProto() const override;
  bool IsEqualTo(const Type* other) const override;

  // Tokens contain no bits.
  int64_t GetFlatBitCount() const override { return 0; }
  int64_t leaf_count() const override { return 0; }
};

// Represents a type that is a function with parameters and return type.
class FunctionType {
 public:
  explicit FunctionType(absl::Span<Type* const> parameters, Type* return_type)
      : parameters_(parameters.begin(), parameters.end()),
        return_type_(return_type) {}
  ~FunctionType() {}
  std::string ToString() const;

  FunctionTypeProto ToProto() const;
  bool IsEqualTo(const FunctionType* other) const;

  int64_t parameter_count() const { return parameters_.size(); }
  absl::Span<Type* const> parameters() const { return parameters_; }
  Type* parameter_type(int64_t i) const { return parameters_.at(i); }
  Type* return_type() const { return return_type_; }

 private:
  std::vector<Type*> parameters_;
  Type* return_type_;
};

// -- Inlines

inline const BitsType* Type::AsBitsOrDie() const {
  XLS_CHECK_EQ(kind(), TypeKind::kBits) << ToString();
  return down_cast<const BitsType*>(this);
}

inline BitsType* Type::AsBitsOrDie() {
  XLS_CHECK_EQ(kind(), TypeKind::kBits) << ToString();
  return down_cast<BitsType*>(this);
}

inline const TupleType* Type::AsTupleOrDie() const {
  XLS_CHECK_EQ(kind(), TypeKind::kTuple);
  return down_cast<const TupleType*>(this);
}

inline TupleType* Type::AsTupleOrDie() {
  XLS_CHECK_EQ(kind(), TypeKind::kTuple);
  return down_cast<TupleType*>(this);
}

inline const ArrayType* Type::AsArrayOrDie() const {
  XLS_CHECK_EQ(kind(), TypeKind::kArray);
  return down_cast<const ArrayType*>(this);
}

inline ArrayType* Type::AsArrayOrDie() {
  XLS_CHECK_EQ(kind(), TypeKind::kArray);
  return down_cast<ArrayType*>(this);
}

inline const TokenType* Type::AsTokenOrDie() const {
  XLS_CHECK_EQ(kind(), TypeKind::kToken);
  return down_cast<const TokenType*>(this);
}

inline TokenType* Type::AsTokenOrDie() {
  XLS_CHECK_EQ(kind(), TypeKind::kToken);
  return down_cast<TokenType*>(this);
}

// Returns type of the nested element inside of "type_to_index" resulting from
// indexing that type with an index containing "index_size" elements. As a
// degenerate case, "type_to_index" need not be an array type if "index_size" is
// zero. In this case, indexing is the identity operation.
//
// Examples:
//   GetIndexedElementType(bits[32][3][4][5], 0)
//     => bits[32][3][4][5]
//   GetIndexedElementType(bits[32][3][4][5], 2)
//     => bits[32][3]
//   GetIndexedElementType(bits[32][3][4][5], 3)
//     => bits[32]
//   GetIndexedElementType(bits[42], 0)
//     => bits[42]
absl::StatusOr<Type*> GetIndexedElementType(Type* type_to_index,
                                            int64_t index_size);

// Returns the number of array dimensions of the given type. A non-array type is
// considered to have zero array dimensions.
//
// Examples:
//   GetArrayDimensionCount(bits[32]) => 0
//   GetArrayDimensionCount((bits[32], bits[111])) => 0
//   GetArrayDimensionCount(bits[8][9]) => 1
//   GetArrayDimensionCount(bits[8][9][10]) => 2
//   GetArrayDimensionCount((bits[8][10], token)[42]) => 1
int64_t GetArrayDimensionCount(Type* type);

// Returns true if the given type is a token type or has a token type as a
// subelement.
bool TypeHasToken(Type* type);

}  // namespace xls

#endif  // XLS_IR_TYPE_H_
