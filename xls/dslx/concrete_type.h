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

// "Concrete" types are fully instantiated / known types.
//
// Type annotations in the AST are not concrete types -- when those are resolved
// by type inference, they resolve to concrete types.
//
// Concrete types are used in both type inference deduction and interpreter
// evaluation of the DSL.

#ifndef XLS_DSLX_CONCRETE_TYPE_H_
#define XLS_DSLX_CONCRETE_TYPE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "xls/common/integral_types.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/parametric_expression.h"

namespace xls::dslx {

class BitsType;
class TupleType;

// See comment on ConcreteType below.
class ConcreteTypeDim {
 public:
  using OwnedParametric = std::unique_ptr<ParametricExpression>;
  using Variant = absl::variant<int64, OwnedParametric>;

  explicit ConcreteTypeDim(Variant value) : value_(std::move(value)) {}
  explicit ConcreteTypeDim(const ParametricExpression& value)
      : value_(value.Clone()) {}
  ConcreteTypeDim(const ConcreteTypeDim& other);
  ConcreteTypeDim& operator=(ConcreteTypeDim&& other) {
    value_ = std::move(other.value_);
    return *this;
  }

  ConcreteTypeDim Clone() const;

  absl::StatusOr<ConcreteTypeDim> Mul(const ConcreteTypeDim& rhs) const;
  absl::StatusOr<ConcreteTypeDim> Add(const ConcreteTypeDim& rhs) const;

  std::string ToString() const;
  bool operator==(const ConcreteTypeDim& other) const;
  bool operator==(
      const absl::variant<int64, const ParametricExpression*>& other) const;

  const Variant& value() const { return value_; }

  bool IsParametric() const {
    return absl::holds_alternative<OwnedParametric>(value_);
  }
  const ParametricExpression& parametric() const {
    return *absl::get<OwnedParametric>(value_);
  }

 private:
  Variant value_;
};

// Represents a 'concrete' (evaluated) type, as determined by evaluating a
// TypeAnnotation in the AST.
//
// Type constructs in the AST may have abstract expressions, but ultimately they
// resolve into some concrete type when those are evaluated. By IR conversion
// time all the dimensions are fully resolvable to integers. This class
// represents both the parametric forms (before final resolution) and the
// resolved result.
//
// Note: During typechecking the dimension members may be either symbols (like
// the 'N' in bits[N,3]) or integers. Once parametric symbols are instantiated
// the symbols (such as 'N') will have resolved into ints, and we will only be
// dealing with ConcreteTypeDims that hold ints.
class ConcreteType {
 public:
  static std::unique_ptr<ConcreteType> MakeNil();

  virtual ~ConcreteType() = default;

  virtual bool operator==(const ConcreteType& other) const = 0;
  bool operator!=(const ConcreteType& other) const { return !(*this == other); }

  virtual std::string ToString() const = 0;

  // Returns whether this type contains an enum type (transitively).
  virtual bool HasEnum() const = 0;

  // Returns a flat sequence of all dimensions contained (transitively) within
  // this type.
  virtual std::vector<ConcreteTypeDim> GetAllDims() const = 0;

  virtual absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const = 0;

  // Returns a "type name" suitable for debugging; e.g. "array", "bits", "enum",
  // etc.
  virtual std::string GetDebugTypeName() const = 0;

  virtual std::unique_ptr<ConcreteType> CloneToUnique() const = 0;

  // Type equality, but ignores tuple member naming discrepancies.
  bool CompatibleWith(const ConcreteType& other) const;

  bool IsNil() const;
};

// Represents a tuple type.
//
// Tuples can have unnamed members or named members. In any case, you can
// request `tuple_type.GetUnnamedMembers()`.
//
// When the members are named the `nominal_type` may refer to the struct
// definition that led to those named members.
class TupleType : public ConcreteType {
 public:
  using UnnamedMembers = std::vector<std::unique_ptr<ConcreteType>>;
  struct NamedMember {
    std::string name;
    std::unique_ptr<ConcreteType> type;
  };
  using NamedMembers = std::vector<NamedMember>;
  using Members = std::variant<UnnamedMembers, NamedMembers>;

  TupleType(Members members, StructDef* struct_def = nullptr)
      : members_(std::move(members)), struct_def_(struct_def) {
    XLS_CHECK_EQ(struct_def_ == nullptr, !is_named());
  }

  bool operator==(const ConcreteType& other) const override {
    if (auto* t = dynamic_cast<const TupleType*>(&other)) {
      return MembersEqual(members_, t->members_);
    }
    return false;
  }
  std::string ToString() const override;
  std::vector<ConcreteTypeDim> GetAllDims() const override;
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override;
  std::string GetDebugTypeName() const override { return "tuple"; }

  bool HasEnum() const override {
    for (const ConcreteType* t : GetUnnamedMembers()) {
      if (t->HasEnum()) {
        return true;
      }
    }
    return false;
  }

  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<TupleType>(CloneMembers(), struct_def_);
  }

  bool empty() const {
    if (absl::holds_alternative<NamedMembers>(members_)) {
      return absl::get<NamedMembers>(members_).empty();
    }
    return absl::get<UnnamedMembers>(members_).empty();
  }
  int64_t size() const {
    if (absl::holds_alternative<NamedMembers>(members_)) {
      return absl::get<NamedMembers>(members_).size();
    }
    return absl::get<UnnamedMembers>(members_).size();
  }

  bool CompatibleWith(const TupleType& other) const {
    std::vector<const ConcreteType*> self_members = GetUnnamedMembers();
    std::vector<const ConcreteType*> other_members = GetUnnamedMembers();
    if (self_members.size() != other_members.size()) {
      return false;
    }
    for (int64 i = 0; i < self_members.size(); ++i) {
      if (!self_members[i]->CompatibleWith(*other_members[i])) {
        return false;
      }
    }
    // Same member count and all compatible members.
    return true;
  }

  bool is_named() const {
    return absl::holds_alternative<NamedMembers>(members_);
  }
  StructDef* nominal_type() const { return struct_def_; }

  const Members& members() const { return members_; }

  // Precondition: this TupleType must have named members and i must be <
  // members().size().
  const std::string& GetMemberName(int64 i) const {
    XLS_CHECK(absl::holds_alternative<NamedMembers>(members_));
    return absl::get<NamedMembers>(members_).at(i).name;
  }
  ConcreteType& GetMemberType(int64 i) {
    if (absl::holds_alternative<NamedMembers>(members_)) {
      return *absl::get<NamedMembers>(members_).at(i).type;
    } else {
      return *absl::get<UnnamedMembers>(members_).at(i);
    }
  }
  const ConcreteType& GetMemberType(int64 i) const {
    return const_cast<TupleType*>(this)->GetMemberType(i);
  }

  // Returns an error status if this TupleType does not have named members.
  absl::StatusOr<std::vector<std::string>> GetMemberNames() const {
    if (!absl::holds_alternative<NamedMembers>(members_)) {
      return absl::InvalidArgumentError(
          "Tuple has unnamed members; cannot retrieve names.");
    }
    std::vector<std::string> results;
    for (const NamedMember& m : absl::get<NamedMembers>(members_)) {
      results.push_back(m.name);
    }
    return results;
  }

  std::vector<const ConcreteType*> GetUnnamedMembers() const;

  absl::optional<const ConcreteType*> GetMemberTypeByName(
      absl::string_view target) const;

  bool HasNamedMember(absl::string_view target) const;

 private:
  static bool MembersEqual(const UnnamedMembers& a, const UnnamedMembers& b);

  static bool MembersEqual(const NamedMembers& a, const NamedMembers& b);

  static bool MembersEqual(const Members& a, const Members& b);

  Members CloneMembers() const;

  Members members_;
  StructDef* struct_def_;  // Note: may be null.
};

// Represents an array type, with an element type and size.
//
// These will nest in the case of multidimensional arrays.
class ArrayType : public ConcreteType {
 public:
  ArrayType(std::unique_ptr<ConcreteType> element_type, ConcreteTypeDim size)
      : element_type_(std::move(element_type)), size_(std::move(size)) {}

  std::string ToString() const override;
  std::vector<ConcreteTypeDim> GetAllDims() const override;
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override;
  bool HasEnum() const override { return element_type_->HasEnum(); }
  bool operator==(const ConcreteType& other) const override {
    if (auto* o = dynamic_cast<const ArrayType*>(&other)) {
      return size_ == o->size_ && *element_type_ == *o->element_type_;
    }
    return false;
  }
  std::string GetDebugTypeName() const override { return "array"; }
  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<ArrayType>(element_type_->CloneToUnique(),
                                        size_.Clone());
  }

  const ConcreteType& element_type() const { return *element_type_; }
  const ConcreteTypeDim& size() const { return size_; }

 private:
  std::unique_ptr<ConcreteType> element_type_;
  ConcreteTypeDim size_;
};

// Represents an enum type.
class EnumType : public ConcreteType {
 public:
  EnumType(EnumDef* enum_def, ConcreteTypeDim bit_count)
      : enum_def_(XLS_DIE_IF_NULL(enum_def)), size_(std::move(bit_count)) {}

  std::string ToString() const override;
  std::vector<ConcreteTypeDim> GetAllDims() const override;
  bool HasEnum() const override { return true; }
  std::string GetDebugTypeName() const override { return "enum"; }
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override {
    return size_.Clone();
  }
  bool operator==(const ConcreteType& other) const override {
    if (auto* o = dynamic_cast<const EnumType*>(&other)) {
      return enum_def_ == o->enum_def_ && size_ == o->size_;
    }
    return false;
  }
  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<EnumType>(enum_def_, size_.Clone());
  }

  EnumDef* nominal_type() const { return enum_def_; }
  const ConcreteTypeDim& size() const { return size_; }

  absl::optional<bool> signedness() const { return enum_def_->signedness(); }

 private:
  EnumDef* enum_def_;
  ConcreteTypeDim size_;
};

// Represents a bits type (either signed or unsigned).
//
// Note that there are related helpers IsUBits() and IsSBits() for concisely
// testing whether a `ConcreteType` is an unsigned or signed BitsType,
// respectively.
class BitsType : public ConcreteType {
 public:
  static std::unique_ptr<BitsType> MakeU32() {
    return absl::make_unique<BitsType>(false, 32);
  }
  static std::unique_ptr<BitsType> MakeU8() {
    return absl::make_unique<BitsType>(false, 8);
  }
  static std::unique_ptr<BitsType> MakeU1() {
    return absl::make_unique<BitsType>(false, 1);
  }

  BitsType(bool is_signed, int64 size)
      : BitsType(is_signed, ConcreteTypeDim(size)) {}
  BitsType(bool is_signed, ConcreteTypeDim size)
      : is_signed_(is_signed), size_(std::move(size)) {}
  ~BitsType() override = default;

  bool operator==(const ConcreteType& other) const override {
    if (auto* t = dynamic_cast<const BitsType*>(&other)) {
      return t->is_signed_ == is_signed_ && t->size_ == size_;
    }
    return false;
  }
  std::string ToString() const override;
  std::string GetDebugTypeName() const override;
  bool HasEnum() const override { return false; }

  std::unique_ptr<BitsType> ToUBits() const;

  bool is_signed() const { return is_signed_; }
  const ConcreteTypeDim& size() const { return size_; }

  std::vector<ConcreteTypeDim> GetAllDims() const override {
    std::vector<ConcreteTypeDim> result;
    result.push_back(size_.Clone());
    return result;
  }
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override {
    return size_.Clone();
  }
  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<BitsType>(is_signed_, size_.Clone());
  }

 private:
  bool is_signed_;
  ConcreteTypeDim size_;
};

// Represents a function type with params and a return type.
class FunctionType : public ConcreteType {
 public:
  FunctionType(std::vector<std::unique_ptr<ConcreteType>> params,
               std::unique_ptr<ConcreteType> return_type)
      : params_(std::move(params)), return_type_(std::move(return_type)) {}

  std::string ToString() const override;
  std::string GetDebugTypeName() const override { return "function"; }
  std::vector<ConcreteTypeDim> GetAllDims() const override;
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override;

  bool operator==(const ConcreteType& other) const override {
    if (auto* o = dynamic_cast<const FunctionType*>(&other)) {
      if (params_.size() != o->params_.size()) {
        return false;
      }
      for (int64 i = 0; i < params_.size(); ++i) {
        if (*params_[i] != *o->params_[i]) {
          return false;
        }
      }
      return *return_type_ == *o->return_type_;
    }
    return false;
  }

  std::vector<const ConcreteType*> GetParams() const;
  int64 GetParamCount() const { return params_.size(); }

  bool HasEnum() const override {
    auto has_enum = [](const auto& param) { return param->HasEnum(); };
    return std::any_of(params_.begin(), params_.end(), has_enum) ||
           return_type_->HasEnum();
  }

  const ConcreteType& return_type() const { return *return_type_; }

  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    std::vector<std::unique_ptr<ConcreteType>> params;
    for (const auto& item : params_) {
      params.push_back(item->CloneToUnique());
    }
    return absl::make_unique<FunctionType>(std::move(params),
                                           return_type_->CloneToUnique());
  }

 private:
  std::vector<std::unique_ptr<ConcreteType>> params_;
  std::unique_ptr<ConcreteType> return_type_;
};

// Returns whether the given concrete type is a unsigned/signed BitsType (for
// IsUBits/IsSBits respectively).
inline bool IsUBits(const ConcreteType& c) {
  if (auto* b = dynamic_cast<const BitsType*>(&c); b != nullptr) {
    return !b->is_signed();
  }
  return false;
}
inline bool IsSBits(const ConcreteType& c) {
  if (auto* b = dynamic_cast<const BitsType*>(&c); b != nullptr) {
    return b->is_signed();
  }
  return false;
}

inline std::unique_ptr<ConcreteType> ConcreteType::MakeNil() {
  return absl::make_unique<TupleType>(TupleType::UnnamedMembers{});
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONCRETE_TYPE_H_
