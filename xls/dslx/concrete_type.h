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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parametric_expression.h"

namespace xls::dslx {

class BitsType;
class TupleType;

// See comment on ConcreteType below.
// TODO(rspringer): 2020-03-25: Some users of ConcreteTypeDim use it to
// represent signed values (think array slicing with a negative index). Type
// dimensions, though, don't make sense as negative values, so we need to find
// all such uses and convert them to another abstraction.
class ConcreteTypeDim {
 public:
  using OwnedParametric = std::unique_ptr<ParametricExpression>;
  // This class _accepts_ an InputVariant, but only stores either the
  // integral or ParametricExpression variants (a "Variant").
  // An accepted InterpValue input variant must be convertable to a uint64_t.
  using InputVariant = absl::variant<int64_t, InterpValue, OwnedParametric>;
  using Variant = absl::variant<int64_t, OwnedParametric>;

  // Evaluates the given value to a 64-bit quantity.
  // TODO(rspringer): 2021-03-25: Eliminate the InputVariant...variant. Once
  // done, consider eliminating the Variant alias.
  static absl::StatusOr<int64_t> GetAs64Bits(const Variant& variant) {
    if (absl::holds_alternative<int64_t>(variant)) {
      return absl::get<int64_t>(variant);
    }

    return absl::InvalidArgumentError(
        "Can't evaluate a ParametricExpression to an integer.");
  }

  static absl::StatusOr<int64_t> GetAs64Bits(const InputVariant& variant) {
    if (absl::holds_alternative<InterpValue>(variant)) {
      return absl::get<InterpValue>(variant).GetBitValueCheckSign();
    }

    if (absl::holds_alternative<int64_t>(variant)) {
      return absl::get<int64_t>(variant);
    }

    return absl::InvalidArgumentError(
        "Can't evaluate a ParametricExpression to an integer.");
  }

  static absl::StatusOr<ConcreteTypeDim> Create(InputVariant variant) {
    if (absl::holds_alternative<InterpValue>(variant)) {
      XLS_ASSIGN_OR_RETURN(
          int64_t int_value,
          absl::get<InterpValue>(variant).GetBitValueCheckSign());
      return ConcreteTypeDim(int_value);
    }

    if (auto ptr = absl::get_if<int64_t>(&variant)) {
      return ConcreteTypeDim(*ptr);
    }

    OwnedParametric op = std::move(absl::get<OwnedParametric>(variant));
    return ConcreteTypeDim(std::move(op));
  }

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

  // Returns a string representation of this dimension, which is either the
  // integral string or the parametric expression string conversion.
  std::string ToString() const;

  // Returns a Python-style "representation" string that shows the construction
  // of this value; e.g. "ConcreteTypeDim(42)".
  std::string ToRepr() const;

  bool operator==(const ConcreteTypeDim& other) const;
  bool operator==(
      const absl::variant<int64_t, InterpValue, const ParametricExpression*>&
          other) const;
  bool operator!=(const ConcreteTypeDim& other) const {
    return !(*this == other);
  }

  const Variant& value() const { return value_; }

  bool IsParametric() const {
    return absl::holds_alternative<OwnedParametric>(value_);
  }
  const ParametricExpression& parametric() const {
    return *absl::get<OwnedParametric>(value_);
  }

 private:
  explicit ConcreteTypeDim(Variant value) : value_(std::move(value)) {}
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
  // Creates a "nil tuple" type (a tuple type with no members).
  static std::unique_ptr<ConcreteType> MakeNil();

  // Creates a concrete type matching that of the given InterpValue.
  static absl::StatusOr<std::unique_ptr<ConcreteType>> FromInterpValue(
      const InterpValue& value);

  virtual ~ConcreteType() = default;

  virtual bool operator==(const ConcreteType& other) const = 0;
  bool operator!=(const ConcreteType& other) const { return !(*this == other); }

  virtual std::string ToString() const = 0;
  virtual std::string ToRepr() const { return ToString(); }

  // Variation on `ToString()` to be used in user-facing error reporting.
  virtual std::string ToErrorString() const { return ToString(); }

  // Returns whether this type contains an enum type (transitively).
  virtual bool HasEnum() const = 0;

  // Returns a flat sequence of all dimensions contained (transitively) within
  // this type.
  virtual std::vector<ConcreteTypeDim> GetAllDims() const = 0;

  virtual absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const = 0;

  // Returns a "type name" suitable for debugging; e.g. "array", "bits", "enum",
  // etc.
  virtual std::string GetDebugTypeName() const = 0;

  // Creates a new unique clone of this ConcreteType.
  //
  // Postcondition: *retval == *this.
  virtual std::unique_ptr<ConcreteType> CloneToUnique() const = 0;

  // Maps all the dimensions contained (transitively) within this ConcreteType
  // (and any concrete types held herein) with "f" and returns the new
  // (resulting) ConcreteType.
  virtual absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const = 0;

  // Type equality, but ignores tuple member naming discrepancies.
  bool CompatibleWith(const ConcreteType& other) const;

  bool IsNil() const;
  bool IsToken() const;
};

inline std::ostream& operator<<(std::ostream& os, const ConcreteType& t) {
  os << t.ToString();
  return os;
}

// Represents the type of a token value.
//
// Tokens are *existential values* used for establishing dataflow dependencies
// for sequence-sensitive operations.
//
// Currently the token type is effectively a singleton-behaving value like nil,
// it cannot be distinguished or parameterized in any way from other token type
// instances (unless you compare by pointer, of course, which is an
// implementation detail and not part of the ConcreteType API).
class TokenType : public ConcreteType {
 public:
  ~TokenType() override;

  bool operator==(const ConcreteType& other) const override {
    return other.IsToken();
  }
  std::string ToString() const override { return "token"; }
  std::vector<ConcreteTypeDim> GetAllDims() const override { return {}; }
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override {
    return ConcreteTypeDim::Create(0);
  }
  std::string GetDebugTypeName() const override { return "token"; }

  bool HasEnum() const override { return false; }

  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<TokenType>();
  }
  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override {
    return absl::make_unique<TokenType>();
  }
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

  TupleType(Members members, StructDef* struct_def = nullptr);

  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override;

  bool operator==(const ConcreteType& other) const override;
  std::string ToString() const override;
  std::vector<ConcreteTypeDim> GetAllDims() const override;
  absl::StatusOr<ConcreteTypeDim> GetTotalBitCount() const override;
  std::string GetDebugTypeName() const override { return "tuple"; }

  // For user-level error reporting, we also note the name of the struct
  // definition if one is available.
  std::string ToErrorString() const override;

  bool HasEnum() const override;

  std::unique_ptr<ConcreteType> CloneToUnique() const override {
    return absl::make_unique<TupleType>(CloneMembers(), struct_def_);
  }

  bool empty() const;
  int64_t size() const;

  bool CompatibleWith(const TupleType& other) const;

  bool is_named() const {
    return absl::holds_alternative<NamedMembers>(members_);
  }
  StructDef* nominal_type() const { return struct_def_; }

  const Members& members() const { return members_; }

  // Precondition: this TupleType must have named members and i must be <
  // members().size().
  const std::string& GetMemberName(int64_t i) const {
    XLS_CHECK(absl::holds_alternative<NamedMembers>(members_));
    return absl::get<NamedMembers>(members_).at(i).name;
  }
  ConcreteType& GetMemberType(int64_t i) {
    if (absl::holds_alternative<NamedMembers>(members_)) {
      return *absl::get<NamedMembers>(members_).at(i).type;
    } else {
      return *absl::get<UnnamedMembers>(members_).at(i);
    }
  }
  const ConcreteType& GetMemberType(int64_t i) const {
    return const_cast<TupleType*>(this)->GetMemberType(i);
  }

  // Returns an InvalidArgument error status if this TupleType does not have
  // named members.
  absl::StatusOr<std::vector<std::string>> GetMemberNames() const;

  // Returns the index of the member with name "name" -- returns a NotFound
  // error if the member is not found (i.e. it is generally expected that the
  // caller knows the name is present), and an InvalidArgument error status if
  // this TupleType does not have named members.
  absl::StatusOr<int64_t> GetMemberIndex(absl::string_view name) const;

  std::vector<const ConcreteType*> GetUnnamedMembers() const;
  const ConcreteType& GetUnnamedMember(int64_t i) const {
    const ConcreteType* result = GetUnnamedMembers()[i];
    XLS_CHECK(result != nullptr);
    return *result;
  }

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

  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override;

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

  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override;

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
  static std::unique_ptr<BitsType> MakeS32() {
    return absl::make_unique<BitsType>(true, 32);
  }
  static std::unique_ptr<BitsType> MakeU8() {
    return absl::make_unique<BitsType>(false, 8);
  }
  static std::unique_ptr<BitsType> MakeU1() {
    return absl::make_unique<BitsType>(false, 1);
  }

  BitsType(bool is_signed, int64_t size)
      : BitsType(is_signed, ConcreteTypeDim::Create(size).value()) {}
  BitsType(bool is_signed, ConcreteTypeDim size)
      : is_signed_(is_signed), size_(std::move(size)) {}
  ~BitsType() override = default;

  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override;

  bool operator==(const ConcreteType& other) const override {
    if (auto* t = dynamic_cast<const BitsType*>(&other)) {
      return t->is_signed_ == is_signed_ && t->size_ == size_;
    }
    return false;
  }
  std::string ToString() const override;
  std::string ToRepr() const override {
    return absl::StrFormat("BitsType(is_signed=%s, size=%s)",
                           is_signed_ ? "true" : "false", size_.ToRepr());
  }
  std::string GetDebugTypeName() const override;
  bool HasEnum() const override { return false; }

  std::unique_ptr<BitsType> ToUBits() const;

  // Returns whether this bits type is a signed type.
  bool is_signed() const { return is_signed_; }

  // Returns the dize (dimension) for this bits type.
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

  absl::StatusOr<std::unique_ptr<ConcreteType>> MapSize(
      const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
      const override;

  bool operator==(const ConcreteType& other) const override;

  // Accessor for owned parameter (formal argument) type vector.
  const std::vector<std::unique_ptr<ConcreteType>>& params() const {
    return params_;
  }

  // Returns a (new value) vector of pointers to parameter types owned by this
  // FunctionType.
  std::vector<const ConcreteType*> GetParams() const;

  // Number of parameters (formal arguments) to this function.
  int64_t GetParamCount() const { return params_.size(); }

  bool HasEnum() const override {
    auto has_enum = [](const auto& param) { return param->HasEnum(); };
    return std::any_of(params_.begin(), params_.end(), has_enum) ||
           return_type_->HasEnum();
  }

  // Return type of the function.
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

// Helper for the case where we have a derived (i.e. non-abstract) ConcreteType,
// and want the clone to also be a unique_ptr of the derived type.
//
// Note: we explicitly instantiate it so we get better type error messages than
// exposing the parametric version directly to callers.
template <typename T>
inline std::unique_ptr<T> CloneToUniqueInternal(const T& type) {
  static_assert(std::is_base_of<ConcreteType, T>::value);
  std::unique_ptr<ConcreteType> cloned = type.CloneToUnique();
  return absl::WrapUnique<T>(dynamic_cast<T*>(cloned.release()));
}
inline std::unique_ptr<BitsType> CloneToUnique(const BitsType& type) {
  return CloneToUniqueInternal(type);
}
inline std::unique_ptr<TupleType> CloneToUnique(const TupleType& type) {
  return CloneToUniqueInternal(type);
}
inline std::unique_ptr<FunctionType> CloneToUnique(const FunctionType& type) {
  return CloneToUniqueInternal(type);
}

// As above, but works on a vector of ConcreteType unique pointers.
inline std::vector<std::unique_ptr<ConcreteType>> CloneToUnique(
    absl::Span<const std::unique_ptr<ConcreteType>> types) {
  std::vector<std::unique_ptr<ConcreteType>> result;
  for (const auto& item : types) {
    result.push_back(item->CloneToUnique());
  }
  return result;
}

// As above, but works on a vector of ConcreteType unique pointers.
inline std::vector<std::unique_ptr<ConcreteType>> CloneToUnique(
    absl::Span<const ConcreteType* const> types) {
  std::vector<std::unique_ptr<ConcreteType>> result;
  for (const auto* item : types) {
    result.push_back(item->CloneToUnique());
  }
  return result;
}

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

// Returns whether the given type, which should be either a bits or an enum
// type, is signed.
absl::StatusOr<bool> IsSigned(const ConcreteType& c);

// Parses the given string (that is expected to come from a
// ConcreteType::ToString() invocation) back into a ConcreteType object.
// Trailing text that is not part of the resulting parsed type is left in "s".
//
// Returns an error if the string cannot be parsed, and the state of s is not
// defined (make a defensive copy if s will be used in the case of error).
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcreteTypeFromString(
    absl::string_view* s);

}  // namespace xls::dslx

#endif  // XLS_DSLX_CONCRETE_TYPE_H_
