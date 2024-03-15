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

#ifndef XLS_DSLX_TYPE_SYSTEM_TYPE_H_
#define XLS_DSLX_TYPE_SYSTEM_TYPE_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_expression.h"

namespace xls::dslx {

class BitsType;
class TupleType;
class StructType;

// Represents a parametric binding in a Type, which is either a) a
// parametric expression or b) evaluated to an InterpValue. When type
// annotations from the AST are evaluated in type inference, some of the
// "concrete types" that result from deduction may still be parametric. When the
// parametric code is instantiated, the concrete types then have concrete
// `InterpValues` as their dimensions.
//
// See comment on `Type` below for more details.
class TypeDim {
 public:
  using OwnedParametric = std::unique_ptr<ParametricExpression>;

  // Evaluates the given value to a 64-bit quantity.
  static absl::StatusOr<int64_t> GetAs64Bits(
      const std::variant<InterpValue, OwnedParametric>& variant);

  // Creates a u32 `InterpValue`-based TypeDim with the given "value".
  static TypeDim CreateU32(uint32_t value) {
    return TypeDim(InterpValue::MakeU32(value));
  }

  // Creates a `u1` / `bool` `InterpValue`-based TypeDim with the given "value".
  static TypeDim CreateBool(bool value) {
    return TypeDim(InterpValue::MakeBool(value));
  }

  explicit TypeDim(std::variant<InterpValue, OwnedParametric> value)
      : value_(std::move(value)) {}

  TypeDim(const TypeDim& other);
  TypeDim(TypeDim&& other) = default;
  TypeDim& operator=(TypeDim&& other) = default;

  TypeDim Clone() const;

  // Arithmetic operators used in e.g. calculating total bit counts.
  absl::StatusOr<TypeDim> Mul(const TypeDim& rhs) const;
  absl::StatusOr<TypeDim> Add(const TypeDim& rhs) const;
  absl::StatusOr<TypeDim> CeilOfLog2() const;

  // Returns a string representation of this dimension, which is either the
  // integral string or the parametric expression string conversion.
  std::string ToString() const;

  bool operator==(const TypeDim& other) const;
  bool operator==(const std::variant<InterpValue, const ParametricExpression*>&
                      other) const;
  bool operator!=(const TypeDim& other) const { return !(*this == other); }

  const std::variant<InterpValue, OwnedParametric>& value() const {
    return value_;
  }

  // Note: if it is not parametric it is an interpreter value.
  bool IsParametric() const {
    return std::holds_alternative<OwnedParametric>(value_);
  }

  const ParametricExpression& parametric() const {
    return *std::get<OwnedParametric>(value_);
  }

  // Retrieves the bit value of the underlying InterpValue as an int64_t.
  absl::StatusOr<int64_t> GetAsInt64() const;

  // Retrieves the bit value of the underlying InterpValue as a bool.
  absl::StatusOr<bool> GetAsBool() const;

 private:
  std::variant<InterpValue, OwnedParametric> value_;
};

inline std::ostream& operator<<(std::ostream& os, const TypeDim& ctd) {
  os << ctd.ToString();
  return os;
}

class EnumType;
class BitsType;
class FunctionType;
class ChannelType;
class TokenType;
class StructType;
class TupleType;
class ArrayType;
class MetaType;
class BitsConstructorType;

// Abstract base class for a Type visitor.
class TypeVisitor {
 public:
  virtual ~TypeVisitor() = default;

  virtual absl::Status HandleEnum(const EnumType& t) = 0;
  virtual absl::Status HandleBits(const BitsType& t) = 0;
  virtual absl::Status HandleBitsConstructor(const BitsConstructorType& t) = 0;
  virtual absl::Status HandleFunction(const FunctionType& t) = 0;
  virtual absl::Status HandleChannel(const ChannelType& t) = 0;
  virtual absl::Status HandleToken(const TokenType& t) = 0;
  virtual absl::Status HandleStruct(const StructType& t) = 0;
  virtual absl::Status HandleTuple(const TupleType& t) = 0;
  virtual absl::Status HandleArray(const ArrayType& t) = 0;
  virtual absl::Status HandleMeta(const MetaType& t) = 0;
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
// the 'N' in `bits[N][3]`) or integers. Once parametric symbols are
// instantiated the symbols (such as 'N') will have resolved into ints, and we
// will only be dealing with TypeDims that hold ints.
class Type {
 public:
  using MapFn = std::function<absl::StatusOr<TypeDim>(TypeDim)>;

  // Creates a "unit" tuple type (a tuple type with no members).
  static std::unique_ptr<Type> MakeUnit();

  // Creates a concrete type matching that of the given InterpValue.
  static absl::StatusOr<std::unique_ptr<Type>> FromInterpValue(
      const InterpValue& value);

  virtual ~Type();

  virtual absl::Status Accept(TypeVisitor& v) const = 0;

  // Note that this is type equivalence; e.g. types are equivalent and
  // substitutable if this equality holds.
  virtual bool operator==(const Type& other) const = 0;
  bool operator!=(const Type& other) const { return !(*this == other); }

  virtual std::string ToString() const = 0;

  // Variation on `ToString()` to be used in user-facing error reporting.
  virtual std::string ToErrorString() const { return ToString(); }

  // Returns whether this type contains an enum type (transitively).
  virtual bool HasEnum() const = 0;

  // Returns a flat sequence of all dimensions contained (transitively) within
  // this type.
  virtual std::vector<TypeDim> GetAllDims() const = 0;

  bool HasParametricDims() const {
    std::vector<TypeDim> all_dims = GetAllDims();
    return std::any_of(all_dims.begin(), all_dims.end(),
                       [](const TypeDim& d) { return d.IsParametric(); });
  }

  virtual absl::StatusOr<TypeDim> GetTotalBitCount() const = 0;

  // Returns a "type name" suitable for debugging; e.g. "array", "bits", "enum",
  // etc.
  virtual std::string GetDebugTypeName() const = 0;

  // Creates a new unique clone of this Type.
  //
  // Postcondition: *retval == *this.
  virtual std::unique_ptr<Type> CloneToUnique() const = 0;

  // Maps all the dimensions contained (transitively) within this Type
  // (and any concrete types held herein) with "f" and returns the new
  // (resulting) Type.
  virtual absl::StatusOr<std::unique_ptr<Type>> MapSize(
      const MapFn& f) const = 0;

  // Type equality, but ignores tuple member naming discrepancies.
  bool CompatibleWith(const Type& other) const;

  bool IsUnit() const;
  bool IsToken() const;
  bool IsStruct() const;
  bool IsArray() const;
  bool IsMeta() const;

  const StructType& AsStruct() const;
  const ArrayType& AsArray() const;

 protected:
  static std::vector<std::unique_ptr<Type>> CloneSpan(
      absl::Span<const std::unique_ptr<Type>> ts);

  static bool Equal(absl::Span<const std::unique_ptr<Type>> a,
                    absl::Span<const std::unique_ptr<Type>> b);
};

inline std::ostream& operator<<(std::ostream& os, const Type& t) {
  os << t.ToString();
  return os;
}

// Indicates that the deduced entity is a type expression -- as opposed to "an
// expression having type T" this indicates "it was type T itself".
//
// For example, if you do `deduce(u32)` where `u32` is the builtin type
// annotation, it will tell you "this thing is a type" and contain "the type is
// u32".
class MetaType : public Type {
 public:
  explicit MetaType(std::unique_ptr<Type> wrapped)
      : wrapped_(std::move(wrapped)) {}

  ~MetaType() override;

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleMeta(*this);
  }
  bool operator==(const Type& other) const override {
    if (const auto* o = dynamic_cast<const MetaType*>(&other)) {
      return *wrapped() == *o->wrapped();
    }
    return false;
  }
  std::string GetDebugTypeName() const override { return "meta-type"; }
  std::string ToString() const override {
    return absl::StrCat("typeof(", wrapped_->ToString(), ")");
  }

  bool HasEnum() const override { return wrapped_->HasEnum(); }
  std::vector<TypeDim> GetAllDims() const override {
    return wrapped_->GetAllDims();
  }
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override {
    XLS_ASSIGN_OR_RETURN(auto wrapped, wrapped_->MapSize(f));
    return std::make_unique<MetaType>(std::move(wrapped));
  }
  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<MetaType>(wrapped_->CloneToUnique());
  }

  const std::unique_ptr<Type>& wrapped() const { return wrapped_; }
  std::unique_ptr<Type>& wrapped() { return wrapped_; }

 private:
  std::unique_ptr<Type> wrapped_;
};

// Represents the type of a token value.
//
// Tokens are *existential values* used for establishing dataflow dependencies
// for sequence-sensitive operations.
//
// Currently the token type is effectively a singleton-behaving value like nil,
// it cannot be distinguished or parameterized in any way from other token type
// instances (unless you compare by pointer, of course, which is an
// implementation detail and not part of the Type API).
class TokenType : public Type {
 public:
  ~TokenType() override;

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleToken(*this);
  }
  bool operator==(const Type& other) const override { return other.IsToken(); }
  std::string ToString() const override { return "token"; }
  std::vector<TypeDim> GetAllDims() const override { return {}; }
  absl::StatusOr<TypeDim> GetTotalBitCount() const override {
    return TypeDim(InterpValue::MakeU32(0));
  }
  std::string GetDebugTypeName() const override { return "token"; }

  bool HasEnum() const override { return false; }

  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<TokenType>();
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override {
    return std::make_unique<TokenType>();
  }
};

// Represents a struct type -- these are similar in spirit to "tuples with named
// fields", but they also identify the nominal struct that they correspond to --
// things like type comparisons
class StructType : public Type {
 public:
  // Note: members must correspond to struct_def's members (same length and
  // order).
  StructType(std::vector<std::unique_ptr<Type>> members,
             const StructDef& struct_def);

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleStruct(*this);
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  bool operator==(const Type& other) const override;
  std::string ToString() const override;
  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  std::string GetDebugTypeName() const override { return "struct"; }

  bool HasEnum() const override;

  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<StructType>(CloneSpan(members_), struct_def_);
  }

  // For user-level error reporting, we also note the name of the struct
  // definition if one is available.
  std::string ToErrorString() const override;

  // Returns an InvalidArgument error status if this TupleType does not have
  // named members.
  absl::StatusOr<std::vector<std::string>> GetMemberNames() const;

  std::string_view GetMemberName(int64_t i) const {
    return struct_def_.GetMemberName(i);
  }

  const Type& GetMemberType(int64_t i) const { return *members_.at(i); }

  // Returns the index of the member with name "name" -- returns a NotFound
  // error if the member is not found (i.e. it is generally expected that the
  // caller knows the name is present), and an InvalidArgument error status if
  // this TupleType does not have named members.
  absl::StatusOr<int64_t> GetMemberIndex(std::string_view name) const;

  std::optional<const Type*> GetMemberTypeByName(std::string_view target) const;

  const StructDef& nominal_type() const { return struct_def_; }

  bool HasNamedMember(std::string_view target) const;

  int64_t size() const { return members_.size(); }

  const std::vector<std::unique_ptr<Type>>& members() const { return members_; }

 private:
  std::vector<std::unique_ptr<Type>> members_;
  const StructDef& struct_def_;
};

// Represents a tuple type. Tuples have unnamed members.
class TupleType : public Type {
 public:
  static std::unique_ptr<TupleType> Create2(std::unique_ptr<Type> t0,
                                            std::unique_ptr<Type> t1) {
    std::vector<std::unique_ptr<Type>> members;
    members.push_back(std::move(t0));
    members.push_back(std::move(t1));
    return std::make_unique<TupleType>(std::move(members));
  }
  static std::unique_ptr<TupleType> Create3(std::unique_ptr<Type> t0,
                                            std::unique_ptr<Type> t1,
                                            std::unique_ptr<Type> t2) {
    std::vector<std::unique_ptr<Type>> members;
    members.push_back(std::move(t0));
    members.push_back(std::move(t1));
    members.push_back(std::move(t2));
    return std::make_unique<TupleType>(std::move(members));
  }

  explicit TupleType(std::vector<std::unique_ptr<Type>> members);

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleTuple(*this);
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  bool operator==(const Type& other) const override;
  std::string ToString() const override;
  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  std::string GetDebugTypeName() const override { return "tuple"; }

  bool HasEnum() const override;

  std::unique_ptr<Type> CloneToUnique() const override;

  bool empty() const;
  int64_t size() const;

  bool CompatibleWith(const TupleType& other) const;

  Type& GetMemberType(int64_t i) { return *members_.at(i); }
  const Type& GetMemberType(int64_t i) const { return *members_.at(i); }
  const std::vector<std::unique_ptr<Type>>& members() const { return members_; }

 private:
  std::vector<std::unique_ptr<Type>> members_;
};

// Represents an array type, with an element type and size.
//
// These will nest in the case of multidimensional arrays.
class ArrayType : public Type {
 public:
  ArrayType(std::unique_ptr<Type> element_type, const TypeDim& size)
      : element_type_(std::move(element_type)), size_(size) {
    CHECK(!element_type_->IsMeta()) << element_type_->ToString();
  }

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleArray(*this);
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  std::string ToString() const override;
  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  bool HasEnum() const override { return element_type_->HasEnum(); }

  bool operator==(const Type& other) const override;

  std::string GetDebugTypeName() const override { return "array"; }
  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<ArrayType>(element_type_->CloneToUnique(),
                                       size_.Clone());
  }

  const Type& element_type() const { return *element_type_; }
  const TypeDim& size() const { return size_; }

 private:
  std::unique_ptr<Type> element_type_;
  TypeDim size_;
};

// Represents an enum type.
class EnumType : public Type {
 public:
  EnumType(const EnumDef& enum_def, TypeDim bit_count, bool is_signed,
           const std::vector<InterpValue>& members)
      : enum_def_(enum_def),
        size_(std::move(bit_count)),
        is_signed_(is_signed),
        members_(members) {}

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleEnum(*this);
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  std::string ToString() const override;
  std::vector<TypeDim> GetAllDims() const override;
  bool HasEnum() const override { return true; }
  std::string GetDebugTypeName() const override { return "enum"; }
  absl::StatusOr<TypeDim> GetTotalBitCount() const override {
    return size_.Clone();
  }
  bool operator==(const Type& other) const override {
    if (auto* o = dynamic_cast<const EnumType*>(&other)) {
      return &enum_def_ == &o->enum_def_ && size_ == o->size_ &&
             is_signed_ == o->is_signed_;
    }
    return false;
  }
  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<EnumType>(enum_def_, size_.Clone(), is_signed_,
                                      members_);
  }

  const EnumDef& nominal_type() const { return enum_def_; }
  const TypeDim& size() const { return size_; }
  const std::vector<InterpValue>& members() const { return members_; }

  bool is_signed() const { return is_signed_; }

 private:
  const EnumDef& enum_def_;  // Definition AST node.
  TypeDim size_;             // Underlying size in bits.
  bool is_signed_;           // Signedness of the underlying bits type.
  std::vector<InterpValue> members_;  // Member values of the enum.
};

// This represents the type of annotations like:
//    bits
//    uN
//    sN
//    xN
//
// Note that the last one has parametric signedness.
class BitsConstructorType : public Type {
 public:
  explicit BitsConstructorType(TypeDim is_signed);

  ~BitsConstructorType() override;

  absl::Status Accept(TypeVisitor& v) const override;
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  bool operator==(const Type& other) const override;
  std::string ToString() const override;
  std::string GetDebugTypeName() const override;
  bool HasEnum() const override;

  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  std::unique_ptr<Type> CloneToUnique() const override;

  const TypeDim& is_signed() const { return is_signed_; }

 private:
  TypeDim is_signed_;
};

// Represents a bits type (either signed or unsigned).
//
// Signedness is given by `is_signed()` and is always literal -- the type that
// can make the signedness parametric is given in `BitsConstructorType`.
//
// Note that there are related helpers IsUBits() and IsSBits() for concisely
// testing whether a `Type` is an unsigned or signed BitsType,
// respectively.
class BitsType : public Type {
 public:
  static std::unique_ptr<BitsType> MakeU64() {
    return std::make_unique<BitsType>(false, 64);
  }
  static std::unique_ptr<BitsType> MakeU32() {
    return std::make_unique<BitsType>(false, 32);
  }
  static std::unique_ptr<BitsType> MakeS32() {
    return std::make_unique<BitsType>(true, 32);
  }
  static std::unique_ptr<BitsType> MakeU8() {
    return std::make_unique<BitsType>(false, 8);
  }
  static std::unique_ptr<BitsType> MakeU1() {
    return std::make_unique<BitsType>(false, 1);
  }

  BitsType(bool is_signed, int64_t size);

  BitsType(bool is_signed, TypeDim size);

  ~BitsType() override = default;

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleBits(*this);
  }
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  bool operator==(const Type& other) const override;
  std::string ToString() const override;
  std::string GetDebugTypeName() const override;
  bool HasEnum() const override { return false; }

  std::unique_ptr<BitsType> ToUBits() const;

  // Returns whether this bits type is a signed type.
  bool is_signed() const { return is_signed_; }

  // Returns the dize (dimension) for this bits type.
  const TypeDim& size() const { return size_; }

  std::vector<TypeDim> GetAllDims() const override {
    std::vector<TypeDim> result;
    result.push_back(size_.Clone());
    return result;
  }
  absl::StatusOr<TypeDim> GetTotalBitCount() const override {
    return size_.Clone();
  }
  std::unique_ptr<Type> CloneToUnique() const override {
    return std::make_unique<BitsType>(is_signed_, size_.Clone());
  }

 private:
  bool is_signed_;
  TypeDim size_;
};

// Represents a function type with params and a return type.
class FunctionType : public Type {
 public:
  FunctionType(std::vector<std::unique_ptr<Type>> params,
               std::unique_ptr<Type> return_type)
      : params_(std::move(params)), return_type_(std::move(return_type)) {
    CHECK(!return_type_->IsMeta());
  }

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleFunction(*this);
  }
  std::string ToString() const override;
  std::string GetDebugTypeName() const override { return "function"; }
  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;

  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;

  bool operator==(const Type& other) const override;

  // Accessor for owned parameter (formal argument) type vector.
  const std::vector<std::unique_ptr<Type>>& params() const { return params_; }

  // Returns a (new value) vector of pointers to parameter types owned by this
  // FunctionType.
  std::vector<const Type*> GetParams() const;

  // Number of parameters (formal arguments) to this function.
  int64_t GetParamCount() const { return params_.size(); }

  bool HasEnum() const override {
    auto has_enum = [](const auto& param) { return param->HasEnum(); };
    return std::any_of(params_.begin(), params_.end(), has_enum) ||
           return_type_->HasEnum();
  }

  // Return type of the function.
  const Type& return_type() const { return *return_type_; }

  std::unique_ptr<Type> CloneToUnique() const override {
    std::vector<std::unique_ptr<Type>> params;
    params.reserve(params_.size());
    for (const auto& item : params_) {
      params.push_back(item->CloneToUnique());
    }
    return std::make_unique<FunctionType>(std::move(params),
                                          return_type_->CloneToUnique());
  }

 private:
  std::vector<std::unique_ptr<Type>> params_;
  std::unique_ptr<Type> return_type_;
};

// Represents the type of a channel (half-duplex), which effectively just wraps
// its payload type and has an associated direction.
//
// Attrs:
//  payload_type: The type of the values that flow through the channel. Note
//    that this cannot be a metatype (checked on construction).
class ChannelType : public Type {
 public:
  ChannelType(std::unique_ptr<Type> payload_type, ChannelDirection direction);

  absl::Status Accept(TypeVisitor& v) const override {
    return v.HandleChannel(*this);
  }
  std::string ToString() const override;
  std::string GetDebugTypeName() const override { return "channel"; }
  std::vector<TypeDim> GetAllDims() const override;
  absl::StatusOr<TypeDim> GetTotalBitCount() const override;
  absl::StatusOr<std::unique_ptr<Type>> MapSize(const MapFn& f) const override;
  bool operator==(const Type& other) const override;
  bool HasEnum() const override;
  std::unique_ptr<Type> CloneToUnique() const override;

  const Type& payload_type() const { return *payload_type_; }
  ChannelDirection direction() const { return direction_; }

 private:
  std::unique_ptr<Type> payload_type_;
  ChannelDirection direction_;
};

// Helper for the case where we have a derived (i.e. non-abstract) Type,
// and want the clone to also be a unique_ptr of the derived type.
//
// Note: we explicitly instantiate it so we get better type error messages than
// exposing the parametric version directly to callers.
template <typename T>
inline std::unique_ptr<T> CloneToUniqueInternal(const T& type) {
  static_assert(std::is_base_of_v<Type, T>);
  std::unique_ptr<Type> cloned = type.CloneToUnique();
  return absl::WrapUnique<T>(dynamic_cast<T*>(cloned.release()));
}
inline std::unique_ptr<BitsType> CloneToUnique(const BitsType& type) {
  return CloneToUniqueInternal(type);
}
inline std::unique_ptr<TupleType> CloneToUnique(const TupleType& type) {
  return CloneToUniqueInternal(type);
}
inline std::unique_ptr<StructType> CloneToUnique(const StructType& type) {
  return CloneToUniqueInternal(type);
}
inline std::unique_ptr<FunctionType> CloneToUnique(const FunctionType& type) {
  return CloneToUniqueInternal(type);
}

// As above, but works on a vector of Type unique pointers.
inline std::vector<std::unique_ptr<Type>> CloneToUnique(
    absl::Span<const std::unique_ptr<Type>> types) {
  std::vector<std::unique_ptr<Type>> result;
  for (const auto& item : types) {
    result.push_back(item->CloneToUnique());
  }
  return result;
}

// As above, but works on a vector of Type unique pointers.
inline std::vector<std::unique_ptr<Type>> CloneToUnique(
    absl::Span<const Type* const> types) {
  std::vector<std::unique_ptr<Type>> result;
  for (const auto* item : types) {
    result.push_back(item->CloneToUnique());
  }
  return result;
}

// TODO(https://github.com/google/xls/issues/480) replace these dynamic casts
// with uses of a TypeVisitor.
// Returns whether the given concrete type is a unsigned/signed BitsType (for
// IsUBits/IsSBits respectively).
inline bool IsUBits(const Type& c) {
  if (auto* b = dynamic_cast<const BitsType*>(&c); b != nullptr) {
    return !b->is_signed();
  }
  return false;
}
inline bool IsSBits(const Type& c) {
  if (auto* b = dynamic_cast<const BitsType*>(&c); b != nullptr) {
    return b->is_signed();
  }
  return false;
}

inline bool IsBitsConstructor(const Type& t) {
  return dynamic_cast<const BitsConstructorType*>(&t) != nullptr;
}
inline bool IsArrayOfBitsConstructor(
    const Type& t, const BitsConstructorType** elem_type_out = nullptr) {
  if (auto* a = dynamic_cast<const ArrayType*>(&t)) {
    if (auto* bc =
            dynamic_cast<const BitsConstructorType*>(&a->element_type())) {
      if (elem_type_out != nullptr) {
        *elem_type_out = bc;
      }
      return true;
    }
  }
  return false;
}

// This predicate should be used when we want to know if a type is effectively a
// bits type -- this includes either the bits type itself or a sized array of
// bits-constructor type.
bool IsBitsLike(const Type& t);

struct BitsLikeProperties {
  TypeDim is_signed;
  TypeDim size;
};

std::optional<BitsLikeProperties> GetBitsLike(const Type& t);

inline bool IsBitsLikeWithNBitsAndSignedness(const Type& t, bool is_signed,
                                             uint32_t size) {
  const TypeDim want = TypeDim::CreateU32(size);
  if (auto* b = dynamic_cast<const BitsType*>(&t)) {
    return b->is_signed() == is_signed && b->size() == want;
  }

  const BitsConstructorType* bc;
  if (IsArrayOfBitsConstructor(t, &bc)) {
    return bc->is_signed() == want &&
           down_cast<const ArrayType&>(t).size() == want;
  }

  return false;
}

// Returns whether the given type, which should be either a bits or an enum
// type, is signed.
absl::StatusOr<bool> IsSigned(const Type& c);

// Attempts to get a ParametricSymbol contained in the given dimension, or
// nullptr if there is none.
const ParametricSymbol* TryGetParametricSymbol(const TypeDim& dim);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_TYPE_H_
