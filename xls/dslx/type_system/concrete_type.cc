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

#include "xls/dslx/type_system/concrete_type.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

ConcreteType::~ConcreteType() = default;

/* static */ bool ConcreteType::Equal(
    absl::Span<const std::unique_ptr<ConcreteType>> a,
    absl::Span<const std::unique_ptr<ConcreteType>> b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int64_t i = 0; i < a.size(); ++i) {
    if (*a[i] != *b[i]) {
      return false;
    }
  }
  return true;
}

/* static */ std::vector<std::unique_ptr<ConcreteType>> ConcreteType::CloneSpan(
    absl::Span<const std::unique_ptr<ConcreteType>> ts) {
  std::vector<std::unique_ptr<ConcreteType>> result;
  result.reserve(ts.size());
  for (const auto& t : ts) {
    XLS_CHECK(t != nullptr);
    XLS_VLOG(10) << "CloneSpan; cloning: " << t->ToString();
    result.push_back(t->CloneToUnique());
  }
  return result;
}

std::unique_ptr<ConcreteType> ConcreteType::MakeUnit() {
  return std::make_unique<TupleType>(
      std::vector<std::unique_ptr<ConcreteType>>{});
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcreteType::FromInterpValue(
    const InterpValue& value) {
  if (value.tag() == InterpValueTag::kUBits ||
      value.tag() == InterpValueTag::kSBits) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
    return std::make_unique<BitsType>(/*is_signed*/ value.IsSigned(),
                                      /*size=*/bit_count);
  }

  if (value.tag() == InterpValueTag::kArray) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         value.GetValues());
    if (elements->empty()) {
      return absl::InvalidArgumentError(
          "Cannot get the ConcreteType of a 0-element array.");
    }
    XLS_ASSIGN_OR_RETURN(auto element_type, FromInterpValue(elements->at(0)));
    XLS_ASSIGN_OR_RETURN(int64_t size, value.GetLength());
    auto dim = ConcreteTypeDim::CreateU32(size);
    return std::make_unique<ArrayType>(std::move(element_type), dim);
  }

  if (value.tag() == InterpValueTag::kTuple) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         value.GetValues());
    std::vector<std::unique_ptr<ConcreteType>> members;
    members.reserve(elements->size());
    for (const auto& element : *elements) {
      XLS_ASSIGN_OR_RETURN(auto member, FromInterpValue(element));
      members.push_back(std::move(member));
    }

    return std::make_unique<TupleType>(std::move(members));
  }

  return absl::InvalidArgumentError(
      "Only bits, array, and tuple types can be converted into concrete.");
}

// -- class ConcreteTypeDim

/* static */ absl::StatusOr<int64_t> ConcreteTypeDim::GetAs64Bits(
    const std::variant<InterpValue, OwnedParametric>& variant) {
  if (std::holds_alternative<InterpValue>(variant)) {
    return std::get<InterpValue>(variant).GetBitValueCheckSign();
  }

  return absl::InvalidArgumentError(
      "Can't evaluate a ParametricExpression to an integer.");
}

ConcreteTypeDim::ConcreteTypeDim(const ConcreteTypeDim& other)
    : value_(std::move(other.Clone().value_)) {}

ConcreteTypeDim ConcreteTypeDim::Clone() const {
  if (std::holds_alternative<InterpValue>(value_)) {
    return ConcreteTypeDim(std::get<InterpValue>(value_));
  }
  if (std::holds_alternative<std::unique_ptr<ParametricExpression>>(value_)) {
    return ConcreteTypeDim(
        std::get<std::unique_ptr<ParametricExpression>>(value_)->Clone());
  }
  XLS_LOG(FATAL) << "Impossible ConcreteTypeDim value variant.";
}

std::string ConcreteTypeDim::ToString() const {
  if (std::holds_alternative<std::unique_ptr<ParametricExpression>>(value_)) {
    return std::get<std::unique_ptr<ParametricExpression>>(value_)->ToString();
  }
  // Note: we don't print out the type/width of the InterpValue that serves as
  // the dimension, because printing `uN[u32:42]` would appear odd vs just
  // `uN[42]`.
  //
  // TODO(https://github.com/google/xls/issues/450) the best solution may to be
  // to have a size type that all InterpValues present on real type dimensions
  // must be. Things are trickier nowadays because we want to permit arbitrary
  // InterpValues to be passed as parametrics, not just ones that become (used)
  // dimension data -- we need to allow for e.g. signed types which may not end
  // up in any particular dimension position.
  return std::get<InterpValue>(value_).GetBitsOrDie().ToString();
}

bool ConcreteTypeDim::operator==(
    const std::variant<int64_t, InterpValue, const ParametricExpression*>&
        other) const {
  if (std::holds_alternative<InterpValue>(other)) {
    if (std::holds_alternative<InterpValue>(value_)) {
      return std::get<InterpValue>(value_) == std::get<InterpValue>(other);
    }
    return false;
  }
  if (std::holds_alternative<const ParametricExpression*>(other)) {
    return std::holds_alternative<std::unique_ptr<ParametricExpression>>(
               value_) &&
           *std::get<std::unique_ptr<ParametricExpression>>(value_) ==
               *std::get<const ParametricExpression*>(other);
  }

  return false;
}

bool ConcreteTypeDim::operator==(const ConcreteTypeDim& other) const {
  if (std::holds_alternative<std::unique_ptr<ParametricExpression>>(value_) &&
      std::holds_alternative<std::unique_ptr<ParametricExpression>>(
          other.value_)) {
    return *std::get<std::unique_ptr<ParametricExpression>>(value_) ==
           *std::get<std::unique_ptr<ParametricExpression>>(other.value_);
  }
  return value_ == other.value_;
}

absl::StatusOr<ConcreteTypeDim> ConcreteTypeDim::Mul(
    const ConcreteTypeDim& rhs) const {
  if (std::holds_alternative<InterpValue>(value_) &&
      std::holds_alternative<InterpValue>(rhs.value_)) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        std::get<InterpValue>(value_).Mul(std::get<InterpValue>(rhs.value_)));
    return ConcreteTypeDim(std::move(result));
  }
  if (IsParametric() && rhs.IsParametric()) {
    return ConcreteTypeDim(std::make_unique<ParametricMul>(
        parametric().Clone(), rhs.parametric().Clone()));
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot multiply dimensions: %s * %s", ToString(), rhs.ToString()));
}

absl::StatusOr<ConcreteTypeDim> ConcreteTypeDim::Add(
    const ConcreteTypeDim& rhs) const {
  if (std::holds_alternative<InterpValue>(value_) &&
      std::holds_alternative<InterpValue>(rhs.value_)) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        std::get<InterpValue>(value_).Add(std::get<InterpValue>(rhs.value_)));
    return ConcreteTypeDim(result);
  }
  if (IsParametric() && rhs.IsParametric()) {
    return ConcreteTypeDim(std::make_unique<ParametricAdd>(
        parametric().Clone(), rhs.parametric().Clone()));
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot add dimensions: %s + %s", ToString(), rhs.ToString()));
}

absl::StatusOr<int64_t> ConcreteTypeDim::GetAsInt64() const {
  if (std::holds_alternative<InterpValue>(value_)) {
    InterpValue value = std::get<InterpValue>(value_);
    if (!value.IsBits()) {
      return absl::InvalidArgumentError(
          "Cannot convert non-bits type to int64_t.");
    }

    if (value.IsSigned()) {
      return value.GetBitValueInt64();
    }
    return value.GetBitValueUint64();
  }

  std::optional<InterpValue> maybe_value = parametric().const_value();
  if (maybe_value.has_value()) {
    InterpValue value = maybe_value.value();
    if (value.IsBits()) {
      if (value.IsSigned()) {
        return value.GetBitValueInt64();
      }
      return value.GetBitValueUint64();
    }
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Expected concrete type dimension to be integral; got: ",
                   std::get<OwnedParametric>(value_)->ToString()));
}

// -- ConcreteType

bool ConcreteType::CompatibleWith(const ConcreteType& other) const {
  if (*this == other) {
    return true;  // Equality implies compatibility.
  }

  // For types that encapsulate other types, they may not be strictly equal, but
  // the contained types may still be compatible.
  if (auto [t, u] = std::make_pair(dynamic_cast<const TupleType*>(this),
                                   dynamic_cast<const TupleType*>(&other));
      t != nullptr && u != nullptr) {
    return t->CompatibleWith(*u);
  }

  if (auto [t, u] = std::make_pair(dynamic_cast<const ArrayType*>(this),
                                   dynamic_cast<const ArrayType*>(&other));
      t != nullptr && u != nullptr) {
    return t->element_type().CompatibleWith(u->element_type()) &&
           t->size() == u->size();
  }

  return false;
}

bool ConcreteType::IsUnit() const {
  if (auto* t = dynamic_cast<const TupleType*>(this)) {
    return t->empty();
  }
  return false;
}

bool ConcreteType::IsStruct() const {
  return dynamic_cast<const StructType*>(this) != nullptr;
}

bool ConcreteType::IsArray() const {
  return dynamic_cast<const ArrayType*>(this) != nullptr;
}

bool ConcreteType::IsToken() const {
  return dynamic_cast<const TokenType*>(this) != nullptr;
}

const StructType& ConcreteType::AsStruct() const {
  auto* s = dynamic_cast<const StructType*>(this);
  XLS_CHECK(s != nullptr) << "ConcreteType is not a struct: " << ToString();
  return *s;
}

const ArrayType& ConcreteType::AsArray() const {
  auto* s = dynamic_cast<const ArrayType*>(this);
  XLS_CHECK(s != nullptr) << "ConcreteType is not an array: " << ToString();
  return *s;
}

// -- TokenType

TokenType::~TokenType() = default;

// -- BitsType

absl::StatusOr<std::unique_ptr<ConcreteType>> BitsType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size, f(size_));
  return std::make_unique<BitsType>(is_signed_, new_size);
}

std::string BitsType::ToString() const {
  return absl::StrFormat("%cN[%s]", is_signed_ ? 's' : 'u', size_.ToString());
}

std::string BitsType::GetDebugTypeName() const {
  return is_signed_ ? "sbits" : "ubits";
}

std::unique_ptr<BitsType> BitsType::ToUBits() const {
  return std::make_unique<BitsType>(false, size_.Clone());
}

// -- StructType

StructType::StructType(std::vector<std::unique_ptr<ConcreteType>> members,
                       const StructDef& struct_def)
    : members_(std::move(members)), struct_def_(struct_def) {
  XLS_CHECK_EQ(members_.size(), struct_def_.members().size());
}

bool StructType::HasEnum() const {
  for (const std::unique_ptr<ConcreteType>& type : members_) {
    if (type->HasEnum()) {
      return true;
    }
  }
  return false;
}

std::string StructType::ToErrorString() const {
  return absl::StrFormat("struct '%s' structure: %s",
                         nominal_type().identifier(), ToString());
}

std::string StructType::ToString() const {
  std::string guts;
  for (int64_t i = 0; i < members().size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&guts, ", ");
    }
    absl::StrAppendFormat(&guts, "%s: %s", GetMemberName(i),
                          GetMemberType(i).ToString());
  }
  if (!guts.empty()) {
    guts = absl::StrCat(" ", guts, " ");
  }
  return absl::StrCat(nominal_type().identifier(), " {", guts, "}");
}

absl::StatusOr<std::vector<std::string>> StructType::GetMemberNames() const {
  std::vector<std::string> results;
  results.reserve(members().size());
  for (int64_t i = 0; i < members().size(); ++i) {
    results.push_back(std::string(GetMemberName(i)));
  }
  return results;
}

absl::StatusOr<int64_t> StructType::GetMemberIndex(
    std::string_view name) const {
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> names, GetMemberNames());
  auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Name not present in tuple type %s: %s", ToString(), name));
  }
  return std::distance(names.begin(), it);
}

std::optional<const ConcreteType*> StructType::GetMemberTypeByName(
    std::string_view target) const {
  for (int64_t i = 0; i < members().size(); ++i) {
    if (GetMemberName(i) == target) {
      return &GetMemberType(i);
    }
  }
  return std::nullopt;
}

std::vector<ConcreteTypeDim> StructType::GetAllDims() const {
  std::vector<ConcreteTypeDim> results;
  for (const std::unique_ptr<ConcreteType>& type : members_) {
    std::vector<ConcreteTypeDim> t_dims = type->GetAllDims();
    for (auto& dim : t_dims) {
      results.push_back(std::move(dim));
    }
  }
  return results;
}

absl::StatusOr<ConcreteTypeDim> StructType::GetTotalBitCount() const {
  auto sum = ConcreteTypeDim::CreateU32(0);
  for (const std::unique_ptr<ConcreteType>& t : members_) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim elem_bit_count, t->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(elem_bit_count));
  }

  return sum;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> StructType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  std::vector<std::unique_ptr<ConcreteType>> new_members;
  for (const std::unique_ptr<ConcreteType>& member : members_) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> mapped,
                         member->MapSize(f));
    new_members.push_back(std::move(mapped));
  }
  return std::make_unique<StructType>(std::move(new_members), struct_def_);
}

bool StructType::HasNamedMember(std::string_view target) const {
  for (int64_t i = 0; i < members().size(); ++i) {
    if (GetMemberName(i) == target) {
      return true;
    }
  }
  return false;
}

bool StructType::operator==(const ConcreteType& other) const {
  if (auto* t = dynamic_cast<const StructType*>(&other)) {
    return Equal(members_, t->members_) && &struct_def_ == &t->struct_def_;
  }
  return false;
}

// -- TupleType

TupleType::TupleType(std::vector<std::unique_ptr<ConcreteType>> members)
    : members_(std::move(members)) {
#ifndef NDEBUG
  for (const auto& member : members_) {
    XLS_DCHECK(member != nullptr);
  }
#endif
}

bool TupleType::operator==(const ConcreteType& other) const {
  if (auto* t = dynamic_cast<const TupleType*>(&other)) {
    return Equal(members_, t->members_);
  }
  return false;
}

bool TupleType::HasEnum() const {
  for (const std::unique_ptr<ConcreteType>& t : members_) {
    if (t->HasEnum()) {
      return true;
    }
  }
  return false;
}

bool TupleType::empty() const { return members_.empty(); }

int64_t TupleType::size() const { return members_.size(); }

bool TupleType::CompatibleWith(const TupleType& other) const {
  if (members_.size() != other.members_.size()) {
    return false;
  }
  for (int64_t i = 0; i < members_.size(); ++i) {
    if (!members_[i]->CompatibleWith(*other.members_[i])) {
      return false;
    }
  }
  // Same member count and all compatible members.
  return true;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> TupleType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  std::vector<std::unique_ptr<ConcreteType>> new_members;
  for (const auto& member_type : members_) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> mapped,
                         member_type->MapSize(f));
    new_members.push_back(std::move(mapped));
  }
  return std::make_unique<TupleType>(std::move(new_members));
}

std::unique_ptr<ConcreteType> TupleType::CloneToUnique() const {
  return std::make_unique<TupleType>(CloneSpan(members_));
}

std::string TupleType::ToString() const {
  std::string guts = absl::StrJoin(
      members_, ", ",
      [](std::string* out, const std::unique_ptr<ConcreteType>& m) {
        absl::StrAppend(out, m->ToString());
      });
  return absl::StrCat("(", guts, ")");
}

std::vector<ConcreteTypeDim> TupleType::GetAllDims() const {
  std::vector<ConcreteTypeDim> results;
  for (const std::unique_ptr<ConcreteType>& t : members_) {
    std::vector<ConcreteTypeDim> t_dims = t->GetAllDims();
    for (auto& dim : t_dims) {
      results.push_back(std::move(dim));
    }
  }
  return results;
}

absl::StatusOr<ConcreteTypeDim> TupleType::GetTotalBitCount() const {
  auto sum = ConcreteTypeDim::CreateU32(0);
  for (const std::unique_ptr<ConcreteType>& t : members_) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim elem_bit_count, t->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(elem_bit_count));
  }

  return sum;
}

// -- ArrayType

absl::StatusOr<std::unique_ptr<ConcreteType>> ArrayType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> new_element_type,
                       element_type_->MapSize(f));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size, f(size_));
  return std::make_unique<ArrayType>(std::move(new_element_type),
                                     std::move(new_size));
}

std::string ArrayType::ToString() const {
  return absl::StrFormat("%s[%s]", element_type_->ToString(), size_.ToString());
}

std::vector<ConcreteTypeDim> ArrayType::GetAllDims() const {
  std::vector<ConcreteTypeDim> results;
  results.push_back(size_.Clone());
  std::vector<ConcreteTypeDim> element_dims = element_type_->GetAllDims();
  for (auto& dim : element_dims) {
    results.push_back(std::move(dim));
  }
  return results;
}

absl::StatusOr<ConcreteTypeDim> ArrayType::GetTotalBitCount() const {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim elem_bits,
                       element_type_->GetTotalBitCount());
  return elem_bits.Mul(size_);
}

// -- EnumType

absl::StatusOr<std::unique_ptr<ConcreteType>> EnumType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size, f(size_));
  return std::make_unique<EnumType>(enum_def_, std::move(new_size), is_signed_,
                                    members_);
}

std::string EnumType::ToString() const { return enum_def_.identifier(); }

std::vector<ConcreteTypeDim> EnumType::GetAllDims() const {
  std::vector<ConcreteTypeDim> result;
  result.push_back(size_.Clone());
  return result;
}

// -- FunctionType

absl::StatusOr<std::unique_ptr<ConcreteType>> FunctionType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  std::vector<std::unique_ptr<ConcreteType>> new_params;
  for (const auto& param : params_) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> new_param,
                         param->MapSize(f));
    new_params.push_back(std::move(new_param));
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> new_return_type,
                       return_type_->MapSize(f));
  return std::make_unique<FunctionType>(std::move(new_params),
                                        std::move(new_return_type));
}

bool FunctionType::operator==(const ConcreteType& other) const {
  if (auto* o = dynamic_cast<const FunctionType*>(&other)) {
    if (params_.size() != o->params_.size()) {
      return false;
    }
    for (int64_t i = 0; i < params_.size(); ++i) {
      if (*params_[i] != *o->params_[i]) {
        return false;
      }
    }
    return *return_type_ == *o->return_type_;
  }
  return false;
}

std::vector<const ConcreteType*> FunctionType::GetParams() const {
  std::vector<const ConcreteType*> results;
  results.reserve(params_.size());
  for (const auto& param : params_) {
    results.push_back(param.get());
  }
  return results;
}

std::string FunctionType::ToString() const {
  std::string params_str = absl::StrJoin(
      params_, ", ",
      [](std::string* out, const std::unique_ptr<ConcreteType>& t) {
        absl::StrAppend(out, t->ToString());
      });
  return absl::StrFormat("(%s) -> %s", params_str, return_type_->ToString());
}

std::vector<ConcreteTypeDim> FunctionType::GetAllDims() const {
  std::vector<ConcreteTypeDim> results;
  for (const auto& param : params_) {
    std::vector<ConcreteTypeDim> param_dims = param->GetAllDims();
    for (auto& dim : param_dims) {
      results.push_back(std::move(dim));
    }
  }
  return results;
}

absl::StatusOr<ConcreteTypeDim> FunctionType::GetTotalBitCount() const {
  auto sum = ConcreteTypeDim::CreateU32(0);
  for (const auto& param : params_) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim param_bits, param->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(param_bits));
  }
  return sum;
}

ChannelType::ChannelType(std::unique_ptr<ConcreteType> payload_type,
                         ChannelDirection direction)
    : payload_type_(std::move(payload_type)), direction_(direction) {
  XLS_CHECK(payload_type_ != nullptr);
}

std::string ChannelType::ToString() const {
  return absl::StrFormat("chan(%s, dir=%s)", payload_type_->ToString(),
                         direction_ == ChannelDirection::kIn ? "in" : "out");
}

std::vector<ConcreteTypeDim> ChannelType::GetAllDims() const {
  return payload_type_->GetAllDims();
}

absl::StatusOr<ConcreteTypeDim> ChannelType::GetTotalBitCount() const {
  return payload_type_->GetTotalBitCount();
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ChannelType::MapSize(
    const MapFn& f) const {
  XLS_ASSIGN_OR_RETURN(auto new_payload, payload_type_->MapSize(f));
  return std::make_unique<ChannelType>(std::move(new_payload), direction_);
}

bool ChannelType::operator==(const ConcreteType& other) const {
  if (auto* o = dynamic_cast<const ChannelType*>(&other)) {
    return *payload_type_ == *o->payload_type_ && direction_ == o->direction_;
  }
  return false;
}

bool ChannelType::HasEnum() const { return payload_type_->HasEnum(); }

std::unique_ptr<ConcreteType> ChannelType::CloneToUnique() const {
  return std::make_unique<ChannelType>(payload_type_->CloneToUnique(),
                                       direction_);
}

absl::StatusOr<bool> IsSigned(const ConcreteType& c) {
  if (auto* bits = dynamic_cast<const BitsType*>(&c)) {
    return bits->is_signed();
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&c)) {
    std::optional<bool> signedness = enum_type->is_signed();
    if (!signedness.has_value()) {
      return absl::InvalidArgumentError(
          "Signedness not present for EnumType: " + c.ToString());
    }
    return signedness.value();
  }
  return absl::InvalidArgumentError(
      "Cannot determined signedness; type is neither enum nor bits.");
}

const ParametricSymbol* TryGetParametricSymbol(const ConcreteTypeDim& dim) {
  if (!std::holds_alternative<ConcreteTypeDim::OwnedParametric>(dim.value())) {
    return nullptr;
  }
  const ParametricExpression* parametric =
      std::get<ConcreteTypeDim::OwnedParametric>(dim.value()).get();
  return dynamic_cast<const ParametricSymbol*>(parametric);
}

}  // namespace xls::dslx
