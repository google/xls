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

#include "xls/dslx/concrete_type.h"

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

ConcreteTypeDim::ConcreteTypeDim(const ConcreteTypeDim& other)
    : value_(std::move(other.Clone().value_)) {}

ConcreteTypeDim ConcreteTypeDim::Clone() const {
  if (absl::holds_alternative<int64>(value_)) {
    return ConcreteTypeDim(absl::get<int64>(value_));
  }
  if (absl::holds_alternative<std::unique_ptr<ParametricExpression>>(value_)) {
    return ConcreteTypeDim(
        absl::get<std::unique_ptr<ParametricExpression>>(value_)->Clone());
  }
  XLS_LOG(FATAL) << "Impossible ConcreteTypeDim value variant.";
}

std::string ConcreteTypeDim::ToString() const {
  if (absl::holds_alternative<std::unique_ptr<ParametricExpression>>(value_)) {
    return absl::get<std::unique_ptr<ParametricExpression>>(value_)->ToString();
  }
  return absl::StrCat(absl::get<int64>(value_));
}

bool ConcreteTypeDim::operator==(
    const absl::variant<int64, const ParametricExpression*>& other) const {
  if (absl::holds_alternative<int64>(other)) {
    return absl::holds_alternative<int64>(value_) &&
           absl::get<int64>(value_) == absl::get<int64>(other);
  }
  if (absl::holds_alternative<const ParametricExpression*>(other)) {
    return absl::holds_alternative<std::unique_ptr<ParametricExpression>>(
               value_) &&
           *absl::get<std::unique_ptr<ParametricExpression>>(value_) ==
               *absl::get<const ParametricExpression*>(other);
  }

  return false;
}

bool ConcreteTypeDim::operator==(const ConcreteTypeDim& other) const {
  if (absl::holds_alternative<std::unique_ptr<ParametricExpression>>(value_) &&
      absl::holds_alternative<std::unique_ptr<ParametricExpression>>(
          other.value_)) {
    return *absl::get<std::unique_ptr<ParametricExpression>>(value_) ==
           *absl::get<std::unique_ptr<ParametricExpression>>(other.value_);
  }
  return value_ == other.value_;
}

absl::StatusOr<ConcreteTypeDim> ConcreteTypeDim::Mul(
    const ConcreteTypeDim& rhs) const {
  if (absl::holds_alternative<int64>(value_) &&
      absl::holds_alternative<int64>(rhs.value_)) {
    return ConcreteTypeDim(absl::get<int64>(value_) *
                           absl::get<int64>(rhs.value_));
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot multiply dimensions: %s * %s", ToString(), rhs.ToString()));
}

absl::StatusOr<ConcreteTypeDim> ConcreteTypeDim::Add(
    const ConcreteTypeDim& rhs) const {
  if (absl::holds_alternative<int64>(value_) &&
      absl::holds_alternative<int64>(rhs.value_)) {
    return ConcreteTypeDim(absl::get<int64>(value_) +
                           absl::get<int64>(rhs.value_));
  }
  if (IsParametric() && rhs.IsParametric()) {
    return ConcreteTypeDim(absl::make_unique<ParametricAdd>(
        parametric().Clone(), rhs.parametric().Clone()));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unhandled add: %s + %s", ToString(), rhs.ToString()));
}

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

bool ConcreteType::IsNil() const {
  if (auto* t = dynamic_cast<const TupleType*>(this)) {
    return t->empty();
  }
  return false;
}

std::string BitsType::ToString() const {
  return absl::StrFormat("%cN[%s]", is_signed_ ? 's' : 'u', size_.ToString());
}

std::string BitsType::GetDebugTypeName() const {
  return is_signed_ ? "sbits" : "ubits";
}

std::unique_ptr<BitsType> BitsType::ToUBits() const {
  return absl::make_unique<BitsType>(false, size_.Clone());
}

absl::optional<const ConcreteType*> TupleType::GetMemberTypeByName(
    absl::string_view target) const {
  if (!is_named()) {
    return absl::nullopt;
  }
  for (const NamedMember& m : absl::get<NamedMembers>(members_)) {
    if (m.name == target) {
      return m.type.get();
    }
  }
  return absl::nullopt;
}

bool TupleType::HasNamedMember(absl::string_view target) const {
  if (!is_named()) {
    return false;
  }
  for (const NamedMember& m : absl::get<NamedMembers>(members_)) {
    if (m.name == target) {
      return true;
    }
  }
  return false;
}

std::vector<const ConcreteType*> TupleType::GetUnnamedMembers() const {
  if (absl::holds_alternative<UnnamedMembers>(members_)) {
    std::vector<const ConcreteType*> results;
    for (const auto& m : absl::get<UnnamedMembers>(members_)) {
      results.push_back(m.get());
    }
    return results;
  }
  XLS_CHECK(absl::holds_alternative<NamedMembers>(members_));
  std::vector<const ConcreteType*> results;
  for (const NamedMember& m : absl::get<NamedMembers>(members_)) {
    results.push_back(m.type.get());
  }
  return results;
}

std::string TupleType::ToString() const {
  auto named_member_to_string = [](std::string* out, const NamedMember& m) {
    absl::StrAppendFormat(out, "%s: %s", m.name, m.type->ToString());
  };
  auto unnamed_member_to_string = [](std::string* out,
                                     const std::unique_ptr<ConcreteType>& m) {
    absl::StrAppend(out, m->ToString());
  };

  std::string guts;
  if (absl::holds_alternative<NamedMembers>(members_)) {
    guts = absl::StrJoin(absl::get<NamedMembers>(members_), ", ",
                         named_member_to_string);
  } else {
    guts = absl::StrJoin(absl::get<UnnamedMembers>(members_), ", ",
                         unnamed_member_to_string);
  }
  return absl::StrCat("(", guts, ")");
}

std::vector<ConcreteTypeDim> TupleType::GetAllDims() const {
  std::vector<ConcreteTypeDim> results;
  for (const ConcreteType* t : GetUnnamedMembers()) {
    std::vector<ConcreteTypeDim> t_dims = t->GetAllDims();
    for (auto& dim : t_dims) {
      results.push_back(std::move(dim));
    }
  }
  return results;
}

absl::StatusOr<ConcreteTypeDim> TupleType::GetTotalBitCount() const {
  std::vector<const ConcreteType*> unnamed = GetUnnamedMembers();

  ConcreteTypeDim sum(0);
  for (const ConcreteType* t : unnamed) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim elem_bit_count, t->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(elem_bit_count));
  }

  return sum;
}

/* static */ bool TupleType::MembersEqual(const UnnamedMembers& a,
                                          const UnnamedMembers& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int64 i = 0; i < a.size(); ++i) {
    if (*a[i] != *b[i]) {
      return false;
    }
  }
  return true;
}

/* static */ bool TupleType::MembersEqual(const NamedMembers& a,
                                          const NamedMembers& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int64 i = 0; i < a.size(); ++i) {
    if (a[i].name != b[i].name || *a[i].type != *b[i].type) {
      return false;
    }
  }
  return true;
}

/* static */ bool TupleType::MembersEqual(const Members& a, const Members& b) {
  if (absl::holds_alternative<UnnamedMembers>(a) &&
      absl::holds_alternative<UnnamedMembers>(b)) {
    return MembersEqual(absl::get<UnnamedMembers>(a),
                        absl::get<UnnamedMembers>(b));
  }
  if (absl::holds_alternative<NamedMembers>(a) &&
      absl::holds_alternative<NamedMembers>(b)) {
    return MembersEqual(absl::get<NamedMembers>(a), absl::get<NamedMembers>(b));
  }
  return false;
}

TupleType::Members TupleType::CloneMembers() const {
  if (absl::holds_alternative<UnnamedMembers>(members_)) {
    UnnamedMembers result;
    for (const auto& t : absl::get<UnnamedMembers>(members_)) {
      result.push_back(t->CloneToUnique());
    }
    return result;
  }
  NamedMembers result;
  for (const NamedMember& m : absl::get<NamedMembers>(members_)) {
    result.push_back(NamedMember{m.name, m.type->CloneToUnique()});
  }
  return result;
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

std::string EnumType::ToString() const { return enum_->identifier(); }

std::vector<ConcreteTypeDim> EnumType::GetAllDims() const {
  std::vector<ConcreteTypeDim> result;
  result.push_back(size_.Clone());
  return result;
}

std::vector<const ConcreteType*> FunctionType::GetParams() const {
  std::vector<const ConcreteType*> results;
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
  ConcreteTypeDim sum(0);
  for (const auto& param : params_) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim param_bits, param->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(param_bits));
  }
  return sum;
}

}  // namespace xls::dslx
