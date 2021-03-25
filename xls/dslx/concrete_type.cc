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

#include <cstdint>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls::dslx {

std::unique_ptr<ConcreteType> ConcreteType::MakeNil() {
  return absl::make_unique<TupleType>(TupleType::UnnamedMembers{});
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcreteType::FromInterpValue(
    const InterpValue& value) {
  if (value.tag() == InterpValueTag::kUBits ||
      value.tag() == InterpValueTag::kSBits) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
    return absl::make_unique<BitsType>(/*is_signed*/ value.IsSigned(),
                                       /*size=*/bit_count);
  } else if (value.tag() == InterpValueTag::kArray) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         value.GetValues());
    if (elements->empty()) {
      return absl::InvalidArgumentError(
          "Cannot get the ConcreteType of a 0-element array.");
    }
    XLS_ASSIGN_OR_RETURN(auto element_type, FromInterpValue(elements->at(0)));
    XLS_ASSIGN_OR_RETURN(int64_t size, value.GetLength());
    XLS_ASSIGN_OR_RETURN(auto dim, ConcreteTypeDim::Create(size));
    return absl::make_unique<ArrayType>(std::move(element_type), dim);
  } else if (value.tag() == InterpValueTag::kTuple) {
    XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                         value.GetValues());
    if (elements->empty()) {
      return absl::make_unique<TupleType>(
          std::vector<std::unique_ptr<ConcreteType>>());
    }
    std::vector<std::unique_ptr<ConcreteType>> members;
    members.reserve(elements->size());
    for (const auto& element : *elements) {
      XLS_ASSIGN_OR_RETURN(auto member, FromInterpValue(element));
      members.push_back(std::move(member));
    }

    return absl::make_unique<TupleType>(std::move(members));
  } else {
    return absl::InvalidArgumentError(
        "Only bits, array, and tuple types can be converted into concrete.");
  }
}

ConcreteTypeDim::ConcreteTypeDim(const ConcreteTypeDim& other)
    : value_(std::move(other.Clone().value_)) {}

ConcreteTypeDim ConcreteTypeDim::Clone() const {
  if (absl::holds_alternative<int64_t>(value_)) {
    return ConcreteTypeDim(absl::get<int64_t>(value_));
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
  return absl::StrCat(absl::get<int64_t>(value_));
}

std::string ConcreteTypeDim::ToRepr() const {
  std::string guts;
  if (absl::holds_alternative<int64_t>(value_)) {
    guts = absl::StrCat(absl::get<int64_t>(value_));
  } else {
    guts = absl::get<OwnedParametric>(value_)->ToRepr();
  }
  return absl::StrFormat("ConcreteTypeDim(%s)", guts);
}

bool ConcreteTypeDim::operator==(
    const absl::variant<int64_t, InterpValue, const ParametricExpression*>&
        other) const {
  if (absl::holds_alternative<int64_t>(other)) {
    if (absl::holds_alternative<int64_t>(value_)) {
      return absl::get<int64_t>(value_) == absl::get<int64_t>(other);
    }
    return false;
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
  if (absl::holds_alternative<int64_t>(value_) &&
      absl::holds_alternative<int64_t>(rhs.value_)) {
    return ConcreteTypeDim(absl::get<int64_t>(value_) *
                           absl::get<int64_t>(rhs.value_));
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Cannot multiply dimensions: %s * %s", ToString(), rhs.ToString()));
}

absl::StatusOr<ConcreteTypeDim> ConcreteTypeDim::Add(
    const ConcreteTypeDim& rhs) const {
  if (absl::holds_alternative<int64_t>(value_) &&
      absl::holds_alternative<int64_t>(rhs.value_)) {
    return ConcreteTypeDim(absl::get<int64_t>(value_) +
                           absl::get<int64_t>(rhs.value_));
  }
  if (IsParametric() && rhs.IsParametric()) {
    return ConcreteTypeDim(absl::make_unique<ParametricAdd>(
        parametric().Clone(), rhs.parametric().Clone()));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unhandled add: %s + %s", ToString(), rhs.ToString()));
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

bool ConcreteType::IsNil() const {
  if (auto* t = dynamic_cast<const TupleType*>(this)) {
    return t->empty();
  }
  return false;
}

bool ConcreteType::IsToken() const {
  return dynamic_cast<const TokenType*>(this) != nullptr;
}

// -- TokenType

TokenType::~TokenType() = default;

// -- BitsType

absl::StatusOr<std::unique_ptr<ConcreteType>> BitsType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size, f(size_));
  return absl::make_unique<BitsType>(is_signed_, new_size);
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

// -- TupleType

TupleType::TupleType(Members members, StructDef* struct_def)
    : members_(std::move(members)), struct_def_(struct_def) {
  // TODO(leary): 2020-12-11 Reintroduce this when we're not plumbing types
  // through strings for Python interop.
  // XLS_CHECK_EQ(struct_def_ == nullptr, !is_named());
}

bool TupleType::operator==(const ConcreteType& other) const {
  if (auto* t = dynamic_cast<const TupleType*>(&other)) {
    return MembersEqual(members_, t->members_);
  }
  return false;
}

bool TupleType::HasEnum() const {
  for (const ConcreteType* t : GetUnnamedMembers()) {
    if (t->HasEnum()) {
      return true;
    }
  }
  return false;
}

bool TupleType::empty() const {
  if (absl::holds_alternative<NamedMembers>(members_)) {
    return absl::get<NamedMembers>(members_).empty();
  }
  return absl::get<UnnamedMembers>(members_).empty();
}
int64_t TupleType::size() const {
  if (absl::holds_alternative<NamedMembers>(members_)) {
    return absl::get<NamedMembers>(members_).size();
  }
  return absl::get<UnnamedMembers>(members_).size();
}

absl::StatusOr<std::vector<std::string>> TupleType::GetMemberNames() const {
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

absl::StatusOr<int64_t> TupleType::GetMemberIndex(
    absl::string_view name) const {
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> names, GetMemberNames());
  auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Name not present in tuple type %s: %s", ToString(), name));
  }
  return std::distance(names.begin(), it);
}

bool TupleType::CompatibleWith(const TupleType& other) const {
  std::vector<const ConcreteType*> self_members = GetUnnamedMembers();
  std::vector<const ConcreteType*> other_members = GetUnnamedMembers();
  if (self_members.size() != other_members.size()) {
    return false;
  }
  for (int64_t i = 0; i < self_members.size(); ++i) {
    if (!self_members[i]->CompatibleWith(*other_members[i])) {
      return false;
    }
  }
  // Same member count and all compatible members.
  return true;
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

absl::StatusOr<std::unique_ptr<ConcreteType>> TupleType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  if (absl::holds_alternative<NamedMembers>(members_)) {
    std::vector<NamedMember> new_members;
    for (const NamedMember& member : absl::get<NamedMembers>(members_)) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> mapped,
                           member.type->MapSize(f));
      new_members.push_back(NamedMember{member.name, std::move(mapped)});
    }
    return absl::make_unique<TupleType>(std::move(new_members), struct_def_);
  }

  std::vector<std::unique_ptr<ConcreteType>> new_members;
  for (const auto& member_type : absl::get<UnnamedMembers>(members_)) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> mapped,
                         member_type->MapSize(f));
    new_members.push_back(std::move(mapped));
  }
  return absl::make_unique<TupleType>(std::move(new_members), struct_def_);
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

  XLS_ASSIGN_OR_RETURN(auto sum, ConcreteTypeDim::Create(0));
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
  for (int64_t i = 0; i < a.size(); ++i) {
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
  for (int64_t i = 0; i < a.size(); ++i) {
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

// -- ArrayType

absl::StatusOr<std::unique_ptr<ConcreteType>> ArrayType::MapSize(
    const std::function<absl::StatusOr<ConcreteTypeDim>(ConcreteTypeDim)>& f)
    const {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> new_element_type,
                       element_type_->MapSize(f));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_size, f(size_));
  return absl::make_unique<ArrayType>(std::move(new_element_type),
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
  return absl::make_unique<EnumType>(enum_def_, std::move(new_size));
}

std::string EnumType::ToString() const { return enum_def_->identifier(); }

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
  return absl::make_unique<FunctionType>(std::move(new_params),
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
  XLS_ASSIGN_OR_RETURN(auto sum, ConcreteTypeDim::Create(0));
  for (const auto& param : params_) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim param_bits, param->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(sum, sum.Add(param_bits));
  }
  return sum;
}

absl::StatusOr<bool> IsSigned(const ConcreteType& c) {
  if (auto* bits = dynamic_cast<const BitsType*>(&c)) {
    return bits->is_signed();
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&c)) {
    absl::optional<bool> signedness = enum_type->signedness();
    if (!signedness.has_value()) {
      return absl::InvalidArgumentError(
          "Signedness not present for EnumType: " + c.ToString());
    }
    return signedness.value();
  }
  return absl::InvalidArgumentError(
      "Cannot determined signedness; type is neither enum nor bits.");
}

// -- ConcreteTypeFromString

static absl::StatusOr<ConcreteTypeDim> ConsumeConcreteTypeDim(
    absl::string_view* s) {
  int64_t size;
  if (RE2::Consume(s, R"((\d+))", &size)) {
    return ConcreteTypeDim::Create(size);
  }
  std::string symbol;
  if (RE2::Consume(s, R"((\w+))", &symbol)) {
    return ConcreteTypeDim::Create(
        absl::make_unique<ParametricSymbol>(std::move(symbol), Span::Fake()));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Could not parse dimension from \"%s\"", *s));
}

static absl::StatusOr<std::unique_ptr<ConcreteType>> ConsumeArraySuffix(
    absl::string_view* s, std::unique_ptr<ConcreteType> element_type) {
  if (absl::ConsumePrefix(s, "[")) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim size, ConsumeConcreteTypeDim(s));
    if (!absl::ConsumePrefix(s, "]")) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Could not parse ']' after dimension in \"%s\"", *s));
    }
    return absl::make_unique<ArrayType>(std::move(element_type),
                                        std::move(size));
  }
  return std::move(element_type);
}

static absl::StatusOr<std::unique_ptr<ConcreteType>> ConsumeConcreteType(
    absl::string_view* s) {
  absl::string_view orig = *s;
  char signedness;
  if (RE2::Consume(s, R"(([us])N\[)", &signedness)) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim size, ConsumeConcreteTypeDim(s));
    if (!absl::ConsumePrefix(s, "]")) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Could not parse ']' after dimension in \"%s\"", *s));
    }
    std::unique_ptr<ConcreteType> t =
        absl::make_unique<BitsType>(signedness == 's', std::move(size));
    return ConsumeArraySuffix(s, std::move(t));
  }
  if (absl::ConsumePrefix(s, "(")) {
    std::vector<std::string> names;
    std::vector<std::unique_ptr<ConcreteType>> members;
    while (true) {
      if (absl::ConsumePrefix(s, ")")) {
        std::unique_ptr<ConcreteType> t;
        if (names.empty()) {
          t = absl::make_unique<TupleType>(std::move(members));
        } else {
          if (names.size() != members.size()) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Differing numbers of names and members seen "
                                "in tuple type: \"%s\"",
                                orig));
          }
          TupleType::NamedMembers named;
          for (int64_t i = 0; i < names.size(); ++i) {
            named.push_back(TupleType::NamedMember{std::move(names[i]),
                                                   std::move(members[i])});
          }
          t = absl::make_unique<TupleType>(std::move(named));
        }
        return ConsumeArraySuffix(s, std::move(t));
      }
      if (!members.empty()) {
        if (!absl::ConsumePrefix(s, ", ")) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Expected ',' between tuple members; saw: '%s' "
                              "after %d members in original \"%s\"",
                              *s, members.size(), orig));
        }
      }

      // We don't know if it's a named tuple or not, so try to parse a type and
      // roll back if not.
      absl::string_view snapshot = *s;
      absl::StatusOr<std::unique_ptr<ConcreteType>> t = ConsumeConcreteType(s);
      if (t.ok()) {
        members.push_back(std::move(t.value()));
      } else {
        *s = snapshot;
        std::string name;
        if (!RE2::Consume(s, "(\\w+): ", &name)) {
          return t.status();
        }
        names.push_back(name);
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> t,
                             ConsumeConcreteType(s));
        members.push_back(std::move(t));
      }
    }
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unrecognized concrete type: \"%s\"", *s));
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcreteTypeFromString(
    absl::string_view* s) {
  return ConsumeConcreteType(s);
}

}  // namespace xls::dslx
