// Copyright 2026 The XLS Authors
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

#include "xls/dslx/frontend/aggregate_construction.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

NamedFieldBindingResult BindNamedFields(
    const NamedFields& fields,
    absl::Span<const std::string_view> source_names) {
  NamedFieldCorrespondence correspondence(fields.size());
  for (int64_t i = 0; i < source_names.size(); ++i) {
    std::optional<int64_t> slot = fields.GetMemberIndex(source_names[i]);
    if (!slot.has_value()) {
      return UnknownNamedField{i};
    } else if (correspondence[*slot].has_value()) {
      return DuplicateNamedField{i};
    } else {
      correspondence[*slot] = i;
    }
  }
  return correspondence;
}

BoundConstruction::BoundConstruction(const Expr* expression,
                                     AggregateDeclaration declaration,
                                     Correspondence correspondence)
    : expression_(expression),
      declaration_(declaration),
      correspondence_(std::move(correspondence)) {
  if (payload_shape() == SumVariant::PayloadShape::kStruct) {
    CHECK(std::holds_alternative<NamedFieldCorrespondence>(correspondence_));
    CHECK_EQ(std::get<NamedFieldCorrespondence>(correspondence_).size(),
             member_count());
  } else {
    CHECK(std::holds_alternative<std::monostate>(correspondence_));
  }
}

SumVariant::PayloadShape BoundConstruction::payload_shape() const {
  if (const auto* variant = std::get_if<const SumVariant*>(&declaration_)) {
    return (*variant)->payload_shape();
  } else {
    return SumVariant::PayloadShape::kStruct;
  }
}

int64_t BoundConstruction::member_count() const {
  if (const auto* variant = std::get_if<const SumVariant*>(&declaration_)) {
    return (*variant)->payload_member_count();
  } else {
    return std::get<const StructDefBase*>(declaration_)->size();
  }
}

std::optional<int64_t> BoundConstruction::GetSourceIndex(
    int64_t member_index) const {
  CHECK_GE(member_index, 0);
  CHECK_LT(member_index, member_count());
  if (const auto* named =
          std::get_if<NamedFieldCorrespondence>(&correspondence_)) {
    return (*named)[member_index];
  } else {
    return member_index;
  }
}

std::optional<const Expr*> BoundConstruction::GetMember(
    int64_t member_index) const {
  const std::optional<int64_t> source_index = GetSourceIndex(member_index);
  if (!source_index.has_value()) {
    return std::nullopt;
  } else if (payload_shape() == SumVariant::PayloadShape::kStruct) {
    return absl::down_cast<const StructInstanceBase*>(expression_)
        ->members()[*source_index]
        .second;
  } else {
    return absl::down_cast<const Invocation*>(expression_)
        ->args()[*source_index];
  }
}

}  // namespace xls::dslx
