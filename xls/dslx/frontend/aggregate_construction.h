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

#ifndef XLS_DSLX_FRONTEND_AGGREGATE_CONSTRUCTION_H_
#define XLS_DSLX_FRONTEND_AGGREGATE_CONSTRUCTION_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

// Maps each declaration slot to the corresponding source slot. Omitted members
// stay absent; the caller decides whether omission is allowed (e.g. a splat).
using NamedFieldCorrespondence = std::vector<std::optional<int64_t>>;

struct UnknownNamedField {
  int64_t source_index;
};
struct DuplicateNamedField {
  int64_t source_index;
};
using NamedFieldBindingResult =
    std::variant<NamedFieldCorrespondence, UnknownNamedField,
                 DuplicateNamedField>;

// Uses the declaration's canonical name index and reports the first unknown or
// repeated name in source order. Names are borrowed only for this call, so the
// same operation can bind construction operands or pattern members.
NamedFieldBindingResult BindNamedFields(
    const NamedFields& fields, absl::Span<const std::string_view> source_names);

// A resolved lexical construction, independent of evaluated parametric values.
// The original expression owns its operands; this record only borrows them.
// Shape/arity and missing-field policy are checked by the constructing caller.
class BoundConstruction {
 public:
  // A monostate means positional identity, including a unit's empty payload.
  using Correspondence = std::variant<std::monostate, NamedFieldCorrespondence>;

  BoundConstruction(const Expr* expression, AggregateDeclaration declaration,
                    Correspondence correspondence = std::monostate{});

  const Expr* expression() const { return expression_; }
  const AggregateDeclaration& declaration() const { return declaration_; }
  const Correspondence& correspondence() const { return correspondence_; }
  SumVariant::PayloadShape payload_shape() const;
  int64_t member_count() const;

  // Both queries are O(1). An absent slot is distinct from a source operand.
  std::optional<int64_t> GetSourceIndex(int64_t member_index) const;
  std::optional<const Expr*> GetMember(int64_t member_index) const;

 private:
  const Expr* expression_;
  AggregateDeclaration declaration_;
  Correspondence correspondence_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_AGGREGATE_CONSTRUCTION_H_
