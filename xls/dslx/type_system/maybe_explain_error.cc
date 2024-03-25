// Copyright 2023 The XLS Authors
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

#include "xls/dslx/type_system/maybe_explain_error.h"

#include <string>
#include <string_view>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"

namespace xls::dslx {
namespace {

// To be raised when a type mismatch is encountered.
absl::Status XlsTypeErrorStatus(const Span& span, const Type& lhs,
                                const Type& rhs, std::string_view message) {
  return absl::InvalidArgumentError(
      absl::StrFormat("XlsTypeError: %s %s vs %s: %s", span.ToString(),
                      lhs.ToErrorString(), rhs.ToErrorString(), message));
}

absl::Status MakeTypeError(const TypeMismatchErrorData& data) {
  return XlsTypeErrorStatus(data.error_span, *data.lhs, *data.rhs,
                            data.message);
}

}  // namespace

absl::Status MaybeExplainError(const TypeMismatchErrorData& data) {
  bool lhs_is_unit = data.lhs->IsUnit();
  bool rhs_is_unit = data.rhs->IsUnit();
  VLOG(10) << "lhs is unit: " << lhs_is_unit << " rhs is unit: " << rhs_is_unit;
  bool only_one_side_is_unit = lhs_is_unit ^ rhs_is_unit;
  if (!only_one_side_is_unit) {
    return MakeTypeError(data);
  }

  // If the expression with type unit:
  // * is a name ref
  // * to a def that was defined by a block
  // * and the block has a trailing semicolon
  //
  // ... then put an additional note that the block defining it had a trailing
  // semicolon
  const AstNode* node_yielding_unit =
      lhs_is_unit ? data.lhs_node : data.rhs_node;
  const NameRef* unit_name_ref =
      dynamic_cast<const NameRef*>(node_yielding_unit);
  if (unit_name_ref == nullptr) {
    return MakeTypeError(data);
  }

  VLOG(10) << "unit name reference: " << unit_name_ref->ToString();

  AnyNameDef any_name_def = unit_name_ref->name_def();
  if (std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return MakeTypeError(data);
  }

  const NameDef* name_def = std::get<const NameDef*>(any_name_def);
  const AstNode* definer = name_def->definer();
  const Block* block = dynamic_cast<const Block*>(definer);
  VLOG(10) << absl::StreamFormat("name_def: %s definer: %p block: %p",
                                 name_def->ToString(), definer, block);
  if (block == nullptr || !block->trailing_semi()) {
    return MakeTypeError(data);
  }

  std::string message = absl::StrFormat(
      "%s; note that \"%s\" was defined by a block with a trailing semicolon @ "
      "%s",
      data.message, unit_name_ref->identifier(),
      block->span().limit().ToString());
  return XlsTypeErrorStatus(data.error_span, *data.lhs, *data.rhs, message);
}

}  // namespace xls::dslx
