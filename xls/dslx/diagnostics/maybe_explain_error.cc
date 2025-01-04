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

#include "xls/dslx/diagnostics/maybe_explain_error.h"

#include <string>
#include <string_view>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/diagnostics/format_type_mismatch.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_mismatch_error_data.h"

namespace xls::dslx {
namespace {

// To be raised when a type mismatch is encountered.
absl::Status XlsTypeErrorStatus(const Span& span, const Type& lhs,
                                const Type& rhs, std::string_view message,
                                const FileTable& file_table) {
  if (lhs.IsAggregate() || rhs.IsAggregate()) {
    XLS_ASSIGN_OR_RETURN(std::string type_diff,
                         FormatTypeMismatch(lhs, rhs, file_table));
    return absl::InvalidArgumentError(
        absl::StrFormat("XlsTypeError: %s %s\n"
                        "%s",
                        span.ToString(file_table), message, type_diff));
  }
  std::string lhs_str = lhs.ToErrorString();
  std::string rhs_str = rhs.ToErrorString();
  return absl::InvalidArgumentError(
      absl::StrFormat("XlsTypeError: %s %s vs %s: %s",
                      span.ToString(file_table), lhs_str, rhs_str, message));
}

// Creates an XlsTypeErrorStatus using the data within the type mismatch struct.
absl::Status MakeTypeError(const TypeMismatchErrorData& data,
                           const FileTable& file_table) {
  return XlsTypeErrorStatus(data.error_span, *data.lhs, *data.rhs, data.message,
                            file_table);
}

bool IsBlockWithTrailingSemi(const AstNode* node) {
  if (auto* block = dynamic_cast<const StatementBlock*>(node);
      block != nullptr) {
    return block->trailing_semi();
  }
  return false;
}

absl::Status ExplainIfBlockWithTrailingSemi(const TypeMismatchErrorData& data,
                                            bool lhs_is_unit, bool rhs_is_unit,
                                            const FileTable& file_table) {
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
    return MakeTypeError(data, file_table);
  }

  VLOG(10) << "unit name reference: " << unit_name_ref->ToString();

  AnyNameDef any_name_def = unit_name_ref->name_def();
  if (std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return MakeTypeError(data, file_table);
  }

  const NameDef* name_def = std::get<const NameDef*>(any_name_def);
  const AstNode* definer = name_def->definer();
  if (!IsBlockWithTrailingSemi(definer)) {
    return MakeTypeError(data, file_table);
  }

  const auto* block = dynamic_cast<const StatementBlock*>(definer);
  VLOG(10) << absl::StreamFormat("name_def: %s definer: %p block: %p",
                                 name_def->ToString(), definer, block);
  std::string message = absl::StrFormat(
      "%s; note that \"%s\" was defined by a block with a trailing semicolon @ "
      "%s",
      data.message, unit_name_ref->identifier(),
      block->span().limit().ToString(file_table));
  return XlsTypeErrorStatus(data.error_span, *data.lhs, *data.rhs, message,
                            file_table);
}

absl::Status ExplainIfConditionalBlockWithTrailingSemi(
    const TypeMismatchErrorData& data, const Conditional* conditional,
    bool lhs_is_unit, bool rhs_is_unit, const FileTable& file_table) {
  const StatementBlock* culprit = nullptr;
  if (lhs_is_unit && IsBlockWithTrailingSemi(data.lhs_node)) {
    culprit = dynamic_cast<const StatementBlock*>(data.lhs_node);
  } else if (rhs_is_unit && IsBlockWithTrailingSemi(data.rhs_node)) {
    culprit = dynamic_cast<const StatementBlock*>(data.rhs_node);
  }

  if (culprit == nullptr) {
    return MakeTypeError(data, file_table);
  }

  std::string message = absl::StrFormat(
      "%s; note that conditional block @ %s had a trailing semicolon",
      data.message, culprit->span().ToString(file_table));
  return XlsTypeErrorStatus(data.error_span, *data.lhs, *data.rhs, message,
                            file_table);
}

}  // namespace

absl::Status MaybeExplainError(const TypeMismatchErrorData& data,
                               const FileTable& file_table) {
  VLOG(5) << absl::StreamFormat(
      "MaybeExplainError; lhs: `%s` lhs_type: `%s` rhs: `%s` rhs_type: `%s`",
      (data.lhs_node != nullptr ? data.lhs_node->ToString() : "null"),
      (data.lhs != nullptr ? data.lhs->ToString() : "null"),
      (data.rhs_node != nullptr ? data.rhs_node->ToString() : "null"),
      (data.rhs != nullptr ? data.rhs->ToString() : "null"));

  bool lhs_is_unit = data.lhs->IsUnit();
  bool rhs_is_unit = data.rhs->IsUnit();
  VLOG(10) << "lhs is unit: " << lhs_is_unit << " rhs is unit: " << rhs_is_unit;
  bool only_one_side_is_unit = lhs_is_unit ^ rhs_is_unit;
  if (!only_one_side_is_unit) {
    return MakeTypeError(data, file_table);
  }

  // Returns whether `node` is a StatementBlock within a Conditional.
  auto is_block_in_conditional = [](const AstNode* node) -> bool {
    auto* block = dynamic_cast<const StatementBlock*>(node);
    if (block == nullptr || block->GetEnclosing() == nullptr) {
      return false;
    }
    return dynamic_cast<const Conditional*>(block->GetEnclosing()) != nullptr;
  };
  // Gets the conditional that encloses the given node.
  auto get_conditional = [&](const AstNode* node) -> const Conditional* {
    DCHECK(is_block_in_conditional(node));
    if (auto* conditional =
            dynamic_cast<const Conditional*>(node->GetEnclosing());
        conditional != nullptr) {
      return conditional;
    }
    return nullptr;
  };

  if (is_block_in_conditional(data.lhs_node) &&
      is_block_in_conditional(data.rhs_node)) {
    if (auto* conditional = get_conditional(data.lhs_node);
        conditional == get_conditional(data.rhs_node)) {
      return ExplainIfConditionalBlockWithTrailingSemi(
          data, conditional, lhs_is_unit, rhs_is_unit, file_table);
    }
  }

  return ExplainIfBlockWithTrailingSemi(data, lhs_is_unit, rhs_is_unit,
                                        file_table);
}

}  // namespace xls::dslx
