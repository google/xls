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

#include "xls/dslx/type_system_v2/decorate_error.h"

#include <optional>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_inference_error_handler.h"

namespace xls::dslx {
namespace {

std::optional<CandidateType> FindTypeWithFlag(
    absl::Span<const CandidateType> candidates, TypeInferenceFlag flag) {
  for (const CandidateType& candidate : candidates) {
    if (candidate.flags.HasFlag(flag)) {
      return candidate;
    }
  }
  return std::nullopt;
}

std::optional<CandidateType> FindEmptyTuple(
    absl::Span<const CandidateType> candidates) {
  for (const CandidateType& candidate : candidates) {
    if (candidate.annotation->IsAnnotation<TupleTypeAnnotation>() &&
        absl::down_cast<const TupleTypeAnnotation*>(candidate.annotation)
            ->empty()) {
      return candidate;
    }
  }
  return std::nullopt;
}

Span GetBlockReturnExprSpan(const StatementBlock& block) {
  return block.statements().empty() ? *block.GetSpan()
                                    : *block.statements().back()->GetSpan();
}

// Returns the `ParametricBinding` whose `expr` is or contains the given `node`.
// The result is `nullopt` if `node` is not a binding expr or a part of one.
std::optional<const ParametricBinding*> GetParametricBindingWithContainingExpr(
    const AstNode* node) {
  const AstNode* parent = node->parent();
  while (parent != nullptr &&
         parent->kind() != AstNodeKind::kParametricBinding) {
    node = parent;
    parent = parent->parent();
  }

  if (parent != nullptr) {
    const auto* binding = absl::down_cast<const ParametricBinding*>(parent);
    if (binding->expr() == node) {
      return binding;
    }
  }

  return std::nullopt;
}

}  // namespace

absl::StatusOr<const TypeAnnotation*> DecorateError(
    const absl::Status& error, const AstNode* node,
    absl::Span<const CandidateType> candidate_types) {
  const FileTable& file_table = *node->owner()->file_table();
  std::optional<CandidateType> formal_return_type =
      FindTypeWithFlag(candidate_types, TypeInferenceFlag::kFormalReturnType);

  // Semicolon on the last statement of a function that is supposed to return
  // something.
  if (const auto* block = dynamic_cast<const StatementBlock*>(node);
      block != nullptr && block->parent() != nullptr &&
      block->parent()->kind() == AstNodeKind::kFunction &&
      block->trailing_semi()) {
    std::optional<CandidateType> empty_tuple = FindEmptyTuple(candidate_types);
    if (formal_return_type.has_value() && empty_tuple.has_value()) {
      return TypeInferenceErrorStatus(
          GetBlockReturnExprSpan(*block), nullptr,
          absl::Substitute(
              "$0. Did you intend to add a trailing semicolon to the last "
              "expression in the function body? If the last expression is "
              "terminated with a semicolon, it is discarded, and the function "
              "implicitly returns ().",
              error.message()),
          file_table);
    }
  }

  // Function body returning a different type than the declared return type (not
  // just via an unexpected semicolon).
  if (formal_return_type.has_value()) {
    std::optional<const Function*> function = GetContainingFunction(node);
    if (function.has_value()) {
      return TypeInferenceErrorStatus(
          *node->GetSpan(), nullptr,
          absl::Substitute(
              "$0. The body of function `$1` does not actually return the "
              "function's declared return type, which is `$2`",
              error.message(), (*function)->identifier(),
              formal_return_type->annotation->ToString()),
          file_table);
    }
  }

  // Parametric declaration like <A: u32 = {s64: -1}>
  std::optional<CandidateType> formal_parametric_type = FindTypeWithFlag(
      candidate_types, TypeInferenceFlag::kFormalParametricType);
  if (formal_parametric_type.has_value()) {
    std::optional<const ParametricBinding*> binding =
        GetParametricBindingWithContainingExpr(node);
    if (binding.has_value()) {
      return TypeInferenceErrorStatus(
          *node->GetSpan(), nullptr,
          absl::Substitute("$0. The default expression for the parametric `$1` "
                           "must have the same type as the parametric.",
                           error.message(), (*binding)->identifier()),
          file_table);
    }
  }

  // TODO: https://github.com/google/xls/issues/3929 - Add handling for more
  // scenarios where v1 had a better error.

  return absl::UnimplementedError("This error does not need to be decorated.");
}

}  // namespace xls::dslx
