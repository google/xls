// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_INFERENCE_ERROR_HANDLER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_INFERENCE_ERROR_HANDLER_H_

#include <functional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// The information for each candidate type passed to an error handler.
struct CandidateType {
  const TypeAnnotation* annotation;
  const Type* type;
  TypeInferenceFlag flags;
};

// A hook that allows external logic to resolve an unsupported type unification
// scenario, so that type inference can proceed where it would otherwise error.
// When invoked, the handler receives the competing type annotations for a node,
// and the concretized renditions of each of them.
//
// If the handler can resolve the situation, it should return its own
// unification result, and internally remember either the fact that the node is
// erroneous, or an automatic fix to apply in a later pass.
//
// If the handler has logic for the given situation, but it fails, the handler
// should yield its own error, which will become the result of the type
// inference pass.
//
// If the handler generally does not deal with the given situation, it should
// return an unimplemented error, which will cause the type inference pass to
// act as if the handler were not present.
//
// Example use cases:
// - Translating another language to DSLX. Translating an expr verbatim may fail
//   type inference due to the different rules. A handler that injects casts
//   where needed can be used as a post-processor. In this case, the handler
//   would return what it plans to cast the offending expr to, and actually
//   insert the cast in a later pass over the AST.
// - Enabling type inference to find other potential errors after the first one.
//   A handler that guesses the user's intent can achieve this. In this case,
//   the handler would return the guess but remember that the given node has an
//   error, and flag it to the user later.
using TypeInferenceErrorHandler =
    std::function<absl::StatusOr<const TypeAnnotation*>(
        const absl::Status& error, const AstNode*,
        absl::Span<const CandidateType> candidate_types)>;

// Chains two error handlers together so that if `first` is non-null and
// does not return an unimplemented error, its result is returned; otherwise the
// result of `second` is returned. `second` must be non-null.
inline TypeInferenceErrorHandler ChainTypeInferenceErrorHandlers(
    TypeInferenceErrorHandler first, TypeInferenceErrorHandler second) {
  CHECK(second != nullptr);
  return [first = std::move(first), second = std::move(second)](
             const absl::Status& error, const AstNode* node,
             absl::Span<const CandidateType> candidate_types)
             -> absl::StatusOr<const TypeAnnotation*> {
    if (first) {
      absl::StatusOr<const TypeAnnotation*> first_result =
          first(error, node, candidate_types);
      if (first_result.ok() || !absl::IsUnimplemented(first_result.status())) {
        return first_result;
      }
    }
    return second(error, node, candidate_types);
  };
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_INFERENCE_ERROR_HANDLER_H_
