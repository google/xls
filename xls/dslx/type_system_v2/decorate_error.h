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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_DECORATE_ERROR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_DECORATE_ERROR_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/type_system_v2/type_inference_error_handler.h"

namespace xls::dslx {

// Generates clearer errors for failed type unification in specific scenarios.
// This is used internally as a `TypeInferenceErrorHandler` by
// `TypecheckModuleV2` and should not be called directly.
absl::StatusOr<const TypeAnnotation*> DecorateError(
    const absl::Status& error, const AstNode* node,
    absl::Span<const CandidateType> candidate_types);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_DECORATE_ERROR_H_
