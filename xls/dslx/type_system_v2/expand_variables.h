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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_EXPAND_VARIABLES_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_EXPAND_VARIABLES_H_

#include <optional>
#include <vector>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system_v2/inference_table.h"
namespace xls::dslx {

// A utility that flattens type annotation trees, with expansion of encountered
// type variables, instead of unification of those variables. This is in
// contrast to `ResolveVariableTypeAnnotations`, which converts encountered
// variables to their unifications. The flattening + expansion behavior of this
// visitor is useful for dependency analysis before we are ready to perform
// unification.
std::vector<const TypeAnnotation*> ExpandVariables(
    const TypeAnnotation* member_annotation, const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_EXPAND_VARIABLES_H_
