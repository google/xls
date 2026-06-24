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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_TYPE_ANNOTATION_UTILS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_TYPE_ANNOTATION_UTILS_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"

namespace xls::dslx {

class ImportData;
class InferenceTable;

// Returns `type` with references mapped by `actual_values` substituted by their
// concrete expression/type annotations. This includes whole-type variables and
// aliases, nested type arguments, and `Self` when `real_self_type` is present.
// When `clone_if_no_parametrics` is false, the original pointer may be returned
// if no substitutions are found.
//
// Cloned nodes, including `Self` replacements, belong to the Module owning
// `type`. Whole-type replacements from `actual_values` are borrowed directly
// and retain their original Module ownership, even within a cloned annotation.
// The original and replacement Modules must outlive uses of the result. After
// cloning or replacement, the result is populated into `table` so later TIv2
// stages can resolve its indirect annotation data.
absl::StatusOr<const TypeAnnotation*> GetParametricFreeType(
    const TypeAnnotation* type,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& actual_values,
    InferenceTable& table, ImportData& import_data,
    std::optional<const TypeAnnotation*> real_self_type = std::nullopt,
    bool clone_if_no_parametrics = true);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_TYPE_ANNOTATION_UTILS_H_
