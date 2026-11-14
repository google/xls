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

#include "xls/dslx/type_system_v2/parametric_type_annotation_utils.h"

#include <memory>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_utils.h"
#include "xls/dslx/type_system_v2/populate_table_visitor.h"

namespace xls::dslx {

absl::StatusOr<const TypeAnnotation*> GetParametricFreeType(
    const TypeAnnotation* type,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& actual_values,
    InferenceTable& table, ImportData& import_data,
    std::optional<const TypeAnnotation*> real_self_type,
    bool clone_if_no_parametrics) {
  XLS_ASSIGN_OR_RETURN(
      const TypeAnnotation* result,
      SubstituteTypeParametrics(type, actual_values, table, real_self_type,
                                clone_if_no_parametrics));
  if (!clone_if_no_parametrics && result == type) {
    return type;
  }
  std::unique_ptr<PopulateTableVisitor> visitor =
      CreatePopulateTableVisitor(type->owner(), &table, &import_data,
                                 /*typecheck_imported_module=*/nullptr);

  // Replacement can fabricate new indirect annotations, so populate the result,
  // including any borrowed whole-type replacements, before it is reused by
  // TIv2.
  XLS_RETURN_IF_ERROR(visitor->PopulateFromTypeAnnotation(result));

  return result;
}

}  // namespace xls::dslx
