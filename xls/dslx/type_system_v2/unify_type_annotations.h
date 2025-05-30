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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_UNIFY_TYPE_ANNOTATIONS_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_UNIFY_TYPE_ANNOTATIONS_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"

namespace xls::dslx {

// An interface used to inject error status generation logic into unification.
class UnificationErrorGenerator {
 public:
  virtual ~UnificationErrorGenerator() = default;

  virtual absl::Status TypeMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* a, const TypeAnnotation* b) = 0;

  virtual absl::Status BitCountMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* a, const TypeAnnotation* b) = 0;

  virtual absl::Status SignednessMismatchError(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* a, const TypeAnnotation* b) = 0;
};

// Comes up with one type annotation reconciling the information in any type
// annotations that have been associated with the given type variable. If the
// information has unreconcilable conflicts, returns an error. The given
// `parametric_context` argument is used as a context for the evaluation of any
// expressions inside the type annotations. If a `filter` is  then annotations
// not accepted by the filter are ignored.
absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
    Module& module, InferenceTable& inference_table,
    const FileTable& file_table, UnificationErrorGenerator& error_generator,
    Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    std::optional<const ParametricContext*> parametric_context,
    std::vector<const TypeAnnotation*> annotations, const Span& span,
    const ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_UNIFY_TYPE_ANNOTATIONS_H_
