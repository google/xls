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

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/evaluator.h"
#include "xls/dslx/type_system_v2/inference_table.h"

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

// An interface used to inject parametric struct instantiation logic into
// unification.
class ParametricStructInstantiator {
 public:
  virtual ~ParametricStructInstantiator() = default;

  virtual absl::StatusOr<const TypeAnnotation*> InstantiateParametricStruct(
      std::optional<const ParametricContext*> parent_context,
      const StructDef& struct_def,
      const std::vector<InterpValue>& explicit_parametrics,
      std::optional<const StructInstance*> instantiator_node) = 0;
};

// An interface used to inject indirect type annotation resolution logic into
// unification.
class IndirectAnnotationResolver {
 public:
  virtual ~IndirectAnnotationResolver() = default;

  virtual absl::Status ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*>& annotations,
      std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
          accept_predicate) = 0;
};

// Comes up with one type annotation reconciling the information in any type
// annotations that have been associated with the given type variable. If the
// information has unreconcilable conflicts, returns an error. The given
// `parametric_context` argument is used as a context for the evaluation of any
// expressions inside the type annotations. If an `accept_predicate` is
// specified, then annotations not accepted by the predicate are ignored.
absl::StatusOr<const TypeAnnotation*> UnifyTypeAnnotations(
    Module& module, InferenceTable& inference_table,
    const FileTable& file_table, UnificationErrorGenerator& error_generator,
    Evaluator& evaluator,
    ParametricStructInstantiator& parametric_struct_instantiator,
    IndirectAnnotationResolver& indirect_annotation_resolver,
    std::optional<const ParametricContext*> parametric_context,
    std::vector<const TypeAnnotation*> annotations, const Span& span,
    std::optional<absl::FunctionRef<bool(const TypeAnnotation*)>>
        accept_predicate);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_UNIFY_TYPE_ANNOTATIONS_H_
