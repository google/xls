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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_RESOLVER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_RESOLVER_H_

#include <functional>
#include <memory>
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
#include "xls/dslx/type_system_v2/simplified_type_annotation_cache.h"
#include "xls/dslx/type_system_v2/type_annotation_filter.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/type_system_v2/unify_type_annotations.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// An object that wraps type unification logic with resolution of indirect type
// annotations. Indirect type annotations, such as `MemberTypeAnnotation`,
// `ElementTypeAnnotation`, and `ParamTypeAnnotation`, are defined in terms of
// another entity. They need to be resolved to direct annotations, like
// `BuiltinTypeAnnotation`, `ArrayTypeAnnotation`, `TupleTypeAnnotation`, etc.
// before they can be unified. The low-level `UnifyTypeAnnotations` function
// assumes this has already been done.
class TypeAnnotationResolver {
 public:
  // Creates a `TypeAnnotationResolver` with the given dependencies.
  static std::unique_ptr<TypeAnnotationResolver> Create(
      Module& module, InferenceTable& table, const FileTable& file_table,
      UnificationErrorGenerator& error_generator, Evaluator& evaluator,
      ParametricStructInstantiator& parametric_struct_instantiator,
      TypeSystemTracer& tracer, WarningCollector& warning_collector,
      ImportData& import_data,
      SimplifiedTypeAnnotationCache& simplified_type_annotation_cache,
      std::function<absl::Status(std::optional<const ParametricContext*>,
                                 const Invocation*)>
          invocation_converter);

  virtual ~TypeAnnotationResolver() = default;

  // Resolves any indirect annotations for the given node in the inference table
  // associated with this `Resolver`, and then unifies the entire set of direct
  // annotations, producing one annotation.
  virtual absl::StatusOr<std::optional<const TypeAnnotation*>>
  ResolveAndUnifyTypeAnnotationsForNode(
      std::optional<const ParametricContext*> parametric_context,
      const AstNode* node,
      TypeAnnotationFilter filter = TypeAnnotationFilter::None()) = 0;

  // Overload that unifies the type annotations for a given type variable,
  // resolving any indirect ones before invoking unification.
  virtual absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* type_variable, const Span& span,
      TypeAnnotationFilter filter, bool require_bits_like) = 0;

  // Overload that resolves and unifies specific type annotations.
  virtual absl::StatusOr<const TypeAnnotation*> ResolveAndUnifyTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*> annotations, const Span& span,
      TypeAnnotationFilter filter, bool require_bits_like) = 0;

  // Returns `annotation` with any indirect annotations resolved into direct
  // annotations. An indirect annotation is an internally-generated one that
  // depends on the resolved type of another entity. This may be a
  // `TypeVariableTypeAnnotation`, a `MemberTypeAnnotation`, or an
  // `ElementTypeAnnotation`. The original `annotation` is returned if there is
  // nothing to resolve, preserving the ability to identify it as an auto
  // literal annotation.
  //
  // If `accept_predicate` is specified, then it is used to filter annotations
  // for entities referred to by `annotation`. For example, the caller may be
  // trying to solve for the value of an implicit parametric `N` by expanding a
  // `TypeVariableTypeAnnotation` that has 2 associated annotations in the
  // inference table: `u32` and `uN[N]`. In that case, the caller does not want
  // attempted resolution of the `uN[N]` annotation by this function. The
  // predicate is not applied to the input `annotation` itself.
  virtual absl::StatusOr<const TypeAnnotation*> ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation, TypeAnnotationFilter filter) = 0;

  // Overload that deeply resolves all `TypeVariableTypeAnnotation`s within a
  // vector of annotations. If `accept_predicate` is specified, then any
  // annotations not accepted by the predicate are filtered from both
  // `annotations` and the expansions of any encountered type variables.
  virtual absl::Status ResolveIndirectTypeAnnotations(
      std::optional<const ParametricContext*> parametric_context,
      std::vector<const TypeAnnotation*>& annotations,
      TypeAnnotationFilter filter) = 0;

  // Returns a copy of `annotation` with any type aliases resolved, but not
  // other kinds of indirect type annotations. This is suitable for user-written
  // annotations, especially formal types of parametric arguments, and avoids
  // attempting to use the unknown parametrics in that case.
  virtual absl::StatusOr<const TypeAnnotation*> ResolveTypeRefs(
      std::optional<const ParametricContext*> parametric_context,
      const TypeAnnotation* annotation) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TYPE_ANNOTATION_RESOLVER_H_
