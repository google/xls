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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_STRUCT_INSTANTIATOR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_STRUCT_INSTANTIATOR_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

// An interface used to inject parametric struct instantiation logic into
// unification.
class ParametricStructInstantiator {
 public:
  virtual ~ParametricStructInstantiator() = default;

  // Instantiates a parametric struct of the type indicated by `struct_def`.
  // Here, "instantiate" means "make the appropriate parameterization exist with
  // a `ParametricContext` in the `InferenceTable`." Sometimes this is done to
  // handle a struct instance expression (which makes a particular object of the
  // struct exist in DSLX), e.g. `Foo { a: 1, b: 2 }`. Other times it is done to
  // handle a type annotation in DSLX referencing a parametric struct, like
  // `zero!<Foo>()`
  //
  // The `module`, `span`, `parametric_context`, and `instantiator_node` are all
  // from the place which is motivating the instantiation of the struct, i.e.
  // the place consuming the struct declaration, rather than necessarily the
  // place where the declaration lives.
  //
  // The return value is a `TypeAnnotation` referring to the struct with a
  // complete parameterization.
  virtual absl::StatusOr<const TypeAnnotation*> InstantiateParametricStruct(
      Module& module, const Span& span,
      std::optional<const ParametricContext*> parent_context,
      const StructDef& struct_def,
      const std::vector<InterpValue>& explicit_parametrics,
      std::optional<const StructInstanceBase*> instantiator_node) = 0;

  // Converts the `member_type` of some member of the entity referenced by
  // `struct_or_proc_ref` into a form that has any struct parametrics replaced
  // by their values.
  virtual absl::StatusOr<const TypeAnnotation*>
  GetParametricFreeStructMemberType(
      std::optional<const ParametricContext*> struct_context,
      const StructOrProcRef& struct_or_proc_ref,
      const TypeAnnotation* member_type) = 0;

  // Lighter-weight variant of `InstantiateParametricStruct` for situations
  // where inference of the parametrics is unnecessary; the parametrics must be
  // provided in `ref`.
  virtual absl::StatusOr<const ParametricContext*>
  GetOrCreateParametricStructContext(
      std::optional<const ParametricContext*> parent_context,
      const StructOrProcRef& ref, const AstNode* node) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_PARAMETRIC_STRUCT_INSTANTIATOR_H_
