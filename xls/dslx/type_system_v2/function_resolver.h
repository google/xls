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

// The result of resolving the target of a function call. If the `target_object`
// is specified, then it is an instance method being invoked on `target_object`.
// Otherwise, it is a static function which may or may not be a member.

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_FUNCTION_RESOLVER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_FUNCTION_RESOLVER_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/parametric_struct_instantiator.h"
#include "xls/dslx/type_system_v2/trait_deriver.h"
#include "xls/dslx/type_system_v2/type_annotation_resolver.h"

namespace xls::dslx {

struct FunctionAndTargetObject {
  const Function* function = nullptr;
  const std::optional<Expr*> target_object;
  std::optional<const ParametricContext*> target_struct_context;
  std::optional<const TypeAnnotation*> target_object_type;
};

// An object that determines which function is referenced by a callee
// expression.
class FunctionResolver {
 public:
  virtual ~FunctionResolver() = default;

  // Determines what function is being invoked by a `callee` expression.
  virtual absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) = 0;
};

std::unique_ptr<FunctionResolver> CreateFunctionResolver(
    Module& module, ImportData& import_data, InferenceTable& table,
    InferenceTableConverter& converter,
    TypeAnnotationResolver& type_annotation_resolver,
    ParametricStructInstantiator& parametric_struct_instantiator);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_FUNCTION_RESOLVER_H_
