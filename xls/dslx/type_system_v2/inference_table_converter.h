// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_CONVERTER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_CONVERTER_H_

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// The result of resolving the target of a function call. If the `target_object`
// is specified, then it is an instance method being invoked on `target_object`.
// Otherwise, it is a static function which may or may not be a member.
struct FunctionAndTargetObject {
  const Function* function = nullptr;
  const std::optional<Expr*> target_object;
  std::optional<const ParametricContext*> target_struct_context;
};

class SemanticsAnalysis;

// Class that facilitates the conversion of an `InferenceTable` to
// `TypeInfo`.
class InferenceTableConverter {
 public:
  virtual ~InferenceTableConverter() = default;

  // Converts all type info for the subtree rooted at `node`. `function` is
  // the containing function of the subtree, if any. `parametric_context` is
  // the invocation in whose context the types should be evaluated, if any.
  //
  // When `node` is an actual function argument that is being converted in order
  // to determine a parametric in its own formal type, special behavior is
  // needed, which is enabled by the `filter_param_type_annotations` flag. In
  // such a case, the argument may have one annotation that is
  // `ParamType(function_type, n)`, and since that is the very thing we are
  // really trying to infer, we can't factor it in to the type of the argument
  // value. In all other cases, the flag should be false.
  virtual absl::Status ConvertSubtree(
      const AstNode* node, std::optional<const Function*> function,
      std::optional<const ParametricContext*> parametric_context,
      bool filter_param_type_annotations = false) = 0;

  // Converts the given type annotation to a concrete `Type`, either statically
  // or in the context of a parametric invocation. The
  // `needs_conversion_before_eval` flag indicates if the annotation needs its
  // subtree converted before evaluating parts of it.
  // The `node` where `annotation` is obtained from may be optionally provided,
  // and for certain kinds of node a simplified TypeAnnotation may be cached for
  // it, which is to improve performance for certain use cases, while the
  // node itself does not affect type concretization.
  virtual absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation,
      std::optional<const ParametricContext*> parametric_context,
      bool needs_conversion_before_eval,
      std::optional<const AstNode*> node = std::nullopt) = 0;

  // Determines what function is being invoked by a `callee` expression.
  virtual absl::StatusOr<const FunctionAndTargetObject> ResolveFunction(
      const Expr* callee, std::optional<const Function*> caller_function,
      std::optional<const ParametricContext*> caller_context) = 0;

  // Returns the resulting base type info for the entire conversion.
  virtual TypeInfo* GetBaseTypeInfo() = 0;

  // Returns the appropriate TypeInfo for a node owned by `module` when analyzed
  // in the given `parametric_context`.
  virtual absl::StatusOr<TypeInfo*> GetTypeInfo(
      const Module* module,
      std::optional<const ParametricContext*> parametric_context) = 0;

  virtual SemanticsAnalysis* GetSemanticsAnalysis() = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_INFERENCE_TABLE_CONVERTER_H_
