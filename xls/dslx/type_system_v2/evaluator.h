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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_EVALUATOR_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_EVALUATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// An interface for evaluating expressions during type inference.
class Evaluator {
 public:
  virtual ~Evaluator() = default;

  // Returns the bool value of `value_or_expr` if it holds a value; otherwise,
  // evaluates it in the given parametric context.
  virtual absl::StatusOr<bool> EvaluateBoolOrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<bool, const Expr*> value_or_expr) = 0;

  // Returns the unsigned 32-bit value of `value_or_expr` if it holds a value;
  // otherwise, evaluates it in the given parametric context. Returns an error
  // if it is not a constexpr, or if the result cannot fit.
  virtual absl::StatusOr<uint32_t> EvaluateU32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) = 0;

  // Returns the signed 32-bit value of `value_or_expr` if it holds a value;
  // otherwise, evaluates it in the given parametric context.  Returns an error
  // if it is not a constexpr, or if the result cannot fit.
  virtual absl::StatusOr<int32_t> EvaluateS32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) = 0;

  // Constexpr-evaluates the given expression, whose dependencies must already
  // be noted as constexpr's in the `TypeInfo` corresponding to the scope for
  // the expression.
  virtual absl::StatusOr<InterpValue> Evaluate(
      const ParametricContextScopedExpr& scoped_expr) = 0;

  // Variant that uses a specific `TypeInfo`. Use this directly when there is a
  // need to target a temporary `TypeInfo` object, e.g. for `StructInstance`
  // parametric values. When populating a real output `TypeInfo` object, prefer
  // the variant that takes an `ParametricContextScopedExpr`.
  virtual absl::StatusOr<InterpValue> Evaluate(
      std::optional<const ParametricContext*> parametric_context,
      TypeInfo* type_info, const TypeAnnotation* type_annotation,
      const Expr* expr) = 0;
};

// Creates an evaluator bound to the given dependencies.
std::unique_ptr<Evaluator> CreateEvaluator(InferenceTable& table,
                                           Module& module,
                                           ImportData& import_data,
                                           WarningCollector& warning_collector,
                                           InferenceTableConverter& converter,
                                           TypeSystemTracer& tracer);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_EVALUATOR_H_
