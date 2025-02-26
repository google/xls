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
#include <optional>
#include <variant>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

// An interface for evaluating expressions during type inference.
class Evaluator {
 public:
  virtual ~Evaluator() = default;

  virtual absl::StatusOr<bool> EvaluateBoolOrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<bool, const Expr*> value_or_expr) = 0;

  virtual absl::StatusOr<int64_t> EvaluateU32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) = 0;

  virtual absl::StatusOr<InterpValue> Evaluate(
      const ParametricContextScopedExpr& scoped_expr) = 0;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_EVALUATOR_H_
