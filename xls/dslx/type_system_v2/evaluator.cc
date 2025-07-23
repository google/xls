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

#include "xls/dslx/type_system_v2/evaluator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"
#include "xls/dslx/type_system_v2/inference_table_converter.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_system_tracer.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

class EvaluatorImpl : public Evaluator {
 public:
  EvaluatorImpl(InferenceTable& table, Module& module, ImportData& import_data,
                WarningCollector& warning_collector,
                InferenceTableConverter& converter, TypeSystemTracer& tracer)
      : table_(table),
        module_(module),
        import_data_(import_data),
        warning_collector_(warning_collector),
        converter_(converter),
        tracer_(tracer) {}

  absl::StatusOr<bool> EvaluateBoolOrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<bool, const Expr*> value_or_expr) override {
    if (std::holds_alternative<bool>(value_or_expr)) {
      return std::get<bool>(value_or_expr);
    }
    const Expr* expr = std::get<const Expr*>(value_or_expr);
    std::optional<InterpValue> value =
        FastEvaluate(parametric_context, BuiltinType::kBool, expr);

    if (!value.has_value()) {
      XLS_RETURN_IF_ERROR(
          converter_.ConvertSubtree(expr, std::nullopt, parametric_context));

      XLS_ASSIGN_OR_RETURN(
          value,
          Evaluate(ParametricContextScopedExpr(
              parametric_context,
              CreateBoolAnnotation(*expr->owner(), expr->span()), expr)));

      if (expr->kind() == AstNodeKind::kNumber) {
        literal_cache_.emplace(
            std::make_pair(BuiltinType::kBool, expr->ToString()), *value);
      }
    }
    return value->GetBitValueUnsigned();
  }

  absl::StatusOr<int64_t> EvaluateU32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) override {
    return Evaluate32BitIntOrExpr(parametric_context, value_or_expr,
                                  /*is_signed=*/false);
  }

  absl::StatusOr<int64_t> EvaluateS32OrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr) override {
    return Evaluate32BitIntOrExpr(parametric_context, value_or_expr,
                                  /*is_signed=*/true);
  }

  absl::StatusOr<InterpValue> Evaluate(
      const ParametricContextScopedExpr& scoped_expr) override {
    VLOG(7) << "Evaluate: " << scoped_expr.expr()->ToString()
            << " with owner: " << scoped_expr.expr()->owner()->name()
            << " in module: " << module_.name()
            << " in context: " << ToString(scoped_expr.context());

    // Note that `scoped_expr` will not have a `context()` in a case like
    //  fn foo<X: u32>(...) { ... }
    //  fn bar() {
    //    foo<SOME_CONSTANT + 1>(...);
    //  }
    // The only scoped expr there is the expression being passed for `X`, which
    // is in a non-parametric caller and therefore cannot possibly refer to any
    // parametrics.
    XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                         converter_.GetTypeInfo(scoped_expr.expr()->owner(),
                                                scoped_expr.context()));
    return Evaluate(scoped_expr.context(), type_info,
                    scoped_expr.type_annotation(), scoped_expr.expr());
  }

  absl::StatusOr<InterpValue> Evaluate(
      std::optional<const ParametricContext*> parametric_context,
      TypeInfo* type_info, const TypeAnnotation* type_annotation,
      const Expr* expr) override {
    TypeSystemTrace trace = tracer_.TraceEvaluate(parametric_context, expr);

    // This is the type of the parametric binding we are talking about, which is
    // typically a built-in type, but the way we are concretizing it here would
    // support it being a complex type that even refers to other parametrics.
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<Type> type,
        converter_.Concretize(type_annotation, parametric_context,
                              /*needs_conversion_before_eval=*/false));
    if (!type_info->Contains(const_cast<Expr*>(expr))) {
      XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(
          expr, /*function=*/std::nullopt, parametric_context));
    }
    if (type_annotation->owner() == type_info->module()) {
      // Prevent bleed-over from a different module.
      type_info->SetItem(type_annotation, MetaType(type->CloneToUnique()));
    }

    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        ConstexprEvaluator::EvaluateToValue(
            &import_data_, type_info, &warning_collector_,
            table_.GetParametricEnv(parametric_context), expr));
    VLOG(7) << "Evaluation result for: " << expr->ToString()
            << " in context: " << ToString(parametric_context)
            << " value: " << result.ToString();
    return result;
  }

 private:
  absl::StatusOr<int64_t> Evaluate32BitIntOrExpr(
      std::optional<const ParametricContext*> parametric_context,
      std::variant<int64_t, const Expr*> value_or_expr, bool is_signed) {
    if (std::holds_alternative<int64_t>(value_or_expr)) {
      return std::get<int64_t>(value_or_expr);
    }

    const Expr* expr = std::get<const Expr*>(value_or_expr);
    const BuiltinType type = is_signed ? BuiltinType::kS32 : BuiltinType::kU32;
    std::optional<InterpValue> value =
        FastEvaluate(parametric_context, type, expr);

    if (!value.has_value()) {
      XLS_RETURN_IF_ERROR(converter_.ConvertSubtree(
          expr, /*function=*/std::nullopt, parametric_context));

      std::optional<const TypeAnnotation*> type_annotation =
          table_.GetTypeAnnotation(expr);
      if (!type_annotation.has_value()) {
        type_annotation =
            is_signed ? CreateS32Annotation(*expr->owner(), expr->span())
                      : CreateU32Annotation(*expr->owner(), expr->span());
      }
      XLS_ASSIGN_OR_RETURN(
          value, Evaluate(ParametricContextScopedExpr(parametric_context,
                                                      *type_annotation, expr)));

      if (expr->kind() == AstNodeKind::kNumber) {
        literal_cache_.emplace(std::make_pair(type, expr->ToString()), *value);
      }
    }

    int64_t result;
    if (value->IsSigned()) {
      XLS_ASSIGN_OR_RETURN(result, value->GetBitValueSigned());
    } else {
      XLS_ASSIGN_OR_RETURN(result, value->GetBitValueUnsigned());
    }
    return result;
  }

  // Evaluates the given `Expr` if there is a faster way to do so than using
  // `ConstexprEvaluator`.
  std::optional<InterpValue> FastEvaluate(
      std::optional<const ParametricContext*> parametric_context,
      BuiltinType type, const Expr* expr) {
    if (expr->kind() == AstNodeKind::kNumber) {
      // If it's a literal, it may be cached.
      const auto it = literal_cache_.find({type, expr->ToString()});
      if (it != literal_cache_.end()) {
        return it->second;
      }
    } else if (expr->kind() == AstNodeKind::kNameRef &&
               parametric_context.has_value()) {
      // If it's a parametric, we can get it from the context.
      const auto* name_ref = down_cast<const NameRef*>(expr);
      if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
        return (*parametric_context)
            ->GetEnvValue(std::get<const NameDef*>(name_ref->name_def()));
      }
    }
    return std::nullopt;
  }

  InferenceTable& table_;
  Module& module_;
  ImportData& import_data_;
  WarningCollector& warning_collector_;
  InferenceTableConverter& converter_;
  TypeSystemTracer& tracer_;
  absl::flat_hash_map<std::pair<BuiltinType, std::string>, InterpValue>
      literal_cache_;
};

}  // namespace

std::unique_ptr<Evaluator> CreateEvaluator(InferenceTable& table,
                                           Module& module,
                                           ImportData& import_data,
                                           WarningCollector& warning_collector,
                                           InferenceTableConverter& converter,
                                           TypeSystemTracer& tracer) {
  return std::make_unique<EvaluatorImpl>(table, module, import_data,
                                         warning_collector, converter, tracer);
}

}  // namespace xls::dslx
