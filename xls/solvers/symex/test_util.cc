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

#include "xls/solvers/symex/test_util.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/z3_ir_translator.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {

bool AreMutuallyExclusive(Z3_context ctx, Z3_ast cond1, Z3_ast cond2) {
  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, cond1);
  Z3_solver_assert(ctx, solver, cond2);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

bool IsExhaustiveCoverage(Z3_context ctx,
                          absl::Span<const SymbolicPath> paths) {
  if (paths.empty()) {
    return false;
  }
  std::vector<Z3_ast> conds;
  conds.reserve(paths.size());
  for (const SymbolicPath& path : paths) {
    conds.push_back(path.path_condition);
  }
  Z3_ast combined = Z3_mk_or(ctx, conds.size(), conds.data());
  Z3_ast not_combined = Z3_mk_not(ctx, combined);

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, not_combined);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

bool VerifyCompletePartition(Z3_context ctx,
                             absl::Span<const SymbolicPath> paths) {
  for (size_t i = 0; i < paths.size(); ++i) {
    for (size_t j = i + 1; j < paths.size(); ++j) {
      if (!AreMutuallyExclusive(ctx, paths[i].path_condition,
                                paths[j].path_condition)) {
        return false;
      }
    }
  }
  return IsExhaustiveCoverage(ctx, paths);
}

absl::StatusOr<Value> InterpretPath(Function* fn, const SymbolicPath& path) {
  return DropInterpreterEvents(InterpretFunction(fn, path.input_values()));
}

Z3_ast CombinePathsToIte(Z3_context ctx, absl::Span<const SymbolicPath> paths) {
  if (paths.empty()) {
    return nullptr;
  }
  if (paths.size() == 1) {
    return paths[0].return_value;
  }
  // Start with the last path return value as the fallback arm.
  Z3_ast ite_tree = paths.back().return_value;
  for (int64_t i = static_cast<int64_t>(paths.size()) - 2; i >= 0; --i) {
    ite_tree = Z3_mk_ite(ctx, paths[i].path_condition, paths[i].return_value,
                         ite_tree);
  }
  return ite_tree;
}

bool CheckSmtEncodingEquivalence(Z3_context ctx, Function* fn,
                                 absl::Span<const SymbolicPath> paths) {
  if (fn == nullptr || paths.empty()) {
    return false;
  }
  auto translator_status =
      solvers::z3::IrTranslator::CreateAndTranslate(ctx, fn->return_value());
  if (!translator_status.ok()) {
    return false;
  }
  auto translator = std::move(translator_status).value();
  Z3_ast mono_return = translator->GetTranslation(fn->return_value());

  Z3_ast combined_ite = CombinePathsToIte(ctx, paths);
  if (combined_ite == nullptr) {
    return false;
  }

  // Prove that combined_ite == mono_return for all inputs (i.e. not(eq) is
  // UNSAT).
  Z3_ast eq = Z3_mk_eq(ctx, combined_ite, mono_return);
  Z3_ast neq_ast = Z3_mk_not(ctx, eq);

  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_solver_assert(ctx, solver, neq_ast);
  Z3_lbool result = Z3_solver_check(ctx, solver);
  Z3_solver_dec_ref(ctx, solver);
  return result == Z3_L_FALSE;
}

absl::Status CheckFormalProperties(Z3_context ctx, Function* func,
                                   absl::Span<const SymbolicPath> paths,
                                   const FormalCheckConfig& config) {
  if (config.expected_paths.has_value() &&
      static_cast<int64_t>(paths.size()) != *config.expected_paths) {
    return absl::InternalError(absl::StrFormat(
        "Expected %d paths, got %d", *config.expected_paths, paths.size()));
  }

  if (config.check_mutual_exclusivity) {
    for (size_t i = 0; i < paths.size(); ++i) {
      for (size_t j = i + 1; j < paths.size(); ++j) {
        if (!AreMutuallyExclusive(ctx, paths[i].path_condition,
                                  paths[j].path_condition)) {
          return absl::InternalError(absl::StrFormat(
              "Paths %d and %d are not mutually exclusive", i, j));
        }
      }
    }
  }

  if (config.check_exhaustiveness) {
    if (!IsExhaustiveCoverage(ctx, paths)) {
      return absl::InternalError(
          "Paths do not collectively cover the input domain");
    }
  }

  if (config.check_smt_ite_equivalence) {
    if (!CheckSmtEncodingEquivalence(ctx, func, paths)) {
      return absl::InternalError(
          "Combined ITE path tree is not logically equivalent to monolithic "
          "IrTranslator model");
    }
  }

  if (config.check_witness_interpretation || config.oracle != nullptr) {
    for (size_t i = 0; i < paths.size(); ++i) {
      const SymbolicPath& path = paths[i];
      if (path.generated_test.size() != func->params().size()) {
        return absl::InternalError(
            absl::StrFormat("Path %d has %d witness params, expected %d", i,
                            path.generated_test.size(), func->params().size()));
      }
      XLS_ASSIGN_OR_RETURN(Value result, InterpretPath(func, path));
      if (!result.IsBits() && !result.IsTuple() && !result.IsArray()) {
        return absl::InternalError(absl::StrFormat(
            "Interpreter returned invalid Value for path %d", i));
      }
      if (config.oracle != nullptr) {
        XLS_RETURN_IF_ERROR(config.oracle(path, result));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace xls::solvers::symex
