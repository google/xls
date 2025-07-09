// Copyright 2022 The XLS Authors
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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/contrib/xlscc/node_manipulation.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3_api.h"

using ::std::shared_ptr;
using ::std::string;
using ::std::vector;

namespace xlscc {

absl::Status Translator::GenerateIR_Loop(
    bool always_first_iter, const clang::Stmt* loop_stmt,
    clang::ArrayRef<const clang::AnnotateAttr*> attrs, const clang::Stmt* init,
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, const clang::PresumedLoc& presumed_loc,
    const xls::SourceInfo& loc, clang::ASTContext& ctx) {
  if (cond_expr != nullptr && cond_expr->isIntegerConstantExpr(ctx)) {
    // special case for "for (;0;) {}" (essentially no op)
    XLS_ASSIGN_OR_RETURN(auto constVal, EvaluateInt64(*cond_expr, ctx, loc));
    if (constVal == 0) {
      return absl::OkStatus();
    }
  }

  XLS_ASSIGN_OR_RETURN(
      std::optional<int64_t> unroll_factor_optional,
      GetAnnotationWithNonNegativeIntegerParam(
          attrs, "hls_unroll", loc, ctx,
          /*default_value=*/std::numeric_limits<int64_t>::max()));

  XLS_ASSIGN_OR_RETURN(std::optional<int64_t> init_interval_optional,
                       GetAnnotationWithNonNegativeIntegerParam(
                           attrs, "hls_pipeline_init_interval", loc, ctx));

  // Both pragmas/attributes cannot be present
  XLSCC_CHECK(!(unroll_factor_optional.has_value() &&
                init_interval_optional.has_value()),
              loc);

  // hls_unroll can indicate either unrolling or pipelining (partial unroll).

  const bool no_pragma = !unroll_factor_optional.has_value() &&
                         !init_interval_optional.has_value();
  const bool default_unroll = no_pragma && context().for_loops_default_unroll;
  const bool inferred_loop_warning_on =
      debug_ir_trace_flags_ & DebugIrTraceFlags_OptimizationWarnings;

  if (default_unroll ||
      (unroll_factor_optional.has_value() &&
       unroll_factor_optional.value() == std::numeric_limits<int64_t>::max())) {
    const bool warn_inferred_loop_type =
        default_unroll && inferred_loop_warning_on;

    return GenerateIR_LoopImpl(always_first_iter, warn_inferred_loop_type, init,
                               cond_expr, inc, body,
                               /*max_iters=*/std::nullopt,
                               /*propagate_break_up=*/false, ctx, loc);
  }

  int64_t init_interval = -1;
  int64_t unroll_factor = -1;
  bool warn_inferred_loop_type = false;

  if (init_interval_optional.has_value()) {
    XLSCC_CHECK((!unroll_factor_optional.has_value()) ||
                    (unroll_factor_optional.value() ==
                     std::numeric_limits<int64_t>::max()),
                loc);
    init_interval = init_interval_optional.value();
    unroll_factor = 1;
  } else if (unroll_factor_optional.has_value()) {
    XLSCC_CHECK(!init_interval_optional.has_value(), loc);
    init_interval = 1;
    unroll_factor = unroll_factor_optional.value();
  } else if (context().outer_pipelined_loop_init_interval > 0) {
    init_interval = context().outer_pipelined_loop_init_interval;
    unroll_factor = 1;
    warn_inferred_loop_type = inferred_loop_warning_on;
  } else {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Loop statement missing #pragma or attribute"));
  }

  CHECK(init_interval > 0 && unroll_factor > 0);

  bool is_asap = HasAnnotation(attrs, "xlscc_asap");

  XLS_RETURN_IF_ERROR(CheckInitIntervalValidity(init_interval, loc));
  if (generate_new_fsm_) {
    return GenerateIR_PipelinedLoopNewFSM(
        always_first_iter, warn_inferred_loop_type, init, cond_expr, inc, body,
        init_interval, unroll_factor, is_asap, ctx, loc);
  }
  return GenerateIR_PipelinedLoopOldFSM(
      always_first_iter, warn_inferred_loop_type, init, cond_expr, inc, body,
      init_interval, unroll_factor, is_asap, ctx, loc);
}

absl::Status Translator::GenerateIR_LoopImpl(
    bool always_first_iter, bool warn_inferred_loop_type,
    const clang::Stmt* init, const clang::Expr* cond_expr,
    const clang::Stmt* inc, const clang::Stmt* body,
    std::optional<int64_t> max_iters, bool propagate_break_up,
    clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  XLSCC_CHECK(!max_iters.has_value() || max_iters.value() > 0, loc);

  const bool add_loop_jump = generate_new_fsm_ && max_iters.has_value();

  Z3_solver current_solver = nullptr;
  xls::solvers::z3::IrTranslator* current_z3_translator = nullptr;

  auto deref_solver = [&current_solver, &current_z3_translator]() {
    if (current_solver == nullptr) {
      return;
    }
    CHECK_NE(current_z3_translator, nullptr);
    Z3_solver_dec_ref(current_z3_translator->ctx(), current_solver);
    current_solver = nullptr;
    current_z3_translator = nullptr;
  };
  auto deref_solver_guard = absl::MakeCleanup(deref_solver);

  auto refresh_z3 = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(xls::solvers::z3::IrTranslator * z3_translator,
                         GetZ3Translator(context().fb->function()));

    if (z3_translator != current_z3_translator) {
      deref_solver();
      current_z3_translator = z3_translator;
      current_solver =
          xls::solvers::z3::CreateSolver(current_z3_translator->ctx(), 1);
    };

    return absl::OkStatus();
  };

  // Generate the declaration within a private context
  PushContextGuard for_init_guard(*this, loc);
  context().propagate_break_up = propagate_break_up;
  context().propagate_continue_up = false;
  context().in_for_body = true;
  context().in_switch_body = false;

  if (init != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(init, ctx));
  }

  bool first_iter_cond_must_be_true = false;
  // Partially unrolled loops can also omit the condition from the first
  // iteration within the activation, as it'll be guarded by the jump condition.

  const int64_t io_ops_before = context().sf->io_ops.size();

  // Loop unrolling causes duplicate NamedDecls which fail the soundness
  // check. Reset the known set before each iteration.
  auto saved_check_ids = unique_decl_ids_;

  absl::Duration slowest_iter = absl::ZeroDuration();

  // Literals won't be propagated through the begin op slice boundary,
  // as this must be preserved for the phis to be created. Therefore this must
  // be checked before that boundary.
  if (cond_expr != nullptr && !always_first_iter) {
    XLS_ASSIGN_OR_RETURN(CValue cond_expr_cval,
                         GenerateIR_Expr(cond_expr, loc));
    CHECK(cond_expr_cval.type()->Is<CBoolType>());

    XLS_RETURN_IF_ERROR(refresh_z3());
    TrackedBValue cond_bval = cond_expr_cval.rvalue();
    XLS_ASSIGN_OR_RETURN(
        bool condition_must_be_true,
        BitMustBe(true, cond_bval, current_solver, current_z3_translator, loc));

    first_iter_cond_must_be_true = condition_must_be_true;
  }

  IOOp* begin_op = nullptr;

  if (add_loop_jump) {
    XLS_ASSIGN_OR_RETURN(begin_op, GenerateIR_AddLoopBegin(loc));
  }

  for (int64_t nIters = 0; !max_iters.has_value() || nIters < max_iters.value();
       ++nIters) {
    const bool first_iter = nIters == 0;
    const bool always_this_iter = always_first_iter && first_iter;

    xls::Stopwatch stopwatch;

    unique_decl_ids_ = saved_check_ids;

    if (nIters > max_unroll_iters_) {
      return absl::ResourceExhaustedError(
          ErrorMessage(loc, "Loop unrolling broke at maximum %i iterations",
                       max_unroll_iters_));
    }
    if (nIters == warn_unroll_iters_ &&
        debug_ir_trace_flags_ & DebugIrTraceFlags_OptimizationWarnings) {
      LOG(WARNING) << WarningMessage(
          loc, "Loop unrolling has reached %i iterations", warn_unroll_iters_);
    }

    // Generate condition.
    //
    // Outside of body context guard so it applies to increment
    // Also, if this is inside the body context guard then the break condition
    // feeds back on itself in an explosion of complexity
    // via assignments to any variables used in the condition.
    if (!always_this_iter && cond_expr != nullptr) {
      XLS_ASSIGN_OR_RETURN(CValue cond_expr_cval,
                           GenerateIR_Expr(cond_expr, loc));
      CHECK(cond_expr_cval.type()->Is<CBoolType>());

      if (!(first_iter && first_iter_cond_must_be_true)) {
        context().or_condition_util(
            context().fb->Not(cond_expr_cval.rvalue(), loc),
            context().relative_break_condition, loc);
        XLS_RETURN_IF_ERROR(and_condition(cond_expr_cval.rvalue(), loc));
      }
    }

    if (!add_loop_jump) {
      XLS_RETURN_IF_ERROR(refresh_z3());

      XLS_ASSIGN_OR_RETURN(
          bool condition_must_be_false,
          BitMustBe(false, context().relative_condition, current_solver,
                    current_z3_translator, loc));

      if (condition_must_be_false) {
        break;
      }
    }

    // Generate body
    {
      PushContextGuard for_body_guard(*this, loc);
      context().propagate_break_up = true;
      context().propagate_continue_up = false;

      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, ctx));
    }

    // Generate increment
    // Outside of body guard because continue would skip.
    if (inc != nullptr) {
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(inc, ctx));
    }

    // Print slow unrolling warning
    const absl::Duration elapsed_time = stopwatch.GetElapsedTime();
    if (debug_ir_trace_flags_ & DebugIrTraceFlags_OptimizationWarnings &&
        elapsed_time > absl::Seconds(0.1) && elapsed_time > slowest_iter) {
      LOG(WARNING) << WarningMessage(
          loc, "Slow loop unrolling iteration %i: %v", nIters, elapsed_time);
      slowest_iter = elapsed_time;
    }
  }

  if (warn_inferred_loop_type) {
    const int64_t total_io_ops = context().sf->io_ops.size() - io_ops_before;

    LOG(WARNING) << WarningMessage(
        loc,
        "Inferred unrolling for loop with %li IO operations after unrolling",
        total_io_ops);
  }

  if (add_loop_jump) {
    XLS_RETURN_IF_ERROR(GenerateIR_AddLoopEndJump(cond_expr, begin_op, loc));
  }

  return absl::OkStatus();
}

bool Translator::LValueContainsOnlyChannels(
    const std::shared_ptr<LValue>& lvalue) {
  if (lvalue == nullptr) {
    return true;
  }

  if (lvalue->get_compounds().empty() && lvalue->channel_leaf() == nullptr) {
    return false;
  }

  for (const auto& [idx, lval_field] : lvalue->get_compounds()) {
    if (!LValueContainsOnlyChannels(lval_field)) {
      return false;
    }
  }

  return true;
}

// Must match order in TranslateLValueConditions

absl::Status Translator::SendLValueConditions(
    const std::shared_ptr<LValue>& lvalue,
    std::vector<TrackedBValue>* lvalue_conditions, const xls::SourceInfo& loc) {
  for (const auto& [idx, compound_lval] : lvalue->get_compounds()) {
    XLS_RETURN_IF_ERROR(
        SendLValueConditions(compound_lval, lvalue_conditions, loc));
  }
  if (!lvalue->is_select()) {
    return absl::OkStatus();
  }
  lvalue_conditions->push_back(lvalue->cond());

  XLS_RETURN_IF_ERROR(
      SendLValueConditions(lvalue->lvalue_true(), lvalue_conditions, loc));
  XLS_RETURN_IF_ERROR(
      SendLValueConditions(lvalue->lvalue_false(), lvalue_conditions, loc));

  return absl::OkStatus();
}

// Must match order in SendLValueConditions
absl::StatusOr<std::shared_ptr<LValue>> Translator::TranslateLValueConditions(
    const std::shared_ptr<LValue>& outer_lvalue,
    TrackedBValue lvalue_conditions_tuple, const xls::SourceInfo& loc,
    int64_t* at_index) {
  if (outer_lvalue == nullptr) {
    return nullptr;
  }
  if (!outer_lvalue->get_compounds().empty()) {
    LValueMap<int64_t> compounds;
    for (const auto& [idx, compound_lval] : outer_lvalue->get_compounds()) {
      XLS_ASSIGN_OR_RETURN(
          compounds[idx],
          TranslateLValueConditions(compound_lval, lvalue_conditions_tuple, loc,
                                    at_index));
    }
    return std::make_shared<LValue>(compounds);
  }

  if (!outer_lvalue->is_select()) {
    return outer_lvalue;
  }
  int64_t at_index_storage = 0;
  if (at_index == nullptr) {
    at_index = &at_index_storage;
  }
  TrackedBValue translated_condition =
      context().fb->TupleIndex(lvalue_conditions_tuple, *at_index, loc);
  ++(*at_index);

  XLS_ASSIGN_OR_RETURN(
      std::shared_ptr<LValue> translated_lvalue_true,
      TranslateLValueConditions(outer_lvalue->lvalue_true(),
                                lvalue_conditions_tuple, loc, at_index));
  XLS_ASSIGN_OR_RETURN(
      std::shared_ptr<LValue> translated_lvalue_false,
      TranslateLValueConditions(outer_lvalue->lvalue_false(),
                                lvalue_conditions_tuple, loc, at_index));

  return std::make_shared<LValue>(translated_condition, translated_lvalue_true,
                                  translated_lvalue_false);
}

absl::StatusOr<IOOp*> Translator::GenerateIR_AddLoopBegin(
    const xls::SourceInfo& loc) {
  IOOp label_op = {.op = OpType::kLoopBegin,
                   // Jump past loop condition
                   .ret_value = context().fb->Literal(xls::UBits(0, 1), loc)};

  XLS_ASSIGN_OR_RETURN(
      IOOp * label_op_ptr,
      AddOpToChannel(label_op, /*channel_param=*/nullptr, loc));

  return label_op_ptr;
}

absl::Status Translator::GenerateIR_AddLoopEndJump(const clang::Expr* cond_expr,
                                                   IOOp* begin_op,
                                                   const xls::SourceInfo& loc) {
  XLSCC_CHECK_NE(begin_op, nullptr, loc);

  TrackedBValue continue_condition = context().full_condition_bval(loc);

  if (cond_expr != nullptr) {
    XLS_ASSIGN_OR_RETURN(CValue cond_expr_cval,
                         GenerateIR_Expr(cond_expr, loc));
    CHECK(cond_expr_cval.type()->Is<CBoolType>());

    continue_condition =
        context().fb->And(continue_condition, cond_expr_cval.rvalue(), loc,
                          /*name=*/"continue_jump_condition");
  }

  IOOp jump_op = {.op = OpType::kLoopEndJump,
                  .loop_op_paired = begin_op,
                  // Jump back to begin condition
                  .ret_value = continue_condition};

  XLS_ASSIGN_OR_RETURN(begin_op->loop_op_paired,
                       AddOpToChannel(jump_op, /*channel_param=*/nullptr, loc));

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_PipelinedLoopNewFSM(
    bool always_first_iter, bool warn_inferred_loop_type,
    const clang::Stmt* init, const clang::Expr* cond_expr,
    const clang::Stmt* inc, const clang::Stmt* body,
    int64_t initiation_interval_arg, int64_t unroll_factor, bool schedule_asap,
    clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  const int64_t prev_init = context().outer_pipelined_loop_init_interval;
  context().outer_pipelined_loop_init_interval = initiation_interval_arg;

  XLS_RETURN_IF_ERROR(GenerateIR_LoopImpl(
      always_first_iter, warn_inferred_loop_type, init, cond_expr, inc, body,
      /*max_iters=*/unroll_factor,
      /*propagate_break_up=*/false, ctx, loc));

  context().outer_pipelined_loop_init_interval = prev_init;
  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_PipelinedLoopOldFSM(
    bool always_first_iter, bool warn_inferred_loop_type,
    const clang::Stmt* init, const clang::Expr* cond_expr,
    const clang::Stmt* inc, const clang::Stmt* body,
    int64_t initiation_interval_arg, int64_t unroll_factor, bool schedule_asap,
    clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  const TranslationContext& outer_context = context();

  XLSCC_CHECK(!generate_new_fsm_, loc);

  // Generate the loop counter declaration within a private context
  // By doing this here, it automatically gets rolled into proc state
  // This causes it to be automatically reset on break
  PushContextGuard for_init_guard(*this, loc);

  if (init != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(init, ctx));
  }

  // Condition must be checked at the start
  if (!always_first_iter && cond_expr != nullptr) {
    XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
    CHECK(cond_cval.type()->Is<CBoolType>());

    XLS_RETURN_IF_ERROR(and_condition(cond_cval.rvalue(), loc));
  }

  // Pack context tuple
  std::shared_ptr<CStructType> context_cvars_struct_ctype;
  std::shared_ptr<CInternalTuple> context_lval_conds_ctype;

  xls::Type* context_struct_xls_type = nullptr;
  xls::Type* context_lvals_xls_type = nullptr;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t> context_field_indices;
  std::vector<const clang::NamedDecl*> variable_fields_order;
  TrackedBValue lvalue_conditions_tuple;
  {
    std::vector<TrackedBValue> full_context_tuple_values;
    std::vector<std::shared_ptr<CField>> full_context_fields;

    XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                         parser_->GetXlsccOnReset());

    // Create a deterministic field order
    for (const auto& [decl, _] : context().variables) {
      CHECK(context().sf->declaration_order_by_name_.contains(decl));
      // Don't pass __xlscc_on_reset in/out
      if (decl == on_reset_var_decl) {
        continue;
      }
      variable_fields_order.push_back(decl);
    }

    context().sf->SortNamesDeterministically(variable_fields_order);

    std::vector<TrackedBValue> lvalue_conditions;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      // Don't mark access
      // These are handled below based on what's really used in the loop body
      XLS_ASSIGN_OR_RETURN(const CValue& cvalue,
                           GetIdentifier(decl, loc, /*record_access=*/false));

      if (cvalue.rvalue().valid()) {
        const uint64_t field_idx = context_field_indices.size();
        context_field_indices[decl] = field_idx;
        auto field_ptr =
            std::make_shared<CField>(decl, field_idx, cvalue.type());
        full_context_fields.push_back(field_ptr);
      }

      if (cvalue.lvalue() != nullptr) {
        XLS_RETURN_IF_ERROR(
            SendLValueConditions(cvalue.lvalue(), &lvalue_conditions, loc));
      }
    }

    lvalue_conditions_tuple =
        context().fb->Tuple(ToNativeBValues(lvalue_conditions), loc,
                            /*name=*/"lvalue_conditions");
    std::vector<std::shared_ptr<CType>> lvalue_conds_tuple_fields;
    lvalue_conds_tuple_fields.resize(lvalue_conditions.size(),
                                     std::make_shared<CBoolType>());
    context_lval_conds_ctype =
        std::make_shared<CInternalTuple>(lvalue_conds_tuple_fields);

    context_cvars_struct_ctype = std::make_shared<CStructType>(
        full_context_fields, /*no_tuple=*/false, /*synthetic_int=*/false);

    XLS_ASSIGN_OR_RETURN(context_struct_xls_type,
                         TranslateTypeToXLS(context_cvars_struct_ctype, loc));
    context_lvals_xls_type = lvalue_conditions_tuple.GetType();
  }

  const std::string name_prefix =
      absl::StrFormat("__for_%i", next_for_number_++);

  // Create loop body proc
  LValueMap<const clang::NamedDecl*> lvalues_out;
  bool uses_on_reset = false;
  XLS_ASSIGN_OR_RETURN(
      PipelinedLoopSubProc sub_proc,
      GenerateIR_PipelinedLoopBody(
          cond_expr, inc, body, initiation_interval_arg, unroll_factor,
          always_first_iter, ctx, name_prefix, context_struct_xls_type,
          context_lvals_xls_type, context_cvars_struct_ctype, &lvalues_out,
          context_field_indices, variable_fields_order, &uses_on_reset, loc));

  if (warn_inferred_loop_type) {
    LOG(WARNING) << WarningMessage(
        loc,
        "Inferred pipelining for loop with %li IO operations after unrolling "
        "(minus inner pipelined loop ops), %li sub procs after unrolling",
        sub_proc.generated_func->io_ops.size() -
            sub_proc.generated_func->sub_procs.size() * 2,
        sub_proc.generated_func->sub_procs.size());
  }

  // Propagate variables accessed to the outer context. Necessary for nested
  // loops.
  // Context in doesn't mark usage so that only things really used
  // in the loop body are counted.
  for (const std::pair<const clang::NamedDecl*, int64_t>& accessed :
       sub_proc.vars_accessed_in_body) {
    const clang::NamedDecl* decl = accessed.first;
    if (context().variables.contains(decl)) {
      context().variables_accessed[decl] += accessed.second;
    }
  }

  std::shared_ptr<CStructType> context_out_cvars_struct_ctype;
  std::shared_ptr<CInternalTuple> context_out_lval_conds_ctype =
      context_lval_conds_ctype;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      context_out_field_indices;

  // Filter context out
  CValue context_tuple_out;
  {
    std::vector<std::shared_ptr<CField>> context_out_fields;
    std::vector<TrackedBValue> context_out_tuple_values;

    absl::flat_hash_set<const clang::NamedDecl*> vars_accessed_in_body_set;
    for (const std::pair<const clang::NamedDecl*, int64_t>& accessed :
         sub_proc.vars_accessed_in_body) {
      const clang::NamedDecl* decl = accessed.first;
      vars_accessed_in_body_set.insert(decl);
    }

    for (const clang::NamedDecl* decl : variable_fields_order) {
      if (!vars_accessed_in_body_set.contains(decl)) {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(const CValue& cvalue,
                           GetIdentifier(decl, loc, /*record_access=*/false));
      // Not concerned with LValues
      if (!cvalue.rvalue().valid()) {
        continue;
      }

      // TODO(seanhaskell): This should allow for direct-in derived values
      // b/321114633
      if (schedule_asap && outer_context.variables.contains(decl)) {
        const TrackedBValue& rvalue = outer_context.variables.at(decl).rvalue();

        if (rvalue.valid() &&
            !EvaluateNode(rvalue.node(), loc, /*do_check=*/false).ok()) {
          return absl::UnimplementedError(
              ErrorMessage(loc,
                           "Cannot access variable in outside scope from loop "
                           "which runs asynchronously: %s",
                           decl->getQualifiedNameAsString().c_str()));
        }
      }

      const int64_t field_idx = context_out_tuple_values.size();

      context_out_tuple_values.push_back(cvalue.rvalue());
      context_out_field_indices[decl] = field_idx;

      const int64_t full_field_idx = context_field_indices.at(decl);
      std::shared_ptr<CType> field_type =
          context_cvars_struct_ctype->fields().at(full_field_idx)->type();
      auto field_ptr = std::make_shared<CField>(decl, field_idx, field_type);
      context_out_fields.push_back(field_ptr);
    }
    context_out_cvars_struct_ctype = std::make_shared<CStructType>(
        context_out_fields, /*no_tuple=*/false, /*synthetic_int=*/false);

    CValue context_struct_out =
        CValue(MakeStructXLS(context_out_tuple_values,
                             *context_out_cvars_struct_ctype, loc),
               context_out_cvars_struct_ctype);

    std::vector<std::shared_ptr<CType>> context_tuple_elem_types;
    context_tuple_elem_types.push_back(context_out_cvars_struct_ctype);
    context_tuple_elem_types.push_back(context_lval_conds_ctype);
    std::shared_ptr<CInternalTuple> context_tuple_type =
        std::make_shared<CInternalTuple>(context_tuple_elem_types);

    // Set later if needed
    TrackedBValue outer_on_reset_value =
        context().fb->Literal(xls::UBits(0, 1), loc);

    // Must match if(uses_on_reset) below
    context_tuple_out = CValue(
        context().fb->Tuple({outer_on_reset_value, context_struct_out.rvalue(),
                             lvalue_conditions_tuple},
                            loc, /*name=*/"context_out_tuple_inner"),
        context_tuple_type);
  }

  // Create synthetic channels and IO ops
  std::shared_ptr<CStructType> context_in_cvars_struct_ctype;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      context_in_field_indices;

  {
    std::vector<std::shared_ptr<CField>> fields;
    for (const clang::NamedDecl* decl : sub_proc.vars_changed_in_body) {
      // Don't assign to variables that don't exist in the outside scope
      if (!outer_context.variables.contains(decl)) {
        continue;
      }
      const uint64_t field_idx = fields.size();
      auto field_ptr = std::make_shared<CField>(
          decl, field_idx, outer_context.variables.at(decl).type());
      fields.push_back(field_ptr);
      context_in_field_indices[decl] = field_idx;
    }

    context_in_cvars_struct_ctype = std::make_shared<CStructType>(
        fields, /*no_tuple=*/false, /*synthetic_int=*/false);
  }

  // Pick a construct to correlate the channels for this construct
  const clang::Stmt* identify_channels_stmt = body;
  XLSCC_CHECK(identify_channels_stmt != nullptr, loc);

  // Create context channels
  IOChannel* context_out_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_out", name_prefix);
    xls::Channel* xls_channel = nullptr;

    IOChannel new_channel;
    new_channel.item_type = context_tuple_out.type();
    new_channel.unique_name = ch_name;
    new_channel.generated = xls_channel;
    context_out_channel = AddChannel(new_channel, loc);
  }
  IOChannel* context_in_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_in", name_prefix);
    xls::Channel* xls_channel = nullptr;

    IOChannel new_channel;
    new_channel.item_type = context_in_cvars_struct_ctype;
    new_channel.unique_name = ch_name;
    new_channel.generated = xls_channel;
    context_in_channel = AddChannel(new_channel, loc);
  }

  // Fill in context variables for sub proc
  sub_proc.context_out_channel = context_out_channel;
  sub_proc.context_in_channel = context_in_channel;
  sub_proc.context_cvars_struct_ctype = context_cvars_struct_ctype;
  sub_proc.context_in_cvars_struct_ctype = context_in_cvars_struct_ctype;
  sub_proc.context_in_field_indices = context_in_field_indices;
  sub_proc.context_out_cvars_struct_ctype = context_out_cvars_struct_ctype;
  sub_proc.context_out_field_indices = context_out_field_indices;
  sub_proc.context_out_lval_conds_ctype = context_out_lval_conds_ctype;

  CHECK_EQ(sub_proc.vars_changed_in_body.size(), lvalues_out.size());

  if (uses_on_reset) {
    XLS_ASSIGN_OR_RETURN(CValue on_reset_cval, GetOnReset(loc));
    XLSCC_CHECK_EQ(on_reset_cval.type()->GetBitWidth(), 1, loc);

    // Must match tuple creation above
    context_tuple_out = CValue(
        context().fb->Tuple(
            {on_reset_cval.rvalue(),
             context().fb->TupleIndex(context_tuple_out.rvalue(), 1, loc,
                                      /*name=*/"context_out_outer_struct"),
             context().fb->TupleIndex(
                 context_tuple_out.rvalue(), 2, loc,
                 /*name=*/"context_out_outer_lvalue_conditions")},
            loc,
            /*name=*/"context_out_tuple_outer"),
        context_tuple_out.type());
  }

  // Send and receive context tuples
  IOOp* ctx_out_op_ptr = nullptr;
  {
    IOOp op;
    op.op = OpType::kSend;
    std::vector<TrackedBValue> sp = {context_tuple_out.rvalue(),
                                     context().full_condition_bval(loc)};
    op.ret_value = context().fb->Tuple(ToNativeBValues(sp), loc,
                                       /*name=*/"context_out_send_tup");
    XLS_ASSIGN_OR_RETURN(ctx_out_op_ptr,
                         AddOpToChannel(op, context_out_channel, loc));
  }

  // AddOpToChannel sequences automatically in the default case according
  // to op_ordering_
  if (schedule_asap) {
    ctx_out_op_ptr->scheduling_option = IOSchedulingOption::kASAPBefore;
    ctx_out_op_ptr->after_ops.clear();
  }

  IOOp* ctx_in_op_ptr;
  {
    IOOp op;
    op.op = OpType::kRecv;
    op.ret_value = context().full_condition_bval(loc);
    XLS_ASSIGN_OR_RETURN(ctx_in_op_ptr,
                         AddOpToChannel(op, context_in_channel, loc));
  }

  if (schedule_asap) {
    ctx_in_op_ptr->scheduling_option = IOSchedulingOption::kASAPBefore;
    ctx_in_op_ptr->after_ops.clear();
  }

  // This must be added explicitly, as op_ordering_ may not add it
  ctx_in_op_ptr->after_ops.push_back(ctx_out_op_ptr);

  // Unpack context tuple
  TrackedBValue context_tuple_recvd = ctx_in_op_ptr->input_value.rvalue();
  {
    // Don't assign to variables that aren't changed in the loop body,
    // as this creates extra state
    for (const clang::NamedDecl* decl : sub_proc.vars_changed_in_body) {
      if (!context_in_field_indices.contains(decl)) {
        continue;
      }

      uint64_t field_idx = context_in_field_indices.at(decl);

      if (schedule_asap) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Cannot assign to variable in outside scope from loop "
                         "which runs asynchronously: %s",
                         decl->getQualifiedNameAsString().c_str()));
      }

      const CValue prev_cval = context().variables.at(decl);

      const CValue cval(GetStructFieldXLS(context_tuple_recvd, field_idx,
                                          *context_in_cvars_struct_ctype, loc),
                        prev_cval.type(), /*disable_type_check=*/false,
                        lvalues_out.at(decl));
      XLS_RETURN_IF_ERROR(Assign(decl, cval, loc));
    }
  }

  // Record sub-proc for generation later
  context().sf->sub_procs.push_back(std::move(sub_proc));

  const PipelinedLoopSubProc* final_sub_proc_ptr =
      &context().sf->sub_procs.back();

  context().sf->pipeline_loops_by_internal_channel[context_out_channel] =
      final_sub_proc_ptr;
  context().sf->pipeline_loops_by_internal_channel[context_in_channel] =
      final_sub_proc_ptr;

  return absl::OkStatus();
}

absl::StatusOr<PipelinedLoopSubProc> Translator::GenerateIR_PipelinedLoopBody(
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t init_interval, int64_t unroll_factor,
    bool always_first_iter, clang::ASTContext& ctx,
    std::string_view name_prefix, xls::Type* context_struct_xls_type,
    xls::Type* context_lvals_xls_type,
    const std::shared_ptr<CStructType>& context_cvars_struct_ctype,
    LValueMap<const clang::NamedDecl*>* lvalues_out,
    const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
        context_field_indices,
    const std::vector<const clang::NamedDecl*>& variable_fields_order,
    bool* uses_on_reset, const xls::SourceInfo& loc) {
  std::vector<std::pair<const clang::NamedDecl*, int64_t>>
      vars_accessed_in_body;
  std::vector<const clang::NamedDecl*> vars_changed_in_body;

  GeneratedFunction& enclosing_func = *context().sf;

  // Generate body function
  const std::string loop_name = absl::StrFormat("%s_loop", name_prefix);

  auto generated_func = std::make_unique<GeneratedFunction>();
  CHECK_NE(context().sf, nullptr);
  CHECK_NE(context().sf->clang_decl, nullptr);
  generated_func->clang_decl = context().sf->clang_decl;

  uint64_t extra_return_count = 0;
  {
    // Set up IR generation
    TrackedFunctionBuilder body_builder(loop_name, package_);

    auto clean_up_bvalues = [&generated_func]() {
      CleanUpBValuesInTopFunction(*generated_func);
    };

    auto clean_up_bvalues_guard = absl::MakeCleanup(clean_up_bvalues);

    TranslationContext& prev_context = context();
    PushContextGuard context_guard(*this, loc);

    context() = TranslationContext();
    context().propagate_up = false;
    context().fb =
        absl::implicit_cast<xls::BuilderBase*>(body_builder.builder());
    context().sf = generated_func.get();
    context().ast_context = prev_context.ast_context;
    context().in_pipelined_for_body = true;
    context().outer_pipelined_loop_init_interval = init_interval;

    TrackedBValue context_struct_val =
        context().fb->Param(absl::StrFormat("%s_context_vars", name_prefix),
                            context_struct_xls_type, loc);
    TrackedBValue context_lvalues_val =
        context().fb->Param(absl::StrFormat("%s_context_lvals", name_prefix),
                            context_lvals_xls_type, loc);
    TrackedBValue context_on_reset_val =
        context().fb->Param(absl::StrFormat("%s_on_reset", name_prefix),
                            package_->GetBitsType(1), loc);

    absl::flat_hash_map<IOChannel*, IOChannel*> inner_channels_by_outer_channel;
    absl::flat_hash_map<IOChannel*, IOChannel*> outer_channels_by_inner_channel;

    // Inherit external channels
    for (IOChannel& enclosing_channel : enclosing_func.io_channels) {
      if (enclosing_channel.generated.has_value()) {
        continue;
      }
      generated_func->io_channels.push_back(enclosing_channel);
      IOChannel* inner_channel = &generated_func->io_channels.back();
      inner_channel->total_ops = 0;

      inner_channels_by_outer_channel[&enclosing_channel] = inner_channel;
      outer_channels_by_inner_channel[inner_channel] = &enclosing_channel;

      XLSCC_CHECK(
          external_channels_by_internal_channel_.contains(&enclosing_channel),
          loc);

      if (external_channels_by_internal_channel_.count(&enclosing_channel) >
          1) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "IO ops in pipelined loops in subroutines called "
                         "with multiple different channel arguments"));
      }

      const ChannelBundle enclosing_bundle =
          external_channels_by_internal_channel_.find(&enclosing_channel)
              ->second;

      // Don't use = .at(), avoid compiler bug
      std::pair<const IOChannel*, ChannelBundle> pair(inner_channel,
                                                      enclosing_bundle);
      if (!ContainsKeyValuePair(external_channels_by_internal_channel_, pair)) {
        external_channels_by_internal_channel_.insert(pair);
      }
    }

    // Declare __xlscc_on_reset
    XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                         parser_->GetXlsccOnReset());
    XLS_RETURN_IF_ERROR(DeclareVariable(
        on_reset_var_decl,
        CValue(context_on_reset_val, std::make_shared<CBoolType>()), loc,
        /*check_unique_ids=*/false));

    // Context in
    CValueMap<const clang::NamedDecl*> prev_vars;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue& outer_value = prev_context.variables.at(decl);
      TrackedBValue param_bval;
      if (context_field_indices.contains(decl)) {
        const uint64_t field_idx = context_field_indices.at(decl);
        param_bval =
            GetStructFieldXLS(context_struct_val, static_cast<int>(field_idx),
                              *context_cvars_struct_ctype, loc);
      }

      std::shared_ptr<LValue> inner_lval;
      XLS_ASSIGN_OR_RETURN(
          inner_lval,
          TranslateLValueChannels(outer_value.lvalue(),
                                  inner_channels_by_outer_channel, loc));

      XLS_ASSIGN_OR_RETURN(
          inner_lval,
          TranslateLValueConditions(inner_lval, context_lvalues_val, loc));

      CValue prev_var(param_bval, outer_value.type(),
                      /*disable_type_check=*/false, inner_lval);
      prev_vars[decl] = prev_var;

      // __xlscc_on_reset handled separately
      if (decl == on_reset_var_decl) {
        continue;
      }

      XLS_RETURN_IF_ERROR(
          DeclareVariable(decl, prev_var, loc, /*check_unique_ids=*/false));
    }

    // Generate initial loop condition, before body, for narrowing
    TrackedBValue initial_loop_cond = context().fb->Literal(xls::UBits(1, 1));

    // always_first_iter = true for do loops, and this optimization opportunity
    // doesn't apply to them
    if (cond_expr != nullptr && !always_first_iter) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);

      XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
      CHECK(cond_cval.type()->Is<CBoolType>());

      initial_loop_cond = cond_cval.rvalue();
    }

    TrackedBValue do_break = context().fb->Literal(xls::UBits(0, 1));

    // Generate body
    // Don't apply continue conditions to increment
    // This context pop will top generate selects
    {
      PushContextGuard context_guard(*this, loc);
      context().propagate_break_up = false;
      context().propagate_continue_up = false;
      context().in_for_body = true;

      CHECK_GT(context().outer_pipelined_loop_init_interval, 0);

      CHECK_NE(body, nullptr);

      // always_first_iter = true is safe because the body is already
      // conditioned by the pipelined loop machinery.
      XLS_RETURN_IF_ERROR(GenerateIR_LoopImpl(
          /*always_first_iter=*/true,
          /*warn_inferred_loop_type=*/false,
          /*init=*/nullptr, cond_expr, /*inc=*/inc, body,
          /*max_iters=*/unroll_factor,
          /*propagate_break_up=*/true, ctx, loc));

      // break_condition is the assignment condition
      if (context().relative_break_condition.valid()) {
        TrackedBValue break_cond = context().relative_break_condition;
        do_break = context().fb->Or(do_break, break_cond, loc);
      }
    }

    // Incrementor is handled by GenerateIR_LoopImpl

    // Check condition
    if (cond_expr != nullptr) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);

      XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
      CHECK(cond_cval.type()->Is<CBoolType>());
      TrackedBValue break_on_cond_val = context().fb->Not(cond_cval.rvalue());

      do_break = context().fb->Or(do_break, break_on_cond_val, loc);
    }

    // Context out
    const uint64_t total_context_values =
        context_cvars_struct_ctype->fields().size();

    std::vector<TrackedBValue> tuple_values;
    tuple_values.resize(total_context_values);
    for (const clang::NamedDecl* decl : variable_fields_order) {
      if (!context_field_indices.contains(decl)) {
        continue;
      }
      const uint64_t field_idx = context_field_indices.at(decl);
      tuple_values[field_idx] = context().variables.at(decl).rvalue();
    }

    TrackedBValue ret_ctx =
        MakeStructXLS(tuple_values, *context_cvars_struct_ctype, loc);
    std::vector<TrackedBValue> return_bvals = {ret_ctx, do_break,
                                               initial_loop_cond};

    // For GenerateIRBlock_Prepare() / GenerateInvokeWithIO()
    extra_return_count += return_bvals.size();

    // First static returns
    for (const clang::NamedDecl* decl :
         generated_func->GetDeterministicallyOrderedStaticValues()) {
      XLS_ASSIGN_OR_RETURN(CValue value, GetIdentifier(decl, loc));
      return_bvals.push_back(value.rvalue());
    }

    // IO returns
    for (IOOp& op : generated_func->io_ops) {
      CHECK(op.ret_value.valid());
      return_bvals.push_back(op.ret_value);
    }

    TrackedBValue ret_val = MakeFlexTuple(return_bvals, loc);
    generated_func->return_value_count = return_bvals.size();

    XLS_ASSIGN_OR_RETURN(context().sf->xls_func,
                         body_builder.builder()->BuildWithReturnValue(ret_val));

    // Analyze context variables changed
    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue prev_bval = prev_vars.at(decl);
      const CValue curr_val = context().variables.at(decl);
      if (!NodesEquivalentWithContinuations(prev_bval.rvalue().node(),
                                            curr_val.rvalue().node()) ||
          prev_bval.lvalue() != curr_val.lvalue()) {
        vars_changed_in_body.push_back(decl);
        XLS_ASSIGN_OR_RETURN(
            (*lvalues_out)[decl],
            TranslateLValueChannels(curr_val.lvalue(),
                                    outer_channels_by_inner_channel, loc));
      }
    }
    // vars_changed_in_body is already sorted deterministically due to
    // iterating over variable_fields_order

    for (const clang::NamedDecl* decl : variable_fields_order) {
      auto found = context().variables_accessed.find(decl);
      if (found == context().variables_accessed.end()) {
        continue;
      }
      vars_accessed_in_body.push_back(std::make_pair(decl, found->second));
      XLSCC_CHECK(context().sf->declaration_order_by_name_.contains(decl), loc);
    }
    // vars_accessed_in_body is already sorted deterministically due to
    // iterating over variable_fields_order
  }

  XLSCC_CHECK_NE(uses_on_reset, nullptr, loc);
  if (generated_func->uses_on_reset) {
    *uses_on_reset = true;
  }

  std::vector<const clang::NamedDecl*> vars_to_save_between_iters;

  {
    absl::flat_hash_set<const clang::NamedDecl*> vars_to_save_between_iters_set;

    // Save any variables which are changed
    for (const clang::NamedDecl* decl : vars_changed_in_body) {
      vars_to_save_between_iters_set.insert(decl);
    }

    for (const clang::NamedDecl* decl : vars_to_save_between_iters_set) {
      vars_to_save_between_iters.push_back(decl);
    }

    context().sf->SortNamesDeterministically(vars_to_save_between_iters);
  }

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopContext) {
    LOG(INFO) << absl::StrFormat("Variables to save for loop %s at %s:\n",
                                 loop_name, LocString(loc));
    for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
      LOG(INFO) << absl::StrFormat("-- %s:\n", decl->getNameAsString().c_str());
    }
  }

  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CType>>
      outer_variable_types;

  for (const auto& [decl, cval] : context().variables) {
    outer_variable_types[decl] = cval.type();
  }

  PipelinedLoopSubProc pipelined_loop_proc = {
      .name_prefix = name_prefix.data(),
      // context_ members are filled in by caller
      .loc = loc,

      .enclosing_func = context().sf,
      .outer_variable_types = outer_variable_types,
      .context_field_indices = context_field_indices,
      .extra_return_count = extra_return_count,
      .generated_func = std::move(generated_func),
      .variable_fields_order = variable_fields_order,
      .vars_changed_in_body = vars_changed_in_body,
      .vars_accessed_in_body = vars_accessed_in_body,
      .vars_to_save_between_iters = vars_to_save_between_iters};

  return pipelined_loop_proc;
}

absl::StatusOr<Translator::PipelinedLoopContentsReturn>
Translator::GenerateIR_PipelinedLoopContents(
    const PipelinedLoopSubProc& pipelined_loop_proc, xls::ProcBuilder& pb,
    TrackedBValue token_in, TrackedBValue received_context_tuple,
    TrackedBValue in_state_condition, bool in_fsm,
    absl::flat_hash_map<const clang::NamedDecl*, xls::StateElement*>*
        state_element_for_variable,
    int nesting_level) {
  const std::shared_ptr<CStructType>& context_in_cvars_struct_ctype =
      pipelined_loop_proc.context_in_cvars_struct_ctype;
  const std::shared_ptr<CStructType>& context_out_cvars_struct_ctype =
      pipelined_loop_proc.context_out_cvars_struct_ctype;
  const std::shared_ptr<CInternalTuple>& context_out_lval_conds_ctype =
      pipelined_loop_proc.context_out_lval_conds_ctype;
  const xls::SourceInfo& loc = pipelined_loop_proc.loc;

  const std::shared_ptr<CStructType>& context_cvars_struct_ctype =
      pipelined_loop_proc.context_cvars_struct_ctype;

  const std::vector<const clang::NamedDecl*>& variable_fields_order =
      pipelined_loop_proc.variable_fields_order;
  const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
      context_field_indices = pipelined_loop_proc.context_field_indices;
  const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
      context_in_field_indices = pipelined_loop_proc.context_in_field_indices;
  const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
      context_out_field_indices = pipelined_loop_proc.context_out_field_indices;

  const uint64_t extra_return_count = pipelined_loop_proc.extra_return_count;
  GeneratedFunction& generated_func = *pipelined_loop_proc.generated_func;

  const std::vector<const clang::NamedDecl*>& vars_to_save_between_iters =
      pipelined_loop_proc.vars_to_save_between_iters;

  // Generate body proc
  const std::string& name_prefix = pipelined_loop_proc.name_prefix;

  XLSCC_CHECK(!in_fsm || in_state_condition.valid(), loc);
  XLSCC_CHECK(!generate_new_fsm_, loc);

  PreparedBlock prepared;

  // Use state elements map from outer scope
  if (state_element_for_variable != nullptr) {
    prepared.state_element_for_variable = *state_element_for_variable;
  }

  if (!in_fsm) {
    in_state_condition =
        pb.Literal(xls::UBits(1, 1), loc,
                   absl::StrFormat("%s_in_state_default_1", name_prefix));
  }

  // Construct initial state
  TrackedBValue last_iter_broke_in =
      pb.StateElement(absl::StrFormat("%s__last_iter_broke", name_prefix),
                      xls::Value(xls::UBits(1, 1)));
  xls::StateElement* last_iter_broke_state =
      last_iter_broke_in.node()->As<xls::StateRead>()->state_element();

  XLS_ASSIGN_OR_RETURN(
      xls::Value default_lval_conds,
      CreateDefaultRawValue(context_out_lval_conds_ctype, loc));
  TrackedBValue lvalue_cond_value =
      pb.StateElement(absl::StrFormat("%s__lvalue_conditions", name_prefix),
                      default_lval_conds);
  xls::StateElement* lvalue_cond_state =
      lvalue_cond_value.node()->As<xls::StateRead>()->state_element();

  TrackedBValueMap<const clang::NamedDecl*> state_reads_by_decl;

  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    if (!context_field_indices.contains(decl)) {
      continue;
    }

    const bool do_create_state_element =
        !prepared.state_element_for_variable.contains(decl);

    if (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopContext) {
      LOG(INFO) << absl::StrFormat(
          "Variable to save %s will create state element? %i:\n",
          decl->getNameAsString().c_str(), (int)do_create_state_element);
    }

    // Only create a state element if one doesn't already exist
    if (do_create_state_element) {
      std::shared_ptr<CType> prev_value_type =
          pipelined_loop_proc.outer_variable_types.at(decl);
      XLS_ASSIGN_OR_RETURN(xls::Value def, CreateDefaultRawValue(
                                               prev_value_type, GetLoc(*decl)));

      TrackedBValue state_read_bval = pb.StateElement(
          absl::StrFormat("%s_%s", name_prefix, decl->getNameAsString()), def);
      xls::StateElement* state_elem =
          state_read_bval.node()->As<xls::StateRead>()->state_element();

      state_reads_by_decl[decl] = state_read_bval;
      prepared.state_element_for_variable[decl] = state_elem;
    } else {
      xls::StateElement* state_elem =
          prepared.state_element_for_variable.at(decl);
      state_reads_by_decl[decl] =
          TrackedBValue(pb.proc()->GetStateRead(state_elem), &pb);
    }
  }

  // For utility functions like MakeStructXls()
  PushContextGuard pb_guard(*this, in_state_condition, loc);

  TrackedBValue token = token_in;

  TrackedBValue received_on_reset = pb.TupleIndex(
      received_context_tuple, 0, loc,
      /*name=*/absl::StrFormat("%s_receive_on_reset", name_prefix));
  TrackedBValue received_context = pb.TupleIndex(
      received_context_tuple, 1, loc,
      /*name=*/absl::StrFormat("%s_receive_context_data", name_prefix));

  TrackedBValue received_lvalue_conds = pb.TupleIndex(
      received_context_tuple, 2, loc,
      /*name=*/absl::StrFormat("%s_receive_context_lvalues", name_prefix));

  TrackedBValue use_context_in = last_iter_broke_in;

  TrackedBValue lvalue_conditions_tuple = context().fb->Select(
      use_context_in, received_lvalue_conds, lvalue_cond_value, loc,
      /*name=*/absl::StrFormat("%s__lvalue_conditions_tuple", name_prefix));

  // Deal with on_reset
  TrackedBValue on_reset_bval;

  if (generated_func.uses_on_reset) {
    // received_on_reset is only valid in the first iteration, but that's okay
    // as use_context_in will always be 0 in subsequent iterations.
    on_reset_bval = pb.And(use_context_in, received_on_reset, loc);
  } else {
    on_reset_bval = pb.Literal(xls::UBits(0, 1), loc);
  }

  // Add selects for changed context variables
  TrackedBValue selected_context;

  {
    const uint64_t total_context_values =
        context_cvars_struct_ctype->fields().size();

    std::vector<TrackedBValue> context_values;
    context_values.resize(total_context_values, TrackedBValue());

    for (const clang::NamedDecl* decl : variable_fields_order) {
      if (!context_field_indices.contains(decl)) {
        continue;
      }

      const uint64_t context_field_idx = context_field_indices.at(decl);

      if (context_out_field_indices.contains(decl)) {
        const uint64_t context_out_field_idx =
            context_out_field_indices.at(decl);
        context_values[context_field_idx] =
            GetStructFieldXLS(received_context, context_out_field_idx,
                              *context_out_cvars_struct_ctype, loc);
      } else {
        auto field_type =
            context_cvars_struct_ctype->fields().at(context_field_idx)->type();
        XLS_ASSIGN_OR_RETURN(context_values[context_field_idx],
                             CreateDefaultValue(field_type, loc));
      }
    }

    // Use context in vs state elements flag
    for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
      if (!context_field_indices.contains(decl)) {
        continue;
      }
      const uint64_t field_idx = context_field_indices.at(decl);
      CHECK_LT(field_idx, context_values.size());
      TrackedBValue context_val = context_values.at(field_idx);
      TrackedBValue prev_state_val = state_reads_by_decl.at(decl);

      TrackedBValue selected_val =
          pb.Select(use_context_in, context_val, prev_state_val, loc);
      context_values[field_idx] = selected_val;
    }
    selected_context =
        MakeStructXLS(context_values, *context_cvars_struct_ctype, loc);
  }

  for (const IOOp& op : generated_func.io_ops) {
    if (op.op == OpType::kTrace) {
      continue;
    }
    if (op.channel->generated.has_value()) {
      continue;
    }
    CHECK(io_test_mode_ ||
          external_channels_by_internal_channel_.contains(op.channel));
  }

  // Invoke loop over IOs
  prepared.xls_func = &generated_func;
  prepared.args.push_back(selected_context);
  prepared.args.push_back(lvalue_conditions_tuple);
  prepared.args.push_back(on_reset_bval);
  prepared.orig_token = token;
  prepared.token = prepared.orig_token;

  TrackedBValue save_full_condition = context().full_condition;

  PushContextGuard pb_guard_block(*this, loc);

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<GeneratedFunction> dummy_top_func,
      GenerateIRBlockPrepare(prepared, pb,
                             /*next_return_index=*/extra_return_count,
                             /*this_type=*/nullptr,
                             /*this_decl=*/nullptr,
                             /*top_decls=*/{},
                             /*caller_sub_function=*/nullptr, loc));

  context().in_pipelined_for_body = true;

  // GenerateIRBlockPrepare resets the context
  if (in_fsm) {
    context().full_condition = save_full_condition;
  }
  XLS_ASSIGN_OR_RETURN(
      GenerateFSMInvocationReturn fsm_ret,
      GenerateOldFSMInvocation(prepared, pb, nesting_level, loc));
  XLSCC_CHECK(
      fsm_ret.return_value.valid() && fsm_ret.returns_this_activation.valid(),
      loc);

  token = prepared.token;

  TrackedBValue updated_context = pb.TupleIndex(
      fsm_ret.return_value, 0, loc,
      /*name=*/absl::StrFormat("%s_updated_context", name_prefix));
  TrackedBValue do_break = pb.TupleIndex(
      fsm_ret.return_value, 1, loc,
      /*name=*/absl::StrFormat("%s_do_break_from_func", name_prefix));
  TrackedBValue initial_loop_cond = pb.TupleIndex(
      fsm_ret.return_value, 2, loc,
      /*name=*/absl::StrFormat("%s_initial_loop_cond", name_prefix));

  if (in_fsm) {
    do_break =
        pb.And(do_break, fsm_ret.returns_this_activation, loc,
               /*name=*/absl::StrFormat("%s_do_break_with_fsm", name_prefix));
  }

  if (prepared.contains_fsm && !in_fsm) {
    // The context send and receive are special cased, not generated by
    // GenerateInvokeWithIO(), so they don't get added to states.
    return absl::UnimplementedError(ErrorMessage(
        loc,
        "Pipelined loops with FSMs nested in pipelined loops without FSMs"));
  }

  TrackedBValue update_state_condition;

  if (in_fsm) {
    update_state_condition = pb.And(
        in_state_condition, fsm_ret.returns_this_activation, loc,
        /*name=*/absl::StrFormat("%s_update_state_condition", name_prefix));
  } else {
    update_state_condition = pb.Literal(
        xls::UBits(1, 1), loc,
        absl::StrFormat("%s_default_update_state_cond", name_prefix));
  }

  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      next_state_values;

  next_state_values.insert(
      {last_iter_broke_state,
       NextStateValue{.value =
                          pb.Select(update_state_condition,
                                    /*on_true=*/do_break,
                                    /*on_false=*/last_iter_broke_in, loc)}});

  next_state_values.insert(
      {lvalue_cond_state, NextStateValue{.value = lvalue_conditions_tuple}});

  TrackedBValue update_state_elements = update_state_condition;

  if (in_fsm && (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl)) {
    TrackedBValue literal_1 = pb.Literal(xls::UBits(1, 1), loc);
    token = pb.Trace(
        token, literal_1,
        /*args=*/
        {in_state_condition, fsm_ret.returns_this_activation,
         update_state_condition, update_state_elements, do_break,
         use_context_in, last_iter_broke_in},
        absl::StrFormat("-- %s in_state {:u} fsm_ret {:u} update_st {:u} "
                        "update_elems {:u} do_break {:u} use_context_in {:u} "
                        "last_iter_broke_in {:u}",
                        name_prefix),
        /*verbosity=*/0, loc);
  }

  std::vector<TrackedBValue> out_tuple_values;
  out_tuple_values.resize(context_in_field_indices.size());
  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    if (!context_field_indices.contains(decl)) {
      continue;
    }
    const uint64_t field_idx = context_field_indices.at(decl);
    TrackedBValue val = GetStructFieldXLS(updated_context, field_idx,
                                          *context_cvars_struct_ctype, loc);

    NextStateValue next_state_value = {
        .priority = nesting_level, .extra_label = name_prefix, .value = val};
    TrackedBValue out_bval = val;

    if (in_fsm) {
      // Could add loop condition here.. but it's FSM only
      TrackedBValue guarded_update_state_elements = pb.And(
          update_state_elements, initial_loop_cond, loc, /*name=*/
          absl::StrFormat("%s_guarded_update_state_elements", name_prefix));

      next_state_value.condition = guarded_update_state_elements;

      out_bval =
          pb.Select(update_state_elements,
                    /*on_true=*/val,
                    /*on_false=*/state_reads_by_decl.at(decl), loc, /*name=*/
                    absl::StrFormat("%s_%s_out_val", name_prefix,
                                    decl->getNameAsString()));
    }

    next_state_values.insert(
        {prepared.state_element_for_variable[decl], next_state_value});

    if (context_in_field_indices.contains(decl)) {
      out_tuple_values[context_in_field_indices.at(decl)] = out_bval;
    }
  }

  TrackedBValue out_tuple =
      MakeStructXLS(out_tuple_values, *context_in_cvars_struct_ctype, loc);

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    CHECK(context().fb == &pb);

    TrackedBValue ret_next =
        pb.TupleIndex(fsm_ret.return_value,
                      prepared.return_index_for_static.at(namedecl), loc,
                      /*name=*/
                      absl::StrFormat("%s_fsm_ret_static_%s", name_prefix,
                                      namedecl->getNameAsString()));

    next_state_values.insert(
        {prepared.state_element_for_variable.at(namedecl),
         NextStateValue{.priority = nesting_level,
                        .extra_label = name_prefix,
                        .value = ret_next,
                        .condition = update_state_elements}});
  }

  for (const auto& [state_elem, bval] : fsm_ret.extra_next_state_values) {
    next_state_values.insert({state_elem, bval});
  }

  // Update state elements map from outer scope
  if (state_element_for_variable != nullptr) {
    for (const auto& [decl, param] : prepared.state_element_for_variable) {
      // Can't re-use state elements that are fed into context output,
      // as the context output must be kept steady outside of the state
      // containing the loop.
      if (context_in_field_indices.contains(decl)) {
        continue;
      }
      (*state_element_for_variable)[decl] = param;
    }
  }

  return PipelinedLoopContentsReturn{
      .token_out = token,
      .do_break = do_break,
      .first_iter = use_context_in,
      .out_tuple = out_tuple,
      .extra_next_state_values = next_state_values};
}

absl::Status Translator::CheckInitIntervalValidity(int initiation_interval_arg,
                                                   const xls::SourceInfo& loc) {
  if (initiation_interval_arg != 1) {
    std::string message = WarningMessage(
        loc,
        "Only initiation interval 1 supported, %i requested, defaulting to 1",
        initiation_interval_arg);
    if (error_on_init_interval_) {
      return absl::UnimplementedError(message);
    }
    LOG(WARNING) << message;
  }
  return absl::OkStatus();
}

}  // namespace xlscc
