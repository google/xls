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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "external/z3/src/api/z3_api.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace xlscc {

absl::Status Translator::GenerateIR_Loop(
    bool always_first_iter, const clang::Stmt* init,
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

  bool have_asap_intrinsic = false;

  bool have_relevant_intrinsic = false;
  bool intrinsic_unroll = false;

  XLS_ASSIGN_OR_RETURN(const clang::CallExpr* intrinsic_call,
                       FindIntrinsicCall(presumed_loc));
  if (intrinsic_call != nullptr) {
    const std::string& intrinsic_name =
        intrinsic_call->getDirectCallee()->getNameAsString();

    if (intrinsic_name == "__xlscc_pipeline") {
      have_relevant_intrinsic = true;
      intrinsic_unroll = false;
    } else if (intrinsic_name == "__xlscc_unroll") {
      have_relevant_intrinsic = true;
      intrinsic_unroll = true;
    } else if (intrinsic_name == "__xlscc_asap") {
      have_relevant_intrinsic = false;
      have_asap_intrinsic = true;
    }
  }

  XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(presumed_loc));

  bool have_relevant_pragma =
      (pragma.type() == Pragma_Unroll || pragma.type() == Pragma_InitInterval);

  if (have_relevant_intrinsic && have_relevant_pragma) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc,
                     "Have both an __xlscc_ intrinsic and a #pragma directive, "
                     "don't know what to do"));
  }

  bool do_unroll = false;

  if ((have_relevant_intrinsic && intrinsic_unroll) ||
      (pragma.type() == Pragma_Unroll) || context().for_loops_default_unroll) {
    do_unroll = true;
  }

  if (do_unroll) {
    return GenerateIR_UnrolledLoop(always_first_iter, init, cond_expr, inc,
                                   body, ctx, loc);
  }

  int64_t init_interval = -1;

  if (have_relevant_intrinsic) {
    XLSCC_CHECK(!intrinsic_unroll, loc);
    XLSCC_CHECK_EQ(intrinsic_call->getNumArgs(), 1, loc);
    XLS_ASSIGN_OR_RETURN(init_interval,
                         EvaluateInt64(*intrinsic_call->getArg(0), ctx, loc));
  } else if (have_relevant_pragma) {
    XLSCC_CHECK(pragma.type() == Pragma_InitInterval, loc);
    init_interval = pragma.int_argument();
  }

  if (have_relevant_intrinsic || have_relevant_pragma) {
    if (init_interval <= 0) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc, "Invalid initiation interval %i", init_interval));
    }
  }

  // Pipelined loops can inherit their initiation interval from enclosing
  // loops, so they can be allowed not to have a #pragma.
  if (init_interval < 0) {
    CHECK(!context().in_pipelined_for_body ||
          (context().outer_pipelined_loop_init_interval > 0));
    init_interval = context().outer_pipelined_loop_init_interval;
  }
  if (init_interval <= 0) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "For loop missing #pragma or __xlscc_ intrinsic"));
  }

  // Pipelined do-while
  return GenerateIR_PipelinedLoop(always_first_iter, init, cond_expr, inc, body,
                                  init_interval, have_asap_intrinsic, ctx, loc);
}

absl::Status Translator::GenerateIR_UnrolledLoop(bool always_first_iter,
                                                 const clang::Stmt* init,
                                                 const clang::Expr* cond_expr,
                                                 const clang::Stmt* inc,
                                                 const clang::Stmt* body,
                                                 clang::ASTContext& ctx,
                                                 const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<xls::solvers::z3::IrTranslator> z3_translator_parent,
      xls::solvers::z3::IrTranslator::CreateAndTranslate(
          /*source=*/nullptr,
          /*allow_unsupported=*/false));

  Z3_solver solver =
      xls::solvers::z3::CreateSolver(z3_translator_parent->ctx(), 1);

  class SolverDeref {
   public:
    SolverDeref(Z3_context ctx, Z3_solver solver)
        : ctx_(ctx), solver_(solver) {}
    ~SolverDeref() { Z3_solver_dec_ref(ctx_, solver_); }

   private:
    Z3_context ctx_;
    Z3_solver solver_;
  };

  // Generate the declaration within a private context
  PushContextGuard for_init_guard(*this, loc);
  context().propagate_break_up = false;
  context().propagate_continue_up = false;
  context().in_for_body = true;
  context().in_switch_body = false;

  if (init != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(init, ctx));
  }

  // Loop unrolling causes duplicate NamedDecls which fail the soundness
  // check. Reset the known set before each iteration.
  auto saved_check_ids = unique_decl_ids_;

  absl::Duration slowest_iter = absl::ZeroDuration();

  for (int64_t nIters = 0;; ++nIters) {
    const bool first_iter = nIters == 0;
    const bool always_this_iter = always_first_iter && first_iter;

    xls::Stopwatch stopwatch;

    unique_decl_ids_ = saved_check_ids;

    if (nIters > max_unroll_iters_) {
      return absl::ResourceExhaustedError(
          ErrorMessage(loc, "Loop unrolling broke at maximum %i iterations",
                       max_unroll_iters_));
    }
    if (nIters == warn_unroll_iters_) {
      LOG(WARNING) << ErrorMessage(
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
      context().or_condition_util(
          context().fb->Not(cond_expr_cval.rvalue(), loc),
          context().relative_break_condition, loc);
      XLS_RETURN_IF_ERROR(and_condition(cond_expr_cval.rvalue(), loc));
    }

    {
      // We use the relative condition so that returns also stop unrolling
      XLS_ASSIGN_OR_RETURN(bool condition_must_be_false,
                           BitMustBe(false, context().relative_condition,
                                     solver, z3_translator_parent->ctx(), loc));
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
    if (elapsed_time > absl::Seconds(0.1) && elapsed_time > slowest_iter) {
      LOG(WARNING) << ErrorMessage(loc, "Slow loop unrolling iteration %i: %v",
                                   nIters, elapsed_time);
      slowest_iter = elapsed_time;
    }
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
    std::vector<xls::BValue>* lvalue_conditions, const xls::SourceInfo& loc) {
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
    xls::BValue lvalue_conditions_tuple, const xls::SourceInfo& loc,
    int64_t* at_index) {
  if (outer_lvalue == nullptr) {
    return nullptr;
  }
  if (!outer_lvalue->get_compounds().empty()) {
    absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compounds;
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
  xls::BValue translated_condition =
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

absl::Status Translator::GenerateIR_PipelinedLoop(
    bool always_first_iter, const clang::Stmt* init,
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t initiation_interval_arg,
    bool schedule_asap, clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  XLS_RETURN_IF_ERROR(CheckInitIntervalValidity(initiation_interval_arg, loc));

  const TranslationContext& outer_context = context();

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
  xls::BValue lvalue_conditions_tuple;
  {
    std::vector<xls::BValue> full_context_tuple_values;
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

    std::vector<xls::BValue> lvalue_conditions;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      // Don't mark access
      // These are handled below based on what's really used in the loop body
      // const CValue& cvalue = context().variables.at(decl);
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

    lvalue_conditions_tuple = context().fb->Tuple(lvalue_conditions, loc,
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
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<LValue>>
      lvalues_out;
  bool uses_on_reset = false;
  XLS_ASSIGN_OR_RETURN(
      PipelinedLoopSubProc sub_proc,
      GenerateIR_PipelinedLoopBody(
          cond_expr, inc, body, initiation_interval_arg, ctx, name_prefix,
          context_struct_xls_type, context_lvals_xls_type,
          context_cvars_struct_ctype, &lvalues_out, context_field_indices,
          variable_fields_order, &uses_on_reset, loc));

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
    std::vector<xls::BValue> context_out_tuple_values;

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
        const xls::BValue& rvalue = outer_context.variables.at(decl).rvalue();

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
    xls::BValue outer_on_reset_value =
        context().fb->Literal(xls::UBits(0, 1), loc);

    // Must match if(uses_on_reset) below
    context_tuple_out = CValue(
        context().fb->Tuple({outer_on_reset_value, context_struct_out.rvalue(),
                             lvalue_conditions_tuple},
                            loc, /*name=*/"context_out_tuple_inner"),
        context_tuple_type);
  }

  // Create synthetic channels and IO ops
  xls::Type* context_out_xls_type = context_tuple_out.rvalue().GetType();

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

  XLS_ASSIGN_OR_RETURN(xls::Type * context_in_struct_xls_type,
                       TranslateTypeToXLS(context_in_cvars_struct_ctype, loc));

  // Pick a construct to correlate the channels for this construct
  const clang::Stmt* identify_channels_stmt = body;
  XLSCC_CHECK(identify_channels_stmt != nullptr, loc);

  // Create context channels
  IOChannel* context_out_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_out", name_prefix);
    xls::Channel* xls_channel = nullptr;
    if (!generate_fsms_for_pipelined_loops_) {
      XLS_ASSIGN_OR_RETURN(
          xls_channel,
          package_->CreateStreamingChannel(
              ch_name, xls::ChannelOps::kSendReceive, context_out_xls_type,
              /*initial_values=*/{},
              /*fifo_config=*/
              xls::FifoConfig(/*depth=*/0, /*bypass=*/true,
                              /*register_push_outputs=*/false,
                              /*register_pop_outputs=*/false),
              xls::FlowControl::kReadyValid));
    }
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
    if (!generate_fsms_for_pipelined_loops_) {
      XLS_ASSIGN_OR_RETURN(xls_channel,
                           package_->CreateStreamingChannel(
                               ch_name, xls::ChannelOps::kSendReceive,
                               context_in_struct_xls_type,
                               /*initial_values=*/{},
                               /*fifo_config=*/
                               xls::FifoConfig(/*depth=*/0, /*bypass=*/true,
                                               /*register_push_outputs=*/false,
                                               /*register_pop_outputs=*/false),
                               xls::FlowControl::kReadyValid));
    }
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

  // TODO(seanhaskell): Move this to GenerateIR_Block() for pipelined loops
  // with multiple different sets of IO ops
  if (!generate_fsms_for_pipelined_loops_) {
    XLS_RETURN_IF_ERROR(GenerateIR_PipelinedLoopProc(sub_proc));
  }

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
    std::vector<xls::BValue> sp = {context_tuple_out.rvalue(),
                                   context().full_condition_bval(loc)};
    op.ret_value =
        context().fb->Tuple(sp, loc, /*name=*/"context_out_send_tup");
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
  xls::BValue context_tuple_recvd = ctx_in_op_ptr->input_value.rvalue();
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
    const clang::Stmt* body, int64_t init_interval, clang::ASTContext& ctx,
    std::string_view name_prefix, xls::Type* context_struct_xls_type,
    xls::Type* context_lvals_xls_type,
    const std::shared_ptr<CStructType>& context_cvars_struct_ctype,
    absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<LValue>>*
        lvalues_out,
    const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
        context_field_indices,
    const std::vector<const clang::NamedDecl*>& variable_fields_order,
    bool* uses_on_reset, const xls::SourceInfo& loc) {
  std::vector<std::pair<const clang::NamedDecl*, int64_t>>
      vars_accessed_in_body;
  std::vector<const clang::NamedDecl*> vars_changed_in_body;

  GeneratedFunction& enclosing_func = *context().sf;

  // Generate body function
  auto generated_func = std::make_unique<GeneratedFunction>();
  CHECK_NE(context().sf, nullptr);
  CHECK_NE(context().sf->clang_decl, nullptr);
  generated_func->clang_decl = context().sf->clang_decl;
  uint64_t extra_return_count = 0;
  {
    // Set up IR generation
    xls::FunctionBuilder body_builder(absl::StrFormat("%s_func", name_prefix),
                                      package_);

    xls::BValue context_struct_val =
        body_builder.Param(absl::StrFormat("%s_context_vars", name_prefix),
                           context_struct_xls_type, loc);
    xls::BValue context_lvalues_val =
        body_builder.Param(absl::StrFormat("%s_context_lvals", name_prefix),
                           context_lvals_xls_type, loc);
    xls::BValue context_on_reset_val =
        body_builder.Param(absl::StrFormat("%s_on_reset", name_prefix),
                           package_->GetBitsType(1), loc);

    TranslationContext& prev_context = context();
    PushContextGuard context_guard(*this, loc);

    context() = TranslationContext();
    context().propagate_up = false;
    context().fb = absl::implicit_cast<xls::BuilderBase*>(&body_builder);
    context().sf = generated_func.get();
    context().ast_context = prev_context.ast_context;
    context().in_pipelined_for_body = true;
    context().outer_pipelined_loop_init_interval = init_interval;

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
    absl::flat_hash_map<const clang::NamedDecl*, CValue> prev_vars;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue& outer_value = prev_context.variables.at(decl);
      xls::BValue param_bval;
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

    xls::BValue do_break = context().fb->Literal(xls::UBits(0, 1));

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
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, ctx));

      // break_condition is the assignment condition
      if (context().relative_break_condition.valid()) {
        xls::BValue break_cond = context().relative_break_condition;
        do_break = context().fb->Or(do_break, break_cond, loc);
      }
    }

    // Increment
    // Break condition skips increment
    if (inc != nullptr) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);
      XLS_RETURN_IF_ERROR(and_condition(context().fb->Not(do_break, loc), loc));
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(inc, ctx));
    }

    // Check condition
    if (cond_expr != nullptr) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);

      XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
      CHECK(cond_cval.type()->Is<CBoolType>());
      xls::BValue break_on_cond_val = context().fb->Not(cond_cval.rvalue());

      do_break = context().fb->Or(do_break, break_on_cond_val, loc);
    }

    // Context out
    const uint64_t total_context_values =
        context_cvars_struct_ctype->fields().size();

    std::vector<xls::BValue> tuple_values;
    tuple_values.resize(total_context_values);
    for (const clang::NamedDecl* decl : variable_fields_order) {
      if (!context_field_indices.contains(decl)) {
        continue;
      }
      const uint64_t field_idx = context_field_indices.at(decl);
      tuple_values[field_idx] = context().variables.at(decl).rvalue();
    }

    xls::BValue ret_ctx =
        MakeStructXLS(tuple_values, *context_cvars_struct_ctype, loc);
    std::vector<xls::BValue> return_bvals = {ret_ctx, do_break};

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

    xls::BValue ret_val = MakeFlexTuple(return_bvals, loc);
    generated_func->return_value_count = return_bvals.size();
    XLS_ASSIGN_OR_RETURN(generated_func->xls_func,
                         body_builder.BuildWithReturnValue(ret_val));

    // Analyze context variables changed
    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue prev_bval = prev_vars.at(decl);
      const CValue curr_val = context().variables.at(decl);
      if (prev_bval.rvalue().node() != curr_val.rvalue().node() ||
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

    // In non-FSM mode, All variables accessed or changed are saved in state,
    // because a streaming channel is used for the context
    if (!generate_fsms_for_pipelined_loops_) {
      for (const std::pair<const clang::NamedDecl*, int64_t>& accessed :
           vars_accessed_in_body) {
        vars_to_save_between_iters_set.insert(accessed.first);
      }
    }

    for (const clang::NamedDecl* decl : vars_to_save_between_iters_set) {
      vars_to_save_between_iters.push_back(decl);
    }

    context().sf->SortNamesDeterministically(vars_to_save_between_iters);
  }

  PipelinedLoopSubProc pipelined_loop_proc = {
      .name_prefix = name_prefix.data(),
      // context_ members are filled in by caller
      .loc = loc,

      .enclosing_func = context().sf,
      .outer_variables = context().variables,
      .context_field_indices = context_field_indices,
      .extra_return_count = extra_return_count,
      .generated_func = std::move(generated_func),
      .variable_fields_order = variable_fields_order,
      .vars_changed_in_body = vars_changed_in_body,
      .vars_accessed_in_body = vars_accessed_in_body,
      .vars_to_save_between_iters = vars_to_save_between_iters};

  return pipelined_loop_proc;
}

absl::Status Translator::GenerateIR_PipelinedLoopProc(
    const PipelinedLoopSubProc& pipelined_loop_proc) {
  const std::string& name_prefix = pipelined_loop_proc.name_prefix;
  IOChannel* context_out_channel = pipelined_loop_proc.context_out_channel;
  IOChannel* context_in_channel = pipelined_loop_proc.context_in_channel;
  const xls::SourceInfo& loc = pipelined_loop_proc.loc;

  xls::ProcBuilder pb(absl::StrFormat("%s_proc", name_prefix), package_);

  auto temp_sf = std::make_unique<GeneratedFunction>();

  PushContextGuard pb_guard(*this, loc);
  context() = TranslationContext();
  context().propagate_up = false;
  context().fb = absl::implicit_cast<xls::BuilderBase*>(&pb);
  context().in_pipelined_for_body = true;
  context().sf = temp_sf.get();

  xls::BValue token = pb.Literal(xls::Value::Token());

  xls::BValue placeholder_cond = pb.Literal(xls::UBits(1, 1));

  XLSCC_CHECK(context_out_channel->generated.has_value(), loc);
  XLSCC_CHECK_NE(context_out_channel->generated.value(), nullptr, loc);

  xls::BValue receive =
      pb.ReceiveIf(context_out_channel->generated.value(), token,
                   /*pred=*/placeholder_cond, loc,
                   /*name=*/absl::StrFormat("%s_receive_context", name_prefix));
  token = pb.TupleIndex(
      receive, 0, loc,
      /*name=*/absl::StrFormat("%s_receive_context_token", name_prefix));
  xls::BValue received_context_tuple = pb.TupleIndex(
      receive, 1, loc,
      /*name=*/absl::StrFormat("%s_receive_context_tup", name_prefix));

  XLS_ASSIGN_OR_RETURN(
      PipelinedLoopContentsReturn contents_ret,
      GenerateIR_PipelinedLoopContents(pipelined_loop_proc, pb, token,
                                       received_context_tuple,
                                       /*in_state_condition=*/xls::BValue(),
                                       /*in_fsm=*/false));

  auto* receive_node = receive.node()->As<xls::Receive>();
  bool replaced = receive_node->ReplaceOperand(
      /*old_operand=*/placeholder_cond.node(),
      /*new_operand=*/contents_ret.first_iter.node());
  XLSCC_CHECK(replaced, loc);

  token = contents_ret.token_out;

  // Send back context on break
  XLSCC_CHECK(context_in_channel->generated.has_value(), loc);
  XLSCC_CHECK_NE(context_in_channel->generated.value(), nullptr, loc);

  token = pb.SendIf(context_in_channel->generated.value(), token,
                    contents_ret.do_break, contents_ret.out_tuple, loc);

  XLS_RETURN_IF_ERROR(BuildWithNextStateValueMap(
                          pb, token, contents_ret.extra_next_state_values, loc)
                          .status());

  return absl::OkStatus();
}

absl::StatusOr<Translator::PipelinedLoopContentsReturn>
Translator::GenerateIR_PipelinedLoopContents(
    const PipelinedLoopSubProc& pipelined_loop_proc, xls::ProcBuilder& pb,
    xls::BValue token_in, xls::BValue received_context_tuple,
    xls::BValue in_state_condition, bool in_fsm,
    absl::flat_hash_map<const clang::NamedDecl*, xls::Param*>*
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
  const GeneratedFunction& generated_func = *pipelined_loop_proc.generated_func;

  const std::vector<const clang::NamedDecl*>& vars_to_save_between_iters =
      pipelined_loop_proc.vars_to_save_between_iters;

  // Generate body proc
  const std::string& name_prefix = pipelined_loop_proc.name_prefix;

  XLSCC_CHECK(!in_fsm || in_state_condition.valid(), loc);

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
  xls::BValue last_iter_broke_in =
      pb.StateElement(absl::StrFormat("%s__last_iter_broke", name_prefix),
                      xls::Value(xls::UBits(1, 1)));

  XLS_ASSIGN_OR_RETURN(
      xls::Value default_lval_conds,
      CreateDefaultRawValue(context_out_lval_conds_ctype, loc));
  xls::BValue lvalue_cond_state =
      pb.StateElement(absl::StrFormat("%s__lvalue_conditions", name_prefix),
                      default_lval_conds);

  absl::flat_hash_map<const clang::NamedDecl*, xls::BValue>
      state_elements_by_decl;

  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    if (!context_field_indices.contains(decl)) {
      continue;
    }
    // Only create a state element if one doesn't already exist
    if (!prepared.state_element_for_variable.contains(decl)) {
      const CValue& prev_value = pipelined_loop_proc.outer_variables.at(decl);
      XLS_ASSIGN_OR_RETURN(
          xls::Value def,
          CreateDefaultRawValue(prev_value.type(), GetLoc(*decl)));

      xls::BValue state_elem_bval = pb.StateElement(
          absl::StrFormat("%s_%s", name_prefix, decl->getNameAsString()), def);

      state_elements_by_decl[decl] = state_elem_bval;
      prepared.state_element_for_variable[decl] =
          state_elem_bval.node()->As<xls::Param>();
    } else {
      xls::Param* state_elem = prepared.state_element_for_variable.at(decl);
      state_elements_by_decl[decl] = xls::BValue(state_elem, &pb);
    }
  }

  // For utility functions like MakeStructXls()
  PushContextGuard pb_guard(*this, in_state_condition, loc);

  xls::BValue token = token_in;

  xls::BValue received_on_reset = pb.TupleIndex(
      received_context_tuple, 0, loc,
      /*name=*/absl::StrFormat("%s_receive_on_reset", name_prefix));
  xls::BValue received_context = pb.TupleIndex(
      received_context_tuple, 1, loc,
      /*name=*/absl::StrFormat("%s_receive_context_data", name_prefix));

  xls::BValue received_lvalue_conds = pb.TupleIndex(
      received_context_tuple, 2, loc,
      /*name=*/absl::StrFormat("%s_receive_context_lvalues", name_prefix));

  xls::BValue use_context_in = last_iter_broke_in;

  xls::BValue lvalue_conditions_tuple = context().fb->Select(
      use_context_in, received_lvalue_conds, lvalue_cond_state, loc,
      /*name=*/absl::StrFormat("%s__lvalue_conditions_tuple", name_prefix));

  // Deal with on_reset
  xls::BValue on_reset_bval;

  if (generated_func.uses_on_reset) {
    // received_on_reset is only valid in the first iteration, but that's okay
    // as use_context_in will always be 0 in subsequent iterations.
    on_reset_bval = pb.And(use_context_in, received_on_reset, loc);
  } else {
    on_reset_bval = pb.Literal(xls::UBits(0, 1), loc);
  }

  // Add selects for changed context variables
  xls::BValue selected_context;

  {
    const uint64_t total_context_values =
        context_cvars_struct_ctype->fields().size();

    std::vector<xls::BValue> context_values;
    context_values.resize(total_context_values, xls::BValue());

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
      xls::BValue context_val = context_values.at(field_idx);
      xls::BValue prev_state_val = state_elements_by_decl.at(decl);

      xls::BValue selected_val =
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

  xls::BValue save_full_condition = context().full_condition;

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<GeneratedFunction> dummy_top_func,
      GenerateIRBlockPrepare(prepared, pb,
                             /*next_return_index=*/extra_return_count,
                             /*this_type=*/nullptr,
                             /*this_decl=*/nullptr,
                             /*top_decls=*/{}, loc));

  context().in_pipelined_for_body = true;

  // GenerateIRBlockPrepare resets the context
  if (in_fsm) {
    context().full_condition = save_full_condition;
  }
  XLS_ASSIGN_OR_RETURN(GenerateFSMInvocationReturn fsm_ret,
                       GenerateFSMInvocation(prepared, pb, nesting_level, loc));
  XLSCC_CHECK(
      fsm_ret.return_value.valid() && fsm_ret.returns_this_activation.valid(),
      loc);

  token = prepared.token;

  xls::BValue updated_context = pb.TupleIndex(
      fsm_ret.return_value, 0, loc,
      /*name=*/absl::StrFormat("%s_updated_context", name_prefix));
  xls::BValue do_break = pb.TupleIndex(
      fsm_ret.return_value, 1, loc,
      /*name=*/absl::StrFormat("%s_do_break_from_func", name_prefix));

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

  xls::BValue update_state_condition;

  if (in_fsm) {
    update_state_condition = pb.And(
        in_state_condition, fsm_ret.returns_this_activation, loc,
        /*name=*/absl::StrFormat("%s_update_state_condition", name_prefix));
  } else {
    update_state_condition = pb.Literal(
        xls::UBits(1, 1), loc,
        absl::StrFormat("%s_default_update_state_cond", name_prefix));
  }

  absl::btree_multimap<const xls::Param*, NextStateValue> next_state_values;

  next_state_values.insert(
      {last_iter_broke_in.node()->As<xls::Param>(),
       NextStateValue{.value =
                          pb.Select(update_state_condition,
                                    /*on_true=*/do_break,
                                    /*on_false=*/last_iter_broke_in, loc)}});

  next_state_values.insert({lvalue_cond_state.node()->As<xls::Param>(),
                            NextStateValue{.value = lvalue_conditions_tuple}});

  xls::BValue update_state_elements = update_state_condition;

  if (in_fsm && (debug_ir_trace_flags_ & DebugIrTraceFlags_LoopControl)) {
    xls::BValue literal_1 = pb.Literal(xls::UBits(1, 1), loc);
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

  std::vector<xls::BValue> out_tuple_values;
  out_tuple_values.resize(context_in_field_indices.size());
  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    if (!context_field_indices.contains(decl)) {
      continue;
    }
    const uint64_t field_idx = context_field_indices.at(decl);
    xls::BValue val = GetStructFieldXLS(updated_context, field_idx,
                                        *context_cvars_struct_ctype, loc);

    NextStateValue next_state_value = {
        .priority = nesting_level, .extra_label = name_prefix, .value = val};
    xls::BValue out_bval = val;

    if (in_fsm) {
      next_state_value.condition = update_state_elements;
      out_bval =
          pb.Select(update_state_elements,
                    /*on_true=*/val,
                    /*on_false=*/state_elements_by_decl.at(decl), loc, /*name=*/
                    absl::StrFormat("%s_%s_out_val", name_prefix,
                                    decl->getNameAsString()));
    }

    next_state_values.insert(
        {state_elements_by_decl.at(decl).node()->As<xls::Param>(),
         next_state_value});

    if (context_in_field_indices.contains(decl)) {
      out_tuple_values[context_in_field_indices.at(decl)] = out_bval;
    }
  }

  xls::BValue out_tuple =
      MakeStructXLS(out_tuple_values, *context_in_cvars_struct_ctype, loc);

  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    CHECK(context().fb == &pb);

    xls::BValue ret_next =
        pb.TupleIndex(fsm_ret.return_value,
                      prepared.return_index_for_static.at(namedecl), loc,
                      /*name=*/
                      absl::StrFormat("%s_fsm_ret_static_%s", name_prefix,
                                      namedecl->getNameAsString()));

    xls::BValue state_elem_bval(
        prepared.state_element_for_variable.at(namedecl), &pb);

    next_state_values.insert(
        {state_elem_bval.node()->As<xls::Param>(),
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

}  // namespace xlscc
