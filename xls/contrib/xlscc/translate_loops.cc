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

#include <memory>

#include "absl/status/status.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Returns monotonically increasing time in seconds
double doubletime() {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return tv.tv_sec + static_cast<double>(tv.tv_usec) / 1000000.0;
}

}  // namespace

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
  XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(presumed_loc));
  if (pragma.type() == Pragma_Unroll || context().for_loops_default_unroll) {
    return GenerateIR_UnrolledLoop(always_first_iter, init, cond_expr, inc,
                                   body, ctx, loc);
  }
  // Pipelined loops can inherit their initiation interval from enclosing
  // loops, so they can be allowed not to have a #pragma.
  int init_interval = pragma.int_argument();
  // Pragma might not be null, because labels get searched backwards
  if (pragma.type() != Pragma_InitInterval) {
    XLS_CHECK(!context().in_pipelined_for_body ||
              (context().outer_pipelined_loop_init_interval > 0));
    init_interval = context().outer_pipelined_loop_init_interval;
  }
  if (init_interval <= 0) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "For loop missing #pragma"));
  }

  // Pipelined do-while
  return GenerateIR_PipelinedLoop(always_first_iter, init, cond_expr, inc, body,
                                  init_interval, ctx, loc);
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
          /*allow_unsupported=*/true));

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

  double slowest_iter = 0;

  for (int64_t nIters = 0;; ++nIters) {
    const bool first_iter = nIters == 0;
    const bool always_this_iter = always_first_iter && first_iter;

    const double iter_start = doubletime();

    unique_decl_ids_ = saved_check_ids;

    if (nIters >= max_unroll_iters_) {
      return absl::ResourceExhaustedError(
          ErrorMessage(loc, "Loop unrolling broke at maximum %i iterations",
                       max_unroll_iters_));
    }
    if (nIters == warn_unroll_iters_) {
      XLS_LOG(WARNING) << ErrorMessage(
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
      XLS_CHECK(cond_expr_cval.type()->Is<CBoolType>());
      context().or_condition_util(
          context().fb->Not(cond_expr_cval.rvalue(), loc),
          context().relative_break_condition, loc);
      XLS_RETURN_IF_ERROR(and_condition(cond_expr_cval.rvalue(), loc));
    }

    // Generate body
    {
      PushContextGuard for_body_guard(*this, loc);
      context().propagate_break_up = true;
      context().propagate_continue_up = false;

      // Check condition first
      if (context().relative_break_condition.valid() && !always_this_iter) {
        // Simplify break logic in easy ways;
        // Z3 fails to solve some cases without this.
        XLS_RETURN_IF_ERROR(
            ShortCircuitBVal(context().relative_break_condition, loc));

        // Use Z3 to check if another loop iteration is possible.
        xls::BValue not_break =
            context().fb->Not(context().relative_break_condition);

        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<xls::solvers::z3::IrTranslator> z3_translator,
            xls::solvers::z3::IrTranslator::CreateAndTranslate(
                /*ctx=*/z3_translator_parent->ctx(),
                /*source=*/not_break.node(),
                /*allow_unsupported=*/true));

        XLS_ASSIGN_OR_RETURN(
            Z3_lbool result,
            IsBitSatisfiable(not_break.node(), solver, *z3_translator));

        // No combination of variables can satisfy !break condition.
        if (result == Z3_L_FALSE) {
          break;
        }
      }

      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, ctx));
    }

    // Generate increment
    // Outside of body guard because continue would skip.
    if (inc != nullptr) {
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(inc, ctx));
    }
    // Print slow unrolling warning
    const double iter_end = doubletime();
    const double iter_seconds = iter_end - iter_start;

    if (iter_seconds > 0.1 && iter_seconds > slowest_iter) {
      XLS_LOG(WARNING) << ErrorMessage(
          loc, "Slow loop unrolling iteration %i: %fms", nIters, iter_seconds);
      slowest_iter = iter_seconds;
    }
  }

  return absl::OkStatus();
}

bool Translator::LValueContainsOnlyChannels(std::shared_ptr<LValue> lvalue) {
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

absl::Status Translator::GenerateIR_PipelinedLoop(
    bool always_first_iter, const clang::Stmt* init,
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t initiation_interval_arg,
    clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  XLS_RETURN_IF_ERROR(CheckInitIntervalValidity(initiation_interval_arg, loc));

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
    XLS_CHECK(cond_cval.type()->Is<CBoolType>());

    XLS_RETURN_IF_ERROR(and_condition(cond_cval.rvalue(), loc));
  }

  // Pack context tuple
  CValue context_tuple_out;
  std::shared_ptr<CStructType> context_struct_type;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t> variable_field_indices;
  std::vector<const clang::NamedDecl*> variable_fields_order;
  {
    std::vector<std::shared_ptr<CField>> fields;
    std::vector<xls::BValue> tuple_values;

    // Create a deterministic field order
    for (const auto& [decl, _] : context().variables) {
      const CValue& cvalue = context().variables.at(decl);
      if (dynamic_cast<CChannelType*>(cvalue.type().get()) != nullptr) {
        continue;
      }
      XLS_CHECK(context().sf->declaration_order_by_name_.contains(decl));
      variable_fields_order.push_back(decl);
    }

    context().sf->SortNamesDeterministically(variable_fields_order);

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue& cvalue = context().variables.at(decl);

      if (!LValueContainsOnlyChannels(cvalue.lvalue())) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "LValue translation for pipelined loops not supported yet"));
      }
      XLS_CHECK(cvalue.rvalue().valid());
      const uint64_t field_idx = tuple_values.size();
      variable_field_indices[decl] = field_idx;
      tuple_values.push_back(cvalue.rvalue());
      auto field_ptr = std::make_shared<CField>(decl, field_idx, cvalue.type());
      fields.push_back(field_ptr);
    }

    context_struct_type = std::make_shared<CStructType>(
        fields, /*no_tuple=*/false, /*synthetic_int=*/false);
    context_tuple_out =
        CValue(MakeStructXLS(tuple_values, *context_struct_type, loc),
               context_struct_type);
  }

  // Create synthetic channels and IO ops
  xls::Type* context_xls_type = context_tuple_out.rvalue().GetType();

  const std::string name_prefix =
      absl::StrFormat("__for_%i", next_for_number_++);

  IOChannel* context_out_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_out", name_prefix);
    XLS_ASSIGN_OR_RETURN(
        xls::Channel * xls_channel,
        package_->CreateStreamingChannel(
            ch_name, xls::ChannelOps::kSendReceive, context_xls_type,
            /*initial_values=*/{}, /*fifo_depth=*/0,
            xls::FlowControl::kReadyValid));
    IOChannel new_channel;
    new_channel.item_type = context_tuple_out.type();
    new_channel.unique_name = ch_name;
    new_channel.channel_op_type = OpType::kSend;
    new_channel.generated = xls_channel;
    context().sf->io_channels.push_back(new_channel);
    context_out_channel = &context().sf->io_channels.back();
  }
  IOChannel* context_in_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_in", name_prefix);
    XLS_ASSIGN_OR_RETURN(
        xls::Channel * xls_channel,
        package_->CreateStreamingChannel(
            ch_name, xls::ChannelOps::kSendReceive, context_xls_type,
            /*initial_values=*/{}, /*fifo_depth=*/0,
            xls::FlowControl::kReadyValid));
    IOChannel new_channel;
    new_channel.item_type = context_tuple_out.type();
    new_channel.unique_name = ch_name;
    new_channel.channel_op_type = OpType::kRecv;
    new_channel.generated = xls_channel;
    context().sf->io_channels.push_back(new_channel);
    context_in_channel = &context().sf->io_channels.back();
  }

  IOOp* ctx_out_op_ptr = nullptr;
  {
    IOOp op;
    op.op = OpType::kSend;
    std::vector<xls::BValue> sp = {context_tuple_out.rvalue(),
                                   context().full_condition_bval(loc)};
    op.ret_value = context().fb->Tuple(sp, loc);
    XLS_ASSIGN_OR_RETURN(ctx_out_op_ptr,
                         AddOpToChannel(op, context_out_channel, loc));
  }

  IOOp* ctx_in_op_ptr;
  {
    IOOp op;
    op.op = OpType::kRecv;
    op.ret_value = context().full_condition_bval(loc);
    XLS_ASSIGN_OR_RETURN(ctx_in_op_ptr,
                         AddOpToChannel(op, context_in_channel, loc));
  }

  ctx_in_op_ptr->after_ops.push_back(ctx_out_op_ptr);

  // Create loop body proc
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<LValue>>
      lvalues_out;
  std::vector<const clang::NamedDecl*> vars_changed_in_body;
  XLS_RETURN_IF_ERROR(GenerateIR_PipelinedLoopBody(
      cond_expr, inc, body, initiation_interval_arg, ctx, name_prefix,
      context_out_channel, context_in_channel, context_xls_type,
      context_struct_type, &lvalues_out, variable_field_indices,
      variable_fields_order, vars_changed_in_body, loc));

  XLS_CHECK_EQ(vars_changed_in_body.size(), lvalues_out.size());

  // Unpack context tuple
  xls::BValue context_tuple_recvd = ctx_in_op_ptr->input_value.rvalue();
  {
    // Don't assign to variables that aren't changed in the loop body,
    // as this creates extra state
    for (const clang::NamedDecl* decl : vars_changed_in_body) {
      const uint64_t field_idx = variable_field_indices.at(decl);

      const CValue prev_cval = context().variables.at(decl);

      const CValue cval(GetStructFieldXLS(context_tuple_recvd, field_idx,
                                          *context_struct_type, loc),
                        prev_cval.type(), /*disable_type_check=*/false,
                        lvalues_out.at(decl));
      XLS_RETURN_IF_ERROR(Assign(decl, cval, loc));
    }
  }

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_PipelinedLoopBody(
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t init_interval, clang::ASTContext& ctx,
    std::string_view name_prefix, IOChannel* context_out_channel,
    IOChannel* context_in_channel, xls::Type* context_xls_type,
    std::shared_ptr<CStructType> context_ctype,
    absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<LValue>>*
        lvalues_out,
    const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
        variable_field_indices,
    const std::vector<const clang::NamedDecl*>& variable_fields_order,
    std::vector<const clang::NamedDecl*>& vars_changed_in_body,
    const xls::SourceInfo& loc) {
  const uint64_t total_context_values = context_ctype->fields().size();
  std::vector<const clang::NamedDecl*> vars_to_save_between_iters;

  // Generate body function
  GeneratedFunction generated_func;
  XLS_CHECK_NE(context().sf, nullptr);
  XLS_CHECK_NE(context().sf->clang_decl, nullptr);
  generated_func.clang_decl = context().sf->clang_decl;
  uint64_t extra_return_count = 0;
  {
    GeneratedFunction& enclosing_func = *context().sf;

    // Set up IR generation
    xls::FunctionBuilder body_builder(absl::StrFormat("%s_func", name_prefix),
                                      package_);

    xls::BValue context_param = body_builder.Param(
        absl::StrFormat("%s_context", name_prefix), context_xls_type, loc);

    TranslationContext& prev_context = context();
    PushContextGuard context_guard(*this, loc);

    context() = TranslationContext();
    context().propagate_up = false;

    context().fb = absl::implicit_cast<xls::BuilderBase*>(&body_builder);
    context().sf = &generated_func;
    context().in_pipelined_for_body = true;
    context().outer_pipelined_loop_init_interval = init_interval;

    absl::flat_hash_map<IOChannel*, IOChannel*> inner_channels_by_outer_channel;
    absl::flat_hash_map<IOChannel*, IOChannel*> outer_channels_by_inner_channel;

    // Inherit external channels
    for (IOChannel& enclosing_channel : enclosing_func.io_channels) {
      if (enclosing_channel.generated != nullptr) {
        continue;
      }
      // TODO(seanhaskell): Merge with AddChannel
      generated_func.io_channels.push_back(enclosing_channel);
      IOChannel* inner_channel = &generated_func.io_channels.back();
      inner_channel->total_ops = 0;

      const clang::NamedDecl* decl =
          enclosing_func.decls_by_io_channel.at(&enclosing_channel);

      generated_func.io_channels_by_decl[decl] = inner_channel;
      generated_func.decls_by_io_channel[inner_channel] = decl;

      inner_channels_by_outer_channel[&enclosing_channel] = inner_channel;
      outer_channels_by_inner_channel[inner_channel] = &enclosing_channel;

      auto channel_type = std::make_shared<CChannelType>(
          inner_channel->item_type, inner_channel->channel_op_type);

      auto lvalue = std::make_shared<LValue>(inner_channel);
      XLS_RETURN_IF_ERROR(
          DeclareVariable(decl,
                          CValue(/*rvalue=*/xls::BValue(), channel_type,
                                 /*disable_type_check=*/true, lvalue),
                          loc, /*check_unique_ids=*/false));
    }

    // Context in
    absl::flat_hash_map<const clang::NamedDecl*, CValue> prev_vars;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      const CValue& outer_value = prev_context.variables.at(decl);
      xls::BValue param_bval =
          GetStructFieldXLS(context_param, field_idx, *context_ctype, loc);

      std::shared_ptr<LValue> inner_lval;

      XLS_ASSIGN_OR_RETURN(
          inner_lval,
          TranslateLValueChannels(outer_value.lvalue(),
                                  inner_channels_by_outer_channel, loc));

      CValue prev_var(param_bval, outer_value.type(),
                      /*disable_type_check=*/false, inner_lval);

      XLS_RETURN_IF_ERROR(
          DeclareVariable(decl, prev_var, loc, /*check_unique_ids=*/false));

      prev_vars[decl] = prev_var;
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

      XLS_CHECK_GT(context().outer_pipelined_loop_init_interval, 0);

      XLS_CHECK_NE(body, nullptr);
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
      XLS_CHECK(cond_cval.type()->Is<CBoolType>());
      xls::BValue break_on_cond_val = context().fb->Not(cond_cval.rvalue());

      do_break = context().fb->Or(do_break, break_on_cond_val, loc);
    }

    // Context out
    std::vector<xls::BValue> tuple_values;
    tuple_values.resize(total_context_values);
    for (const clang::NamedDecl* decl : variable_fields_order) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      tuple_values[field_idx] = context().variables.at(decl).rvalue();
    }

    xls::BValue ret_ctx = MakeStructXLS(tuple_values, *context_ctype, loc);
    std::vector<xls::BValue> return_bvals = {ret_ctx, do_break};

    // For GenerateIRBlock_Prepare() / GenerateIOInvokes()
    extra_return_count += return_bvals.size();

    // First static returns
    for (const clang::NamedDecl* decl :
         generated_func.GetDeterministicallyOrderedStaticValues()) {
      XLS_ASSIGN_OR_RETURN(CValue value, GetIdentifier(decl, loc));
      return_bvals.push_back(value.rvalue());
    }

    // IO returns
    for (IOOp& op : generated_func.io_ops) {
      XLS_CHECK(op.ret_value.valid());
      return_bvals.push_back(op.ret_value);
    }

    xls::BValue ret_val = MakeFlexTuple(return_bvals, loc);
    generated_func.return_value_count = return_bvals.size();
    XLS_ASSIGN_OR_RETURN(generated_func.xls_func,
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

    context().sf->SortNamesDeterministically(vars_changed_in_body);

    // All variables now are saved in state, because a streaming channel
    // is used for the context
    vars_to_save_between_iters = variable_fields_order;
  }

  // Generate body proc
  xls::ProcBuilder pb(absl::StrFormat("%s_proc", name_prefix),
                      /*token_name=*/"tkn", package_);

  int64_t extra_state_count = 0;

  // Construct initial state
  pb.StateElement("__first_tick", xls::Value(xls::UBits(1, 1)));
  ++extra_state_count;

  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    const CValue& prev_value = context().variables.at(decl);
    XLS_ASSIGN_OR_RETURN(xls::Value def, CreateDefaultRawValue(
                                             prev_value.type(), GetLoc(*decl)));
    pb.StateElement(decl->getNameAsString(), def);
    ++extra_state_count;
  }

  // For utility functions like MakeStructXls()
  PushContextGuard pb_guard(*this, loc);
  context().fb = absl::implicit_cast<xls::BuilderBase*>(&pb);

  xls::BValue token = pb.GetTokenParam();

  xls::BValue first_iter_state_in = pb.GetStateParam(0);

  xls::BValue recv_condition = first_iter_state_in;
  XLS_CHECK_EQ(recv_condition.GetType()->GetFlatBitCount(), 1);

  xls::BValue receive =
      pb.ReceiveIf(context_out_channel->generated, token, recv_condition, loc);
  xls::BValue token_ctx = pb.TupleIndex(receive, 0);
  xls::BValue received_context = pb.TupleIndex(receive, 1);

  token = token_ctx;

  // Add selects for changed context variables
  xls::BValue selected_context;
  {
    std::vector<xls::BValue> context_values;
    for (uint64_t fi = 0; fi < total_context_values; ++fi) {
      context_values.push_back(
          GetStructFieldXLS(received_context, fi, *context_ctype, loc));
    }

    // After first flag
    uint64_t state_tup_idx = 1;
    for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      XLS_CHECK_LT(field_idx, context_values.size());
      xls::BValue context_val =
          GetStructFieldXLS(received_context, field_idx, *context_ctype, loc);
      xls::BValue prev_state_val = pb.GetStateParam(state_tup_idx++);
      context_values[field_idx] =
          pb.Select(first_iter_state_in, context_val, prev_state_val, loc);
    }
    selected_context = MakeStructXLS(context_values, *context_ctype, loc);
  }

  for (const IOOp& op : generated_func.io_ops) {
    if (op.channel->generated != nullptr) {
      continue;
    }
    const clang::NamedDecl* param =
        generated_func.decls_by_io_channel.at(op.channel);
    XLS_CHECK(io_test_mode_ || external_channels_by_decl_.contains(param));
  }

  // Invoke loop over IOs
  PreparedBlock prepared;
  prepared.xls_func = &generated_func;
  prepared.args.push_back(selected_context);
  prepared.token = token;

  XLS_RETURN_IF_ERROR(
      GenerateIRBlockPrepare(prepared, pb,
                             /*next_return_index=*/extra_return_count,
                             /*next_state_index=*/extra_state_count,
                             /*this_type=*/nullptr,
                             /*this_decl=*/nullptr,
                             /*top_decls=*/{}, loc));

  XLS_ASSIGN_OR_RETURN(xls::BValue ret_tup,
                       GenerateIOInvokes(prepared, pb, loc));

  token = prepared.token;

  xls::BValue updated_context = pb.TupleIndex(ret_tup, 0, loc);
  xls::BValue do_break = pb.TupleIndex(ret_tup, 1, loc);

  // Send back context on break
  token = pb.SendIf(context_in_channel->generated, token, do_break,
                    updated_context, loc);

  // Construct next state
  std::vector<xls::BValue> next_state_values = {// First iteration next tick?
                                                do_break};
  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    const uint64_t field_idx = variable_field_indices.at(decl);
    xls::BValue val =
        GetStructFieldXLS(updated_context, field_idx, *context_ctype, loc);
    next_state_values.push_back(val);
  }
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    XLS_CHECK(context().fb == &pb);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "__xlscc_on_reset unsupported in pipelined loops"));
    }

    next_state_values.push_back(pb.TupleIndex(
        ret_tup, prepared.return_index_for_static.at(namedecl), loc));
  }

  //  xls::BValue next_state = pb.Tuple(next_state_values);
  XLS_RETURN_IF_ERROR(pb.Build(token, next_state_values).status());

  return absl::OkStatus();
}

}  // namespace xlscc
