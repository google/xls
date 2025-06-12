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

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xlscc {

namespace {

absl::StatusOr<std::optional<std::string>> FindContinuationNamesInThisContext(
    const TranslationContext& context, int64_t idx_from_top,
    const TrackedBValue* bval,
    absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>
        decl_by_bval,
    const xls::SourceInfo& loc) {
  CHECK_NE(bval, nullptr);

  // Look for names of special context values
  std::string ctx_found_name = "";

  // Look for name and decl in variables
  if (decl_by_bval.contains(bval)) {
    const clang::NamedDecl* decl = decl_by_bval.at(bval);
    ctx_found_name = decl->getNameAsString();
  }

  if (bval == &context.last_return_condition) {
    ctx_found_name = "last_return_condition";
  } else if (bval == &context.have_returned_condition) {
    ctx_found_name = "have_returned_condition";
  } else if (bval == &context.full_condition) {
    ctx_found_name = "full_condition";
  } else if (bval == &context.full_condition_on_enter_block) {
    ctx_found_name = "full_condition_on_enter_block";
  } else if (bval == &context.relative_condition) {
    ctx_found_name = "relative_condition";
  } else if (bval == &context.relative_break_condition) {
    ctx_found_name = "relative_break_condition";
  } else if (bval == &context.relative_continue_condition) {
    ctx_found_name = "relative_continue_condition";
  } else if (bval == &context.full_switch_cond) {
    ctx_found_name = "full_switch_cond";
  }

  if (!ctx_found_name.empty()) {
    return absl::StrFormat("ctx[%li].%s", idx_from_top, ctx_found_name);
  }

  return std::nullopt;
}

std::string GraphvizEscape(std::string_view s) {
  const int64_t max_label_length = 64;
  std::string label(s);
  if (label.size() > max_label_length) {
    label = label.substr(0, max_label_length);
  }
  return absl::StrFormat("\"%s\"",
                         absl::StrReplaceAll(label, {{"\"", "\\\""}}));
};

}  // namespace

absl::StatusOr<std::vector<NATIVE_BVAL>>
Translator::ConvertBValuesToContinuationOutputsForCurrentSlice(
    absl::flat_hash_map<const ContinuationValue*, std::vector<TrackedBValue*>>&
        bvalues_by_continuation_output,
    absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>&
        continuation_outputs_by_bval,
    absl::flat_hash_map<const TrackedBValue*, std::string>& name_found_for_bval,
    absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>&
        decls_by_bval_top_context,
    const xls::SourceInfo& loc) {
  XLSCC_CHECK(!context().sf->slices.empty(), loc);
  GeneratedFunctionSlice& current_slice = context().sf->slices.back();
  std::vector<NATIVE_BVAL> ret_vals;

  // Locked TrackedBValues scope
  {
    std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>>
        locked_bvalues = TrackedBValue::OrderedBValuesForBuilder(context().fb);

    TrackedBValue::Lock lock = std::move(std::get<0>(locked_bvalues));
    std::vector<TrackedBValue*> bvalues = std::get<1>(locked_bvalues);

    std::vector<xls::Node*> tracked_nodes_in_order;
    absl::flat_hash_map<xls::Node*, std::vector<TrackedBValue*>>
        tracked_bvalues_by_node;

    for (TrackedBValue* bval : bvalues) {
      // Invalid BValues are not recorded
      XLSCC_CHECK(bval->valid(), loc);
      XLSCC_CHECK_EQ(bval->builder(), context().fb, loc);
      if (!tracked_bvalues_by_node.contains(bval->node())) {
        tracked_nodes_in_order.push_back(bval->node());
      }
      tracked_bvalues_by_node[bval->node()].push_back(bval);
    }

    for (const auto& [decl, cval] : context().variables) {
      // TODO(seanhaskell): RValues in LValues in feedbacks
      if (!cval.rvalue().valid()) {
        continue;
      }
      decls_by_bval_top_context[&cval.rvalue()] = decl;
    }

    for (xls::Node* node : tracked_nodes_in_order) {
      std::vector<TrackedBValue*>& bvals = tracked_bvalues_by_node.at(node);
      ContinuationValue continuation_out;

      // Filled in for name search, identity is inserted later
      continuation_out.output_node = node;

      absl::StatusOr<xls::Value> result =
          EvaluateNode(node, loc, /*do_check=*/false);
      if (result.ok()) {
        continuation_out.literal = result.value();
      }

      current_slice.continuations_out.push_back(continuation_out);

      ContinuationValue& new_continuation =
          current_slice.continuations_out.back();

      CHECK(!bvalues_by_continuation_output.contains(&continuation_out));
      bvalues_by_continuation_output[&new_continuation] = bvals;

      for (TrackedBValue* bval : bvals) {
        CHECK(!continuation_outputs_by_bval.contains(bval));
        continuation_outputs_by_bval[bval] = &new_continuation;
        if (decls_by_bval_top_context.contains(bval)) {
          new_continuation.decls.insert(decls_by_bval_top_context.at(bval));
        }
      }
    }

    // Prefer names from the top of the stack first
    {
      int64_t idx_from_top = 0;
      for (auto rev_it = context_stack_.rbegin();
           rev_it != context_stack_.rend(); ++rev_it, ++idx_from_top) {
        const TranslationContext& context = *rev_it;

        absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>
            decl_by_bval;

        for (const auto& [decl, cval] : context.variables) {
          // TODO(seanhaskell): Find LValue names
          if (!cval.rvalue().valid()) {
            continue;
          }
          decl_by_bval[&cval.rvalue()] = decl;
        }

        for (const TrackedBValue* bval : bvalues) {
          std::string& name_found = name_found_for_bval[bval];
          // Don't overwrite a name already found (eg in another context)
          if (!name_found.empty()) {
            continue;
          }
          XLS_ASSIGN_OR_RETURN(
              std::optional<std::string> name_found_opt,
              FindContinuationNamesInThisContext(context, idx_from_top, bval,
                                                 decl_by_bval, loc));
          if (name_found_opt.has_value()) {
            name_found = name_found_opt.value();
          }
        }
      }

      // Fill in default unique names for those not found
      int64_t continuation_idx = 0;
      absl::flat_hash_set<std::string> names_inserted;
      for (const TrackedBValue* bval : bvalues) {
        std::string& name_found = name_found_for_bval[bval];
        if (name_found.empty()) {
          name_found = absl::StrFormat("continuation_%li", continuation_idx);
          continuation_idx++;
        }
        // Ensure the name is unique, even if it's a decl name
        std::string name_found_base = name_found;
        for (int64_t id = 1; names_inserted.contains(name_found); ++id) {
          name_found = absl::StrFormat("%s_%li", name_found_base, id);
        }
        names_inserted.insert(name_found);
      }

      // Check uniqueness of names
      absl::flat_hash_set<std::string> names_found;
      for (const auto& [bval, name] : name_found_for_bval) {
        XLSCC_CHECK(!names_found.contains(name), loc);
        names_found.insert(name);
      }
    }

    // Create continuation outputs
    ret_vals.reserve(current_slice.continuations_out.size());

    for (ContinuationValue& continuation_out :
         current_slice.continuations_out) {
      const std::vector<TrackedBValue*>& bvals =
          bvalues_by_continuation_output.at(&continuation_out);

      std::vector<std::string> names_found;
      names_found.reserve(bvals.size());
      for (const TrackedBValue* bval : bvals) {
        names_found.push_back(name_found_for_bval.at(bval));
      }
      continuation_out.name = absl::StrJoin(names_found, " ");

      static constexpr int64_t max_continuation_name_len = 32;
      if (continuation_out.name.size() > max_continuation_name_len) {
        continuation_out.name =
            continuation_out.name.substr(0, max_continuation_name_len);
      }

      NATIVE_BVAL identity_bval = context().fb->Identity(
          NATIVE_BVAL(continuation_out.output_node, context().fb), loc,
          /*name*/ absl::StrFormat("%s_output", continuation_out.name));

      continuation_out.output_node = identity_bval.node();
      ret_vals.push_back(identity_bval);
    }

    // Unregister all the TrackedBValues that are being continued
    XLSCC_CHECK_EQ(current_slice.continuations_out.size(),
                   bvalues_by_continuation_output.size(), loc);

    // Record top context outputs for feedbacks
    for (const auto& [decl, cval] : context().variables) {
      // TODO(seanhaskell): RValues in LValues in feedbacks
      if (!cval.rvalue().valid()) {
        continue;
      }
      current_slice.continuation_outputs_by_decl_top_context[decl] =
          continuation_outputs_by_bval.at(&cval.rvalue());
    }
  }

  // Reset tracked BValues to avoid registration error
  for (auto& [_, bvals] : bvalues_by_continuation_output) {
    for (TrackedBValue* bval : bvals) {
      bval->destroy();
    }
  }

  return ret_vals;
}

absl::Status Translator::AddContinuationsToNewSlice(
    const IOOp& after_op, GeneratedFunctionSlice& last_slice,
    GeneratedFunctionSlice& new_slice,
    const absl::flat_hash_map<const ContinuationValue*,
                              std::vector<TrackedBValue*>>&
        bvalues_by_continuation_output,
    const absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>&
        continuation_outputs_by_bval,
    const absl::flat_hash_map<const TrackedBValue*, std::string>&
        name_found_for_bval,
    const absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>&
        decls_by_bval_top_context,
    const xls::SourceInfo& loc) {
  // Create continuation inputs
  for (ContinuationValue& continuation_out : last_slice.continuations_out) {
    const std::vector<TrackedBValue*>& bvals =
        bvalues_by_continuation_output.at(&continuation_out);

    for (TrackedBValue* bval : bvals) {
      const std::string& name_found = name_found_for_bval.at(bval);
      NATIVE_BVAL input_bval = context().fb->Param(
          /*name*/ name_found, continuation_out.output_node->GetType(), loc);

      new_slice.continuations_in.push_back(
          ContinuationInput{.continuation_out = &continuation_out,
                            .input_node = input_bval.node()->As<xls::Param>(),
                            .name = name_found,
                            .decls = continuation_out.decls});

      if (decls_by_bval_top_context.contains(bval)) {
        const clang::NamedDecl* top_context_decl =
            decls_by_bval_top_context.at(bval);
        CHECK(!new_slice.continuation_inputs_by_decl_top_context.contains(
            top_context_decl));
        new_slice.continuation_inputs_by_decl_top_context[top_context_decl] =
            &new_slice.continuations_in.back();
      }
    }
  }

  // Each TrackedBValue gets its own input
  XLSCC_CHECK_GE(new_slice.continuations_in.size(),
                 last_slice.continuations_out.size(), loc);
  XLSCC_CHECK_EQ(last_slice.continuations_out.size(),
                 bvalues_by_continuation_output.size(), loc);

  // Update TrackedBValues
  for (const ContinuationInput& continuation_in : new_slice.continuations_in) {
    XLSCC_CHECK_NE(continuation_in.continuation_out, nullptr, loc);

    TrackedBValue in_bval;

    // Substitute literals to enable unrolling, IO short circuiting, etc.
    // Do not do this for the inputs of pipelined loops, as feedbacks will be
    // added later. Literals can still be substituted for them during
    // optimization.
    if (continuation_in.continuation_out->literal.has_value() &&
        after_op.op != OpType::kLoopBegin) {
      // Literals should only be propagated downstream, as upstream feedbacks
      // imply statefulness and a need for inductive reasoning about values.
      //
      // In this method, literals will naturally only come from upstream,
      // as the downstream slices have not been created yet.
      //
      // The unused continuation input will get optimized away later.
      in_bval = context().fb->Literal(
          continuation_in.continuation_out->literal.value(), loc,
          /*name=*/absl::StrFormat("%s_literal", continuation_in.name));
    } else {
      XLSCC_CHECK_EQ(continuation_in.input_node->function_base(),
                     context().fb->function(), loc);
      in_bval = TrackedBValue(continuation_in.input_node, context().fb);
    }
    XLSCC_CHECK(in_bval.valid(), loc);

    const std::vector<TrackedBValue*>& bvals =
        bvalues_by_continuation_output.at(continuation_in.continuation_out);

    for (TrackedBValue* bval : bvals) {
      *bval = in_bval;
    }
  }

  return absl::OkStatus();
}

// The continuation comes before the IO op, and so does not include its input
// parameter
absl::Status Translator::NewContinuation(IOOp& op) {
  // If there is no first slice, then don't generate any
  if (context().sf->slices.empty()) {
    return absl::OkStatus();
  }

  const xls::SourceInfo& loc = op.op_location;

  // ConvertBValuesToContinuationOutputsForCurrentSlice() will invalidate
  // BValues
  const NATIVE_BVAL ret_value_saved = op.ret_value;

  // Create only one ContinuationValue per xls::Node
  //
  // This prevents unnecessary complexity in the generated IR, such as selects
  // when propagating variables.
  //
  // It is safe because state element allocation considers the lifetimes of the
  // continuation values.
  absl::flat_hash_map<const ContinuationValue*, std::vector<TrackedBValue*>>
      bvalues_by_continuation_output;
  absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>
      continuation_outputs_by_bval;
  absl::flat_hash_map<const TrackedBValue*, std::string> name_found_for_bval;
  absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>
      decls_by_bval_top_context;

  XLS_ASSIGN_OR_RETURN(
      std::vector<NATIVE_BVAL> ret_vals,
      ConvertBValuesToContinuationOutputsForCurrentSlice(
          bvalues_by_continuation_output, continuation_outputs_by_bval,
          name_found_for_bval, decls_by_bval_top_context, loc));

  // TODO(seanhaskell): Turn into a check when subroutine calls work with new
  // FSM
  if (ret_value_saved.valid()) {
    ret_vals.push_back(ret_value_saved);
  }

  NATIVE_BVAL ret_bval =
      context().fb->Tuple(ret_vals, loc, /*name=*/"continuation_out");

  // Finish building the current slice
  XLS_RETURN_IF_ERROR(FinishSlice(ret_bval, loc));

  GeneratedFunctionSlice& last_slice = context().sf->slices.back();

  // Start building the next slice
  context().sf->slices.push_back(GeneratedFunctionSlice{});

  GeneratedFunctionSlice& new_slice = context().sf->slices.back();

  xls::BuilderBase* last_builder = context().fb;

  XLSCC_CHECK(functions_in_progress_.contains(context().sf->clang_decl), loc);
  FunctionInProgress& function_in_progress =
      *functions_in_progress_.at(context().sf->clang_decl);

  std::string_view xls_name =
      xls_names_for_functions_generated_.at(context().sf->clang_decl);

  function_in_progress.builder = std::make_unique<TrackedFunctionBuilder>(
      absl::StrFormat("%s_slice_after_%s_%i", xls_name, Debug_OpName(op),
                      op.channel_op_index),
      package_);

  // Update xls::FunctionBuilder pointers in TranslationContexts
  for (TranslationContext& context : context_stack_) {
    if (context.fb != last_builder) {
      continue;
    }
    context.fb = function_in_progress.builder->builder();
  }

  XLS_RETURN_IF_ERROR(AddContinuationsToNewSlice(
      op, last_slice, new_slice, bvalues_by_continuation_output,
      continuation_outputs_by_bval, name_found_for_bval,
      decls_by_bval_top_context, loc));

  return absl::OkStatus();
}

absl::Status Translator::AddFeedbacksForSlice(GeneratedFunctionSlice& slice,
                                              const xls::SourceInfo& loc) {
  if (slice.after_op == nullptr) {
    return absl::OkStatus();
  }
  if (slice.after_op->op != OpType::kLoopEndJump) {
    return absl::OkStatus();
  }

  absl::flat_hash_map<const IOOp*, std::list<GeneratedFunctionSlice>::iterator>
      slice_iters_by_after_op;

  for (std::list<GeneratedFunctionSlice>::iterator slice_it =
           context().sf->slices.begin();
       slice_it != context().sf->slices.end(); ++slice_it) {
    const GeneratedFunctionSlice& slice = *slice_it;
    if (slice.after_op == nullptr) {
      continue;
    }
    XLSCC_CHECK(!slice_iters_by_after_op.contains(slice.after_op), loc);
    slice_iters_by_after_op[slice.after_op] = slice_it;
  }

  // Add feedback inputs
  // This is done before optimization
  // These go from outputs of slice before jump to inputs of slice after begin

  const IOOp* jump_op = slice.after_op;
  std::list<GeneratedFunctionSlice>::iterator slice_after_jump_it =
      slice_iters_by_after_op.at(jump_op);
  XLSCC_CHECK(slice_after_jump_it != context().sf->slices.begin(), loc);
  --slice_after_jump_it;
  GeneratedFunctionSlice& slice_before_jump = *slice_after_jump_it;
  const IOOp* begin_op = jump_op->loop_op_paired;
  XLSCC_CHECK_NE(begin_op, nullptr, loc);
  GeneratedFunctionSlice& slice_after_begin =
      *slice_iters_by_after_op.at(begin_op);

  std::vector<const clang::NamedDecl*>
      slice_after_begin_decls_top_context_ordered;
  for (const auto& [decl, input] :
       slice_after_begin.continuation_inputs_by_decl_top_context) {
    slice_after_begin_decls_top_context_ordered.push_back(decl);
  }
  context().sf->SortNamesDeterministically(
      slice_after_begin_decls_top_context_ordered);

  std::vector<const clang::NamedDecl*>
      slice_before_jump_decls_top_context_ordered;
  for (const auto& [decl, input] :
       slice_before_jump.continuation_outputs_by_decl_top_context) {
    slice_before_jump_decls_top_context_ordered.push_back(decl);
  }
  context().sf->SortNamesDeterministically(
      slice_before_jump_decls_top_context_ordered);

  CHECK(slice_after_begin_decls_top_context_ordered ==
        slice_before_jump_decls_top_context_ordered);

  const std::vector<const clang::NamedDecl*>& decls_top_context =
      slice_after_begin_decls_top_context_ordered;

  for (const clang::NamedDecl* decl : decls_top_context) {
    ContinuationValue* feedback_out =
        slice_before_jump.continuation_outputs_by_decl_top_context.at(decl);
    ContinuationInput* feedback_in =
        slice_after_begin.continuation_inputs_by_decl_top_context.at(decl);
    ContinuationInput new_input = *feedback_in;
    new_input.continuation_out = feedback_out;
    slice_after_begin.continuations_in.push_back(new_input);
  }
  return absl::OkStatus();
}

absl::Status Translator::FinishSlice(NATIVE_BVAL return_bval,
                                     const xls::SourceInfo& loc) {
  XLSCC_CHECK(return_bval.valid(), loc);

  xls::FunctionBuilder* function_builder =
      dynamic_cast<xls::FunctionBuilder*>(context().fb);

  XLS_ASSIGN_OR_RETURN(xls::Function * last_slice_function,
                       function_builder->BuildWithReturnValue(return_bval));

  XLSCC_CHECK(!context().sf->slices.empty(), loc);
  context().sf->slices.back().function = last_slice_function;

  XLS_RETURN_IF_ERROR(AddFeedbacksForSlice(context().sf->slices.back(), loc));
  return absl::OkStatus();
}

absl::Status Translator::FinishLastSlice(TrackedBValue return_bval,
                                         const xls::SourceInfo& loc) {
  XLS_RETURN_IF_ERROR(FinishSlice(return_bval, loc));

  XLS_RETURN_IF_ERROR(OptimizeContinuations(*context().sf, loc));

  return absl::OkStatus();
}

namespace {

absl::Status RemoveUnusedContinuationOutputs(GeneratedFunction& func,
                                             bool& changed,
                                             const xls::SourceInfo& loc) {
  absl::flat_hash_set<const ContinuationValue*> outputs_used_by_inputs;

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      CHECK_NE(continuation_in.continuation_out, nullptr);
      outputs_used_by_inputs.insert(continuation_in.continuation_out);
    }
  }

  for (GeneratedFunctionSlice& slice : func.slices) {
    // Last slice has non-continuation output
    if (&slice == &func.slices.back()) {
      CHECK_EQ(slice.continuations_out.size(), 0);
      continue;
    }
    xls::Node* prev_return = slice.function->return_value();
    CHECK(prev_return->GetType()->IsTuple());

    const int64_t extra_returns =
        prev_return->operand_count() - slice.continuations_out.size();

    std::vector<xls::Node*> new_output_elems;
    std::vector<xls::Node*> removed_outputs;
    for (auto cont_out_it = slice.continuations_out.begin();
         cont_out_it != slice.continuations_out.end();) {
      ContinuationValue& continuation_out = *cont_out_it;
      if (outputs_used_by_inputs.contains(&continuation_out)) {
        ++cont_out_it;
        new_output_elems.push_back(continuation_out.output_node);
        continue;
      }

      removed_outputs.push_back(continuation_out.output_node);

      cont_out_it = slice.continuations_out.erase(cont_out_it);

      changed = true;
    }

    // If any outputs were removed, create one new output tuple for the slice
    if (!removed_outputs.empty()) {
      CHECK_EQ(new_output_elems.size(), slice.continuations_out.size());

      for (int64_t i = prev_return->operand_count() - extra_returns;
           i < prev_return->operand_count(); ++i) {
        new_output_elems.push_back(prev_return->operand(i));
      }

      XLS_ASSIGN_OR_RETURN(
          xls::Node * new_return,
          slice.function->MakeNode<xls::Tuple>(loc, new_output_elems));
      CHECK_EQ(new_return->operand_count(),
               extra_returns + slice.continuations_out.size());
      XLS_RETURN_IF_ERROR(slice.function->set_return_value(new_return));
      XLS_RETURN_IF_ERROR(slice.function->RemoveNode(prev_return));

      for (xls::Node* node : removed_outputs) {
        CHECK_EQ(node->function_base(), slice.function);
        XLS_RETURN_IF_ERROR(slice.function->RemoveNode(node));
      }

      changed = true;
    }
  }
  return absl::OkStatus();
}

absl::Status RemoveUnusedContinuationInputs(GeneratedFunction& func,
                                            bool& changed,
                                            const xls::SourceInfo& loc) {
  // Multiple inputs can share a parameter in the case of a phi /
  // feedback, so the already deleted parameters are tracked to remove
  // all of the inputs for an unused parameter.
  absl::flat_hash_set<const xls::Param*> deleted_params;

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (auto cont_in_it = slice.continuations_in.begin();
         cont_in_it != slice.continuations_in.end();) {
      ContinuationInput& continuation_in = *cont_in_it;

      if (deleted_params.contains(continuation_in.input_node)) {
        cont_in_it = slice.continuations_in.erase(cont_in_it);
        changed = true;
        continue;
      }

      CHECK_EQ(continuation_in.input_node->function_base(), slice.function);

      if (!continuation_in.input_node->users().empty() ||
          slice.function->HasImplicitUse(continuation_in.input_node)) {
        ++cont_in_it;
        continue;
      }

      XLS_RETURN_IF_ERROR(
          slice.function->RemoveNode(continuation_in.input_node));

      deleted_params.insert(continuation_in.input_node);

      cont_in_it = slice.continuations_in.erase(cont_in_it);
      changed = true;
    }
  }
  return absl::OkStatus();
}

absl::Status RemovePassThroughs(GeneratedFunction& func, bool& changed,
                                const xls::SourceInfo& loc) {
  absl::flat_hash_map<const xls::Param*, std::vector<ContinuationInput*>>
      continuation_inputs_by_input_node;
  absl::flat_hash_map<const xls::Node*, std::vector<ContinuationInput*>>
      continuation_inputs_by_output_node;
  absl::flat_hash_map<ContinuationInput*, GeneratedFunctionSlice*>
      slice_by_continuation_input;

  auto update_maps = [&]() {
    continuation_inputs_by_input_node.clear();
    continuation_inputs_by_output_node.clear();
    slice_by_continuation_input.clear();

    for (GeneratedFunctionSlice& slice : func.slices) {
      for (ContinuationInput& continuation_in : slice.continuations_in) {
        continuation_inputs_by_input_node[continuation_in.input_node].push_back(
            &continuation_in);

        continuation_inputs_by_output_node[continuation_in.continuation_out
                                               ->output_node]
            .push_back(&continuation_in);

        slice_by_continuation_input[&continuation_in] = &slice;
      }
    }
  };

  update_maps();

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (ContinuationValue& continuation_out : slice.continuations_out) {
      CHECK(continuation_out.output_node->op() == xls::Op::kIdentity);
      xls::Node* pass_in_node =
          continuation_out.output_node->operand(xls::UnOp::kArgOperand);
      if (!pass_in_node->Is<xls::Param>()) {
        continue;
      }
      xls::Param* pass_in_param = pass_in_node->As<xls::Param>();
      bool pass_through =
          continuation_inputs_by_input_node.contains(pass_in_param);
      if (!pass_through) {
        continue;
      }
      // If we reach here, then this output is fed directly from an input.
      // Therefore it is safe to redirect the inputs fed from this output
      // to the previous output.
      //
      // Output will get removed by other pass if it is now unused.
      // Input will get removed by other pass now that it is unused.

      CHECK(continuation_inputs_by_output_node.contains(
          continuation_out.output_node));
      std::vector<ContinuationInput*> pass_through_to_inputs =
          continuation_inputs_by_output_node.at(continuation_out.output_node);
      CHECK_GE(pass_through_to_inputs.size(), 1);

      // In the case of phis, optimization can end up with a slice passing
      // through to itself, need to break the cycle by deleting this
      for (ContinuationInput* pass_through_to_input : pass_through_to_inputs) {
        GeneratedFunctionSlice* downstream_slice =
            slice_by_continuation_input.at(pass_through_to_input);

        if (downstream_slice != &slice) {
          continue;
        }

        const int64_t prev_num = slice.continuations_in.size();
        slice.continuations_in.remove_if(
            [pass_through_to_input](const ContinuationInput& input) -> bool {
              return &input == pass_through_to_input;
            });
        CHECK_EQ(slice.continuations_in.size(), prev_num - 1);
        changed = true;
      }

      update_maps();

      if (!continuation_inputs_by_output_node.contains(
              continuation_out.output_node)) {
        CHECK(!continuation_inputs_by_output_node.contains(
            continuation_out.output_node));
        continue;
      }
      pass_through_to_inputs =
          continuation_inputs_by_output_node.at(continuation_out.output_node);

      // Each downstream input now needs to become N inputs, where N is the
      // number of upstream inputs for the pass-through output
      for (ContinuationInput* pass_through_to_input : pass_through_to_inputs) {
        GeneratedFunctionSlice* downstream_slice =
            slice_by_continuation_input.at(pass_through_to_input);

        const ContinuationInput pass_through_to_input_org =
            *pass_through_to_input;

        // Get all the inputs that use this parameter
        const std::vector<ContinuationInput*>& this_slice_inputs =
            continuation_inputs_by_input_node.at(pass_in_param);

        CHECK(!this_slice_inputs.empty());

        // The first input can simply be forwarded without creating new
        // downstream inputs
        auto this_slice_inputs_it = this_slice_inputs.begin();
        ContinuationInput* first_this_slice_input = *this_slice_inputs_it;

        CHECK_NE(pass_through_to_input->input_node, pass_in_param);
        CHECK_NE(first_this_slice_input->continuation_out, &continuation_out);

        pass_through_to_input->continuation_out =
            first_this_slice_input->continuation_out;
        changed = true;

        for (++this_slice_inputs_it;
             this_slice_inputs_it != this_slice_inputs.end();
             ++this_slice_inputs_it) {
          ContinuationInput* this_slice_input = *this_slice_inputs_it;
          CHECK(slice_by_continuation_input.contains(this_slice_input));

          CHECK_NE(pass_through_to_input->input_node, pass_in_param);
          CHECK_NE(this_slice_input->continuation_out, &continuation_out);

          ContinuationInput new_input = pass_through_to_input_org;
          new_input.continuation_out = this_slice_input->continuation_out;
          downstream_slice->continuations_in.push_back(new_input);
          changed = true;
        }
      }

      update_maps();

      CHECK(!continuation_inputs_by_output_node.contains(
          continuation_out.output_node));
    }
  }
  return absl::OkStatus();
}

// Uses XLS' Dead Code Removal pass to remove the nodes between unused outputs,
// which have been removed, and inputs feeding only unused outputs.
absl::Status RemoveDeadCode(GeneratedFunction& func, bool& changed,
                            xls::Package* package,
                            xls::OptimizationContext& context,
                            const xls::SourceInfo& loc) {
  xls::DeadCodeEliminationPass dce_pass;
  xls::PassResults results;
  xls::OptimizationPassOptions options;

  XLS_ASSIGN_OR_RETURN(bool dce_changed,
                       dce_pass.Run(package, options, &results, context));

  changed = changed || dce_changed;
  return absl::OkStatus();
}

absl::Status RemoveDuplicateInputs(GeneratedFunction& func, bool& changed,
                                   const xls::SourceInfo& loc) {
  struct InputKey {
    const xls::Param* input_node = nullptr;
    const ContinuationValue* continuation_out = nullptr;

    bool operator<(const InputKey& other) const {
      if (input_node->id() != other.input_node->id()) {
        return input_node->id() < other.input_node->id();
      }
      return continuation_out->output_node->id() <
             other.continuation_out->output_node->id();
    }
  };

  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::btree_map<InputKey,
                    std::vector<std::list<ContinuationInput>::iterator>>
        inputs_by_key;

    for (auto cont_in_it = slice.continuations_in.begin();
         cont_in_it != slice.continuations_in.end(); ++cont_in_it) {
      const ContinuationInput& continuation_in = *cont_in_it;
      InputKey key = {continuation_in.input_node,
                      continuation_in.continuation_out};
      inputs_by_key[key].push_back(cont_in_it);
    }

    // Delete all but the first
    for (auto& [_, inputs] : inputs_by_key) {
      bool first = true;
      for (auto input_it : inputs) {
        if (first) {
          first = false;
          continue;
        }
        slice.continuations_in.erase(input_it);
      }
    }
  }

  return absl::OkStatus();
}

// Note that literals are also propagated as continuations are created
// but none are propagated into pipelined loops, as it isn't known until
// all slices are generated whether or not a phi will be at a given input.
absl::Status SubstituteLiterals(GeneratedFunction& func, bool& changed,
                                const xls::SourceInfo& loc) {
  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::flat_hash_map<const xls::Param*, int64_t> input_counts_for_param;
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      ++input_counts_for_param[continuation_in.input_node];
    }
    for (ContinuationInput& continuation_in : slice.continuations_in) {
      if (!continuation_in.continuation_out->literal.has_value()) {
        continue;
      }
      // Can't propagate literals across phis
      if (input_counts_for_param.at(continuation_in.input_node) > 1) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          xls::Node * new_literal,
          slice.function->MakeNode<xls::Literal>(
              loc, continuation_in.continuation_out->literal.value()));
      XLS_RETURN_IF_ERROR(
          continuation_in.input_node->ReplaceUsesWith(new_literal));
      changed = true;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Translator::OptimizeContinuations(GeneratedFunction& func,
                                               const xls::SourceInfo& loc) {
  bool changed = true;
  xls::OptimizationContext context;

  do {
    changed = false;
    XLS_RETURN_IF_ERROR(RemoveUnusedContinuationInputs(func, changed, loc));
    XLS_RETURN_IF_ERROR(RemoveUnusedContinuationOutputs(func, changed, loc));
    XLS_RETURN_IF_ERROR(RemovePassThroughs(func, changed, loc));
    XLS_RETURN_IF_ERROR(RemoveDeadCode(func, changed, package_, context, loc));
    XLS_RETURN_IF_ERROR(RemoveDuplicateInputs(func, changed, loc));
    XLS_RETURN_IF_ERROR(SubstituteLiterals(func, changed, loc));
  } while (changed);

  return absl::OkStatus();
}

std::string GenerateReadableTypeName(xls::Type* type) {
  constexpr int64_t max_type_len = 64;
  std::string type_str = type->ToString();
  if (type_str.size() > max_type_len) {
    type_str = type_str.substr(0, max_type_len);
  }
  return type_str;
}

std::string GenerateSliceGraph(const GeneratedFunction& func) {
  // Pointers are used as names, as labels are not always unique
  std::vector<std::string> node_names;
  std::vector<std::string> rank_orders;
  std::vector<std::string> nodes_in_ranks;
  std::vector<std::string> nodes_with_edges;

  std::string last_rank_name = "";

  int64_t slice_index = -1;
  for (const GeneratedFunctionSlice& slice : func.slices) {
    ++slice_index;

    std::string new_rank = "(first)";
    if (slice.after_op != nullptr) {
      new_rank = Debug_OpName(*slice.after_op);
    }

    const std::string rank_input_name =
        GraphvizEscape(absl::StrFormat("%p_inputs", &slice));
    const std::string rank_output_name =
        GraphvizEscape(absl::StrFormat("%p_outputs", &slice));

    rank_orders.push_back(
        absl::StrFormat("  %s -> %s", rank_input_name, rank_output_name));
    if (!last_rank_name.empty()) {
      rank_orders.push_back(
          absl::StrFormat("  %s -> %s", last_rank_name, rank_input_name));
    }

    node_names.push_back(
        absl::StrFormat("  %s [label=%s style=rounded];", rank_input_name,
                        GraphvizEscape(absl::StrFormat(
                            "after [%i] %s inputs", slice_index, new_rank))));
    node_names.push_back(
        absl::StrFormat("  %s [label=%s];", rank_output_name,
                        GraphvizEscape(absl::StrFormat(
                            "after [%i] %s outputs", slice_index, new_rank))));

    last_rank_name = rank_output_name;

    std::vector<std::string> nodes_in_input_rank = {rank_input_name};
    std::vector<std::string> nodes_in_output_rank = {rank_output_name};

    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      const std::string output_name =
          GraphvizEscape(absl::StrFormat("%p", &continuation_out));
      const std::string type_str =
          GenerateReadableTypeName(continuation_out.output_node->GetType());

      node_names.push_back(
          absl::StrFormat("  %s [label=%s];", output_name,
                          GraphvizEscape(absl::StrFormat(
                              "%s : %s", continuation_out.name, type_str))));

      nodes_in_output_rank.push_back(output_name);
    }
    absl::flat_hash_map<xls::Param*, std::vector<const ContinuationInput*>>
        continuation_inputs_by_param;
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      continuation_inputs_by_param[continuation_in.input_node].push_back(
          &continuation_in);
    }
    for (auto& [param, continuation_inputs] : continuation_inputs_by_param) {
      const std::string input_name =
          GraphvizEscape(absl::StrFormat("%p", param));

      std::vector<std::string> label_parts;
      for (const ContinuationInput* continuation_in : continuation_inputs) {
        label_parts.push_back(continuation_in->name);
      }
      const std::string label = absl::StrJoin(label_parts, " ");

      node_names.push_back(absl::StrFormat("  %s [label=%s  style=rounded];",
                                           input_name, GraphvizEscape(label)));
      nodes_in_input_rank.push_back(input_name);
    }
    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      nodes_with_edges.push_back(absl::StrFormat(
          "  %s -> %s",
          GraphvizEscape(
              absl::StrFormat("%p", continuation_in.continuation_out)),
          GraphvizEscape(absl::StrFormat("%p", continuation_in.input_node))));
    }

    nodes_in_ranks.push_back(absl::StrFormat(
        "  { rank = same; %s; }", absl::StrJoin(nodes_in_input_rank, ";")));
    nodes_in_ranks.push_back(absl::StrFormat(
        "  { rank = same; %s; }", absl::StrJoin(nodes_in_output_rank, ";")));
  }

  return absl::StrFormat(
      R"(
digraph {
  nodesep = 0.4
  ranksep = 0.25

  node [shape=box];

  // node names
%s

  // rank_orders
%s

  // nodes_in_ranks
%s

  // nodes_with_edges
%s
}
    )",
      absl::StrJoin(node_names, "\n"), absl::StrJoin(rank_orders, "\n"),
      absl::StrJoin(nodes_in_ranks, "\n"),
      absl::StrJoin(nodes_with_edges, "\n"));
}

absl::Status Translator::GenerateFunctionSliceWrapper(
    GeneratedFunction& func, const xls::SourceInfo& loc) {
  XLSCC_CHECK(func.slices.size() == (func.io_ops.size() + 1), loc);

  // If there is only one slice, then no wrapper is needed
  if (func.slices.size() == 1) {
    context().sf->xls_func = func.slices.front().function;
    std::string_view xls_name =
        xls_names_for_functions_generated_.at(func.clang_decl);
    context().sf->xls_func->SetName(xls_name);
    return absl::OkStatus();
  }

  std::string_view xls_name =
      xls_names_for_functions_generated_.at(func.clang_decl);

  TrackedFunctionBuilder tracked_builder(xls_name, package_);
  xls::FunctionBuilder* builder = tracked_builder.builder();

  absl::flat_hash_map<const ContinuationValue*, TrackedBValue> prev_slice_ret;
  TrackedBValue last_slice_ret;

  for (GeneratedFunctionSlice& slice : func.slices) {
    std::vector<TrackedBValue> args;

    XLSCC_CHECK_GE(slice.function->params().size(),
                   slice.continuations_in.size(), loc);

    // Continuation params come first
    auto continuation_in_it = slice.continuations_in.begin();
    for (int64_t i = 0; continuation_in_it != slice.continuations_in.end();
         ++i, ++continuation_in_it) {
      XLSCC_CHECK_EQ(slice.function->params().at(i),
                     continuation_in_it->input_node, loc);

      TrackedBValue prev_slice_val =
          prev_slice_ret.at(continuation_in_it->continuation_out);
      XLSCC_CHECK(prev_slice_val.valid(), loc);

      args.push_back(prev_slice_val);
    }

    // Then parameters that should be forwarded from the top
    for (int64_t p = slice.continuations_in.size();
         p < slice.function->params().size(); ++p) {
      const xls::Param* slice_param = slice.function->params().at(p);
      TrackedBValue outer_param =
          builder->Param(slice_param->name(), slice_param->GetType(), loc);
      args.push_back(outer_param);
    }

    TrackedBValue slice_ret =
        builder->Invoke(ToNativeBValues(args), slice.function, loc);
    XLSCC_CHECK(slice_ret.valid(), loc);

    int64_t output_idx = 0;
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      XLSCC_CHECK(slice_ret.GetType()->IsTuple(), loc);
      XLSCC_CHECK_LT(output_idx, slice_ret.GetType()->AsTupleOrDie()->size(),
                     loc);

      prev_slice_ret[&continuation_out] =
          builder->TupleIndex(slice_ret, output_idx++, loc,
                              /*name=*/continuation_out.name);
    }

    last_slice_ret = slice_ret;
  }

  XLSCC_CHECK(last_slice_ret.valid(), loc);
  XLS_ASSIGN_OR_RETURN(func.xls_func,
                       builder->BuildWithReturnValue(last_slice_ret));
  XLSCC_CHECK_NE(func.xls_func, nullptr, loc);

  return absl::OkStatus();
}

}  // namespace xlscc
