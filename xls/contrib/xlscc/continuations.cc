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
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"

namespace xlscc {

namespace {

absl::Status FindContinuationNamesInThisContext(
    const TranslationContext& context, int64_t idx_from_top,
    GeneratedFunctionSlice& current_slice,
    const absl::flat_hash_map<const ContinuationValue*, TrackedBValue*>&
        bvalues_by_continuation_output,
    const xls::SourceInfo& loc) {
  absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>
      decl_by_node;

  for (const auto& [decl, cval] : context.variables) {
    // TODO(seanhaskell): Find LValue names
    if (!cval.rvalue().valid()) {
      continue;
    }
    decl_by_node[&cval.rvalue()] = decl;
  }

  for (ContinuationValue& continuation_out : current_slice.continuations_out) {
    // Don't overwrite a name already found (eg in another context)
    if (!continuation_out.name.empty()) {
      continue;
    }

    CHECK_EQ(continuation_out.decl, nullptr);

    const TrackedBValue* bval =
        bvalues_by_continuation_output.at(&continuation_out);
    CHECK_NE(bval, nullptr);

    // Look for names of special context values
    std::string ctx_found_name = "";

    // Look for name and decl in variables
    if (decl_by_node.contains(bval)) {
      const clang::NamedDecl* decl = decl_by_node.at(bval);
      continuation_out.decl = decl;
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
      continuation_out.name =
          absl::StrFormat("ctx[%li].%s", idx_from_top, ctx_found_name);
    }
  }

  return absl::OkStatus();
}

}  // namespace

// The continuation comes before the IO op, and so does not include its input
// parameter
absl::Status Translator::NewContinuation(IOOp& op) {
  const xls::SourceInfo& loc = op.op_location;

  std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>> locked_bvalues =
      TrackedBValue::OrderedBValuesForBuilder(context().fb);

  TrackedBValue::Lock lock = std::move(std::get<0>(locked_bvalues));
  std::vector<TrackedBValue*> bvalues = std::get<1>(locked_bvalues);

  XLSCC_CHECK(!context().sf->slices.empty(), loc);
  GeneratedFunctionSlice& current_slice = context().sf->slices.back();

  absl::flat_hash_set<const TrackedBValue*> io_out_bvals;
  for (const IOOp& io_op : context().sf->io_ops) {
    io_out_bvals.insert(&io_op.ret_value);
  }
  io_out_bvals.insert(&op.ret_value);

  // This may create multiple continuation values for a given xls::Node*
  absl::flat_hash_map<const ContinuationValue*, TrackedBValue*>
      bvalues_by_continuation_output;
  for (TrackedBValue* bval : bvalues) {
    if (io_out_bvals.contains(bval)) {
      continue;
    }

    // Invalid BValues are not recorded
    XLSCC_CHECK(bval->valid(), loc);
    XLSCC_CHECK_EQ(bval->builder(), context().fb, loc);

    ContinuationValue continuation_out;

    // Filled in for name search
    continuation_out.output_node = bval->node();

    current_slice.continuations_out.push_back(continuation_out);

    CHECK(!bvalues_by_continuation_output.contains(&continuation_out));
    bvalues_by_continuation_output[&current_slice.continuations_out.back()] =
        bval;
  }

  // Prefer names from the top of the stack first
  {
    int64_t idx_from_top = 0;
    for (auto rev_it = context_stack_.rbegin(); rev_it != context_stack_.rend();
         ++rev_it, ++idx_from_top) {
      const TranslationContext& context = *rev_it;
      XLS_RETURN_IF_ERROR(FindContinuationNamesInThisContext(
          context, idx_from_top, current_slice, bvalues_by_continuation_output,
          loc));
    }
  }

  // Create continuation outputs
  int64_t continuation_idx = 0;
  for (ContinuationValue& continuation_out : current_slice.continuations_out) {
    if (continuation_out.name.empty()) {
      continuation_out.name =
          absl::StrFormat("continuation_%li", continuation_idx);
    }
    continuation_idx++;

    TrackedBValue* bval = bvalues_by_continuation_output.at(&continuation_out);

    // TODO(seanhaskell): Create output from function
    NATIVE_BVAL identity_bval = context().fb->Identity(
        *bval, loc,
        /*name*/ absl::StrFormat("%s_output", continuation_out.name));

    continuation_out.output_node = identity_bval.node();
  }

  // TODO(seanhaskell): Create a new function builder
  current_slice.function = nullptr;
  context().sf->slices.push_back(GeneratedFunctionSlice{});

  GeneratedFunctionSlice& new_slice = context().sf->slices.back();

  // Create continuation inputs
  for (ContinuationValue& continuation_out : current_slice.continuations_out) {
    NATIVE_BVAL output_bval(continuation_out.output_node, context().fb);
    // TODO(seanhaskell): Create parameter for input
    NATIVE_BVAL input_bval = context().fb->Identity(
        output_bval, loc,
        /*name*/ absl::StrFormat("%s_input", continuation_out.name));

    new_slice.continuations_in.push_back(
        ContinuationInput{.continuation_out = &continuation_out,
                          .input_node = input_bval.node()});
  }

  lock.UnlockEarly();

  // Update TrackedBValues
  XLSCC_CHECK_EQ(new_slice.continuations_in.size(),
                 current_slice.continuations_out.size(), loc);
  XLSCC_CHECK_EQ(current_slice.continuations_out.size(),
                 bvalues_by_continuation_output.size(), loc);

  for (const ContinuationInput& continuation_in : new_slice.continuations_in) {
    XLSCC_CHECK_NE(continuation_in.continuation_out, nullptr, loc);
    TrackedBValue* bval =
        bvalues_by_continuation_output.at(continuation_in.continuation_out);
    *bval = TrackedBValue(continuation_in.input_node, context().fb);
    XLSCC_CHECK(bval->valid(), loc);
  }

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
    for (auto cont_out_it = slice.continuations_out.begin();
         cont_out_it != slice.continuations_out.end();) {
      ContinuationValue& continuation_out = *cont_out_it;
      if (outputs_used_by_inputs.contains(&continuation_out)) {
        ++cont_out_it;
        continue;
      }

      XLS_RETURN_IF_ERROR(
          continuation_out.output_node->function_base()->RemoveNode(
              continuation_out.output_node));

      cont_out_it = slice.continuations_out.erase(cont_out_it);
      changed = true;
    }
  }
  return absl::OkStatus();
}

absl::Status RemoveUnusedContinuationInputs(GeneratedFunction& func,
                                            bool& changed,
                                            const xls::SourceInfo& loc) {
  for (GeneratedFunctionSlice& slice : func.slices) {
    for (auto cont_in_it = slice.continuations_in.begin();
         cont_in_it != slice.continuations_in.end();) {
      ContinuationInput& continuation_in = *cont_in_it;
      if (!continuation_in.input_node->users().empty()) {
        ++cont_in_it;
        continue;
      }

      XLS_RETURN_IF_ERROR(
          continuation_in.input_node->function_base()->RemoveNode(
              continuation_in.input_node));

      cont_in_it = slice.continuations_in.erase(cont_in_it);
      changed = true;
    }
  }
  return absl::OkStatus();
}

absl::Status RemovePassThroughs(GeneratedFunction& func, bool& changed,
                                const xls::SourceInfo& loc) {
  absl::flat_hash_map<const xls::Node*, ContinuationInput*>
      continuation_input_by_input_node;
  absl::flat_hash_map<const xls::Node*, std::vector<ContinuationInput*>>
      continuation_inputs_by_output_node;

  auto update_maps = [&]() {
    continuation_input_by_input_node.clear();
    continuation_inputs_by_output_node.clear();

    for (GeneratedFunctionSlice& slice : func.slices) {
      for (ContinuationInput& continuation_in : slice.continuations_in) {
        continuation_input_by_input_node[continuation_in.input_node] =
            &continuation_in;

        continuation_inputs_by_output_node[continuation_in.continuation_out
                                               ->output_node]
            .push_back(&continuation_in);
      }
    }
  };

  update_maps();

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (ContinuationValue& continuation_out : slice.continuations_out) {
      CHECK(continuation_out.output_node->op() == xls::Op::kIdentity);
      xls::Node* pass_in_node =
          continuation_out.output_node->operand(xls::UnOp::kArgOperand);
      bool pass_through =
          continuation_input_by_input_node.contains(pass_in_node);
      if (!pass_through) {
        continue;
      }
      // If we reach here, then this output is fed directly from an input.
      // Therefore it is safe to redirect the inputs fed from this output
      // to the previous output.

      // Output will get removed by other pass if it is now unused.
      // Input will get removed by other pass now that it is unused.
      ContinuationInput* this_slice_input =
          continuation_input_by_input_node.at(pass_in_node);

      ContinuationValue* pass_through_from_value =
          this_slice_input->continuation_out;

      CHECK(continuation_inputs_by_output_node.contains(
          continuation_out.output_node));
      const std::vector<ContinuationInput*>& pass_through_to_inputs =
          continuation_inputs_by_output_node.at(continuation_out.output_node);
      CHECK_GE(pass_through_to_inputs.size(), 1);

      // Make the inputs point to the output that feeds the pass-through
      for (ContinuationInput* pass_through_to_input : pass_through_to_inputs) {
        pass_through_to_input->continuation_out = pass_through_from_value;
        CHECK(pass_through_to_input->input_node->op() == xls::Op::kIdentity);
        XLS_RETURN_IF_ERROR(
            pass_through_to_input->input_node->ReplaceOperandNumber(
                xls::UnOp::kArgOperand, pass_through_from_value->output_node));
        changed = true;
      }

      update_maps();

      CHECK(!continuation_inputs_by_output_node.contains(
          continuation_out.output_node));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status Translator::OptimizeContinuations(GeneratedFunction& func,
                                               const xls::SourceInfo& loc) {
  bool changed = true;

  while (changed) {
    changed = false;
    XLS_RETURN_IF_ERROR(RemoveUnusedContinuationInputs(func, changed, loc));
    XLS_RETURN_IF_ERROR(RemoveUnusedContinuationOutputs(func, changed, loc));
    XLS_RETURN_IF_ERROR(RemovePassThroughs(func, changed, loc));

    // TODO(seanhaskell): Dead code removal
  }

  // TODO: Mark loop phis

  // TODO: Literals (consider phi)
  // TODO: Remove continuation inputs that feed through other nodes to only
  // unused outputs

  return absl::OkStatus();
}

}  // namespace xlscc
