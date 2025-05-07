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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
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

  // This may create multiple continuation values for a given xls::Node*
  absl::flat_hash_map<const ContinuationValue*, TrackedBValue*>
      bvalues_by_continuation_output;
  for (TrackedBValue* bval : bvalues) {
    // Invalid BValues are not recorded
    XLSCC_CHECK(bval->valid(), loc);
    XLSCC_CHECK_EQ(bval->builder(), context().fb, loc);

    ContinuationValue continuation_out;

    // Filled in for name search
    continuation_out.output_node = bval->node();

    current_slice.continuations_out.push_back(continuation_out);

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
    NATIVE_BVAL identity_bval = *bval;

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
    NATIVE_BVAL input_bval = output_bval;
    continuation_out.input_node = input_bval.node();

    new_slice.continuations_in.push_back(&continuation_out);
  }

  lock.UnlockEarly();

  // Update TrackedBValues
  for (ContinuationValue& continuation_out : current_slice.continuations_out) {
    TrackedBValue* bval = bvalues_by_continuation_output.at(&continuation_out);
    *bval = TrackedBValue(continuation_out.input_node, context().fb);
    XLSCC_CHECK(bval->valid(), loc);
  }

  return absl::OkStatus();
}

absl::Status Translator::OptimizeContinuations(GeneratedFunction& func) {
  return absl::OkStatus();
}

}  // namespace xlscc
