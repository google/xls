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

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/Decl.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/generate_fsm.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/data_flow_node_info.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/node_source_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/partial_info_query_engine.h"
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

  auto compound_cond_contains = [](const TrackedBValue* bval,
                                   const CompoundPredicate& pred) -> bool {
    for (const TrackedBValue& term : pred.all_terms()) {
      if (&term == bval) {
        return true;
      }
    }
    return false;
  };

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
  } else if (compound_cond_contains(bval, context.full_condition)) {
    ctx_found_name = "full_condition";
  } else if (bval == &context.full_condition_on_enter_block) {
    ctx_found_name = "full_condition_on_enter_block";
  } else if (compound_cond_contains(bval, context.relative_condition)) {
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

absl::Status ValidateContinuations(GeneratedFunction& func,
                                   const xls::SourceInfo& loc) {
  absl::flat_hash_map<const ContinuationValue*, int64_t>
      slice_index_by_continuation_out;
  absl::flat_hash_map<const GeneratedFunctionSlice*, int64_t>
      slice_index_by_slice;

  for (GeneratedFunctionSlice& slice : func.slices) {
    const int64_t slice_index = slice_index_by_slice.size();
    slice_index_by_slice[&slice] = slice_index;
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      slice_index_by_continuation_out[&continuation_out] = slice_index;
    }
  }
  {
    absl::flat_hash_map<const xls::Param*, absl::btree_set<StateId>>
        state_ids_by_param;

    for (GeneratedFunctionSlice& slice : func.slices) {
      for (const ContinuationInput& continuation_in : slice.continuations_in) {
        for (const StateId& state_id : continuation_in.choose_in_states) {
          if (state_ids_by_param.contains(continuation_in.input_node) &&
              state_ids_by_param.at(continuation_in.input_node)
                  .contains(state_id)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Param %s to slice[%li] has multiple inputs for state %s\n",
                continuation_in.input_node->name(),
                slice_index_by_slice.at(&slice), state_id.ToString()));
          }
          state_ids_by_param[continuation_in.input_node].insert(state_id);
        }
      }
    }
  }
  for (GeneratedFunctionSlice& slice : func.slices) {
    const int64_t slice_index = slice_index_by_slice.at(&slice);

    absl::flat_hash_map<const xls::Param*, int64_t>
        num_upstream_inputs_by_param;

    absl::flat_hash_map<const xls::Param*,
                        absl::flat_hash_set<ContinuationValue*>>
        values_inputted_by_param;

    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      const int64_t upstream_slice_index =
          slice_index_by_continuation_out.at(continuation_in.continuation_out);

      const bool is_feedback = slice_index <= upstream_slice_index;
      int64_t& num_upstream_inputs_for_param =
          num_upstream_inputs_by_param[continuation_in.input_node];

      if (!is_feedback) {
        ++num_upstream_inputs_for_param;
      }

      if (values_inputted_by_param[continuation_in.input_node].contains(
              continuation_in.continuation_out)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Param %s to slice[%li] %s has multiple inputs for value %s\n",
            continuation_in.input_node->name(), slice_index,
            slice.function->name(), continuation_in.continuation_out->name));
      }

      values_inputted_by_param[continuation_in.input_node].insert(
          continuation_in.continuation_out);
    }

    for (const auto& [param, num_upstream_inputs] :
         num_upstream_inputs_by_param) {
      if (num_upstream_inputs != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Param %s to slice %s has %i upstream inputs, should have exactly "
            "1",
            param->name(), slice.function->name(), num_upstream_inputs));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status GenerateLayoutAndInsertChooseInStates(GeneratedFunction& func,
                                                   NewFSMGenerator& generator,
                                                   const xls::SourceInfo& loc) {
  NewFSMLayout layout_ref;

  XLS_RETURN_IF_ERROR(
      generator.LayoutNewFSMNoStateElements(layout_ref, func.slices, loc));

  absl::flat_hash_map<std::tuple<const xls::Param*, const ContinuationValue*>,
                      ContinuationInput*>
      continuation_in_by_param_and_continuation_out;

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (ContinuationInput& continuation_in : slice.continuations_in) {
      continuation_in_by_param_and_continuation_out[std::make_tuple(
          continuation_in.input_node, continuation_in.continuation_out)] =
          &continuation_in;
      continuation_in.choose_in_states.clear();
    }
  }
  for (const NewFSMState& state : layout_ref.states) {
    StateId state_id = {
        .slice_index = state.slice_index,
    };
    for (const JumpInfo& jump_info : state.jumped_from_slice_indices) {
      state_id.from_jump_slice_indices.insert(JumpId{
          .from_slice_index = jump_info.from_slice, .count = jump_info.count});
    }

    for (const auto& [param, continuation_out] :
         state.current_inputs_by_input_param) {
      ContinuationInput* continuation_in =
          continuation_in_by_param_and_continuation_out.at(
              std::make_tuple(param, continuation_out));

      continuation_in->choose_in_states.insert(state_id);
    }
  }

  return absl::OkStatus();
}

}  // namespace

SourcesSetNodeInfo::SourcesSetNodeInfo()
    : xls::DataFlowLazyNodeInfo<SourcesSetNodeInfo, ParamSet>(
          /*compute_tree_for_source=*/false, /*default_info_source=*/false,
          /*include_selectors=*/true) {}

ParamSet SourcesSetNodeInfo::ComputeInfoForBitsLiteral(
    const xls::Bits& literal) const {
  return ParamSet();
}

ParamSet SourcesSetNodeInfo::ComputeInfoForNode(xls::Node* node) const {
  if (node->Is<xls::Param>()) {
    return ParamSet{node->As<xls::Param>()};
  }
  return ParamSet();
}

xls::LeafTypeTree<ParamSet> SourcesSetNodeInfo::ComputeInfoTreeForNode(
    xls::Node* node) const {
  LOG(FATAL)
      << "ComputeInfoTreeForNode should be unused for SourcesInSetNodeInfo";
  return xls::LeafTypeTree<ParamSet>();
}

ParamSet SourcesSetNodeInfo::MergeInfos(
    absl::Span<const absl::Span<const ParamSet>> spans) const {
  ParamSet ret;
  for (const auto& span : spans) {
    for (const ParamSet& info : span) {
      ret.insert(info.begin(), info.end());
    }
  }
  return ret;
}

SourcesSetTreeNodeInfo::SourcesSetTreeNodeInfo()
    : xls::DataFlowLazyNodeInfo<SourcesSetTreeNodeInfo, NodeSourceSetPtr>(
          /*compute_tree_for_source=*/true, /*default_info_source=*/true,
          /*include_selectors=*/false) {}

NodeSourceSetPtr SourcesSetTreeNodeInfo::ComputeInfoForBitsLiteral(
    const xls::Bits& literal) const {
  LOG(FATAL) << "ComputeInfoForBitsLiteral should be unused for "
                "SourcesSetTreeNodeInfo";
  return NodeSourceSetPtr();
}

NodeSourceSetPtr SourcesSetTreeNodeInfo::ComputeInfoForNode(
    xls::Node* node) const {
  LOG(FATAL)
      << "ComputeInfoForNode should be unused for SourcesSetTreeNodeInfo";
  return NodeSourceSetPtr();
}

xls::LeafTypeTree<NodeSourceSetPtr>
SourcesSetTreeNodeInfo::ComputeInfoTreeForNode(xls::Node* node) const {
  auto result = xls::LeafTypeTree<NodeSourceSetPtr>::CreateFromFunction(
      node->GetType(),
      [&](xls::Type* element_type, absl::Span<const int64_t> index) {
        return std::make_shared<const absl::flat_hash_set<xls::NodeSource>>(
            std::initializer_list<xls::NodeSource>{
                {node, std::vector(index.begin(), index.end())}});
      });
  CHECK_OK(result.status());
  return *std::move(result);
}

NodeSourceSetPtr SourcesSetTreeNodeInfo::MergeTwo(NodeSourceSetPtr a,
                                                  NodeSourceSetPtr b) const {
  if (a == nullptr || a->empty()) {
    return b;
  }
  if (a == b || b == nullptr || b->empty()) {
    return a;
  }

  // Make sure the key doesn't depend on the input order.
  auto key = a > b ? std::make_pair(b, a) : std::make_pair(a, b);

  auto [it, inserted] = merge_cache_.try_emplace(key, nullptr);
  if (!inserted) {
    return it->second;
  }

  NodeSourceSetPtr large = a;
  NodeSourceSetPtr small = b;
  if (large->size() < small->size()) {
    std::swap(large, small);
  }

  auto merged_set =
      std::make_shared<absl::flat_hash_set<xls::NodeSource>>(*large);
  merged_set->insert(small->begin(), small->end());

  if (merged_set->size() == large->size()) {
    // The small set was a subset of the large set.
    it->second = large;
    return large;
  }

  it->second = merged_set;
  return merged_set;
}

NodeSourceSetPtr SourcesSetTreeNodeInfo::MergeInfos(
    absl::Span<const absl::Span<const NodeSourceSetPtr>> spans) const {
  NodeSourceSetPtr ret = nullptr;
  for (const auto& span : spans) {
    for (const NodeSourceSetPtr& info : span) {
      ret = MergeTwo(ret, info);
    }
  }
  if (ret == nullptr) {
    static const auto empty_set =
        std::make_shared<const absl::flat_hash_set<xls::NodeSource>>();
    return empty_set;
  }
  return ret;
}

absl::StatusOr<xls::PartialInfoQueryEngine*>
OptimizationContext::GetQueryEngineForFunction(xls::FunctionBase* in_function) {
  if (!query_engines_by_function_.contains(in_function)) {
    auto new_query_engine_ptr = std::make_unique<xls::PartialInfoQueryEngine>();
    XLS_RETURN_IF_ERROR(new_query_engine_ptr->Populate(in_function).status());
    query_engines_by_function_[in_function] = std::move(new_query_engine_ptr);
  }
  return query_engines_by_function_.at(in_function).get();
}

absl::StatusOr<SourcesSetNodeInfo*>
OptimizationContext::GetSourcesSetNodeInfoForFunction(
    xls::FunctionBase* in_function) {
  if (!sources_net_node_infos_by_function_.contains(in_function)) {
    CHECK(in_function->IsFunction());
    XLS_ASSIGN_OR_RETURN(xls::PartialInfoQueryEngine * query_engine,
                         GetQueryEngineForFunction(in_function));
    auto new_info_ptr = std::make_unique<SourcesSetNodeInfo>();
    new_info_ptr->set_query_engine(query_engine);
    XLS_RETURN_IF_ERROR(new_info_ptr->Attach(in_function).status());
    sources_net_node_infos_by_function_[in_function] = std::move(new_info_ptr);
  }
  return sources_net_node_infos_by_function_.at(in_function).get();
}

absl::StatusOr<SourcesSetTreeNodeInfo*>
OptimizationContext::GetSourcesSetTreeNodeInfoForFunction(
    xls::FunctionBase* in_function) {
  if (!sources_set_tree_node_infos_by_function_.contains(in_function)) {
    CHECK(in_function->IsFunction());
    XLS_ASSIGN_OR_RETURN(xls::PartialInfoQueryEngine * query_engine,
                         GetQueryEngineForFunction(in_function));
    auto new_info_ptr = std::make_unique<SourcesSetTreeNodeInfo>();
    new_info_ptr->set_query_engine(query_engine);
    XLS_RETURN_IF_ERROR(new_info_ptr->Attach(in_function).status());
    sources_set_tree_node_infos_by_function_[in_function] =
        std::move(new_info_ptr);
  }
  return sources_set_tree_node_infos_by_function_.at(in_function).get();
}

const xls::LeafTypeTree<std::monostate>&
OptimizationContext::GetBlankTypeTreeForType(xls::Type* type) {
  CHECK_NE(type, nullptr);
  auto [param_tree_it, _] = param_tree_cache_.try_emplace(type, type);
  xls::LeafTypeTree<std::monostate>& found = param_tree_it->second;
  CHECK(found.type()->IsEqualTo(type));
  return found;
}

absl::StatusOr<bool> OptimizationContext::CheckNodeSourcesInSet(
    xls::FunctionBase* in_function, xls::Node* node,
    absl::flat_hash_set<const xls::Param*> sources_set,
    bool allow_empty_sources_result) {
  // Save lazy node analysis for each function for efficiency
  XLS_ASSIGN_OR_RETURN(SourcesSetNodeInfo * info,
                       GetSourcesSetNodeInfoForFunction(in_function));

  ParamSet param_sources = info->GetSingleInfoForNode(node);

  if (param_sources.empty()) {
    return allow_empty_sources_result;
  }

  bool all_in_set = true;
  for (const xls::Param* param : param_sources) {
    CHECK_EQ(param->function_base(), in_function);
    if (!sources_set.contains(param)) {
      all_in_set = false;
      break;
    }
  }

  return all_in_set;
}

absl::StatusOr<std::vector<NATIVE_BVAL>>
Translator::ConvertBValuesToContinuationOutputsForCurrentSlice(
    absl::flat_hash_map<const ContinuationValue*, std::vector<TrackedBValue*>>&
        bvalues_by_continuation_output,
    absl::flat_hash_map<const TrackedBValue*, ContinuationValue*>&
        continuation_outputs_by_bval,
    absl::flat_hash_map<const TrackedBValue*, std::string>& name_found_for_bval,
    absl::flat_hash_map<const TrackedBValue*, const clang::NamedDecl*>&
        decls_by_bval_top_context,
    int64_t* total_bvals_out, const xls::SourceInfo& loc) {
  XLSCC_CHECK(!context().sf->slices.empty(), loc);
  GeneratedFunctionSlice& current_slice = context().sf->slices.back();
  std::vector<NATIVE_BVAL> ret_vals;

  // Locked TrackedBValues scope
  {
    std::tuple<TrackedBValue::Lock, std::vector<TrackedBValue*>>
        locked_bvalues = TrackedBValue::OrderedBValuesForBuilder(context().fb);

    TrackedBValue::Lock lock = std::move(std::get<0>(locked_bvalues));
    std::vector<TrackedBValue*> bvalues = std::get<1>(locked_bvalues);

    *total_bvals_out = bvalues.size();

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
          new_continuation.decls.insert(
              DeclLeaf{.decl = decls_by_bval_top_context.at(bval)});
        }
      }
    }

    // Prefer names from the top of the stack first
    absl::flat_hash_set<const TrackedBValue*> bvals_with_default_names;
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
          bvals_with_default_names.insert(bval);
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
      // Continuations added by subroutine calls don't need name assignment
      if (bvalues_by_continuation_output.contains(&continuation_out)) {
        const std::vector<TrackedBValue*>& bvals =
            bvalues_by_continuation_output.at(&continuation_out);

        std::optional<std::string> default_name = std::nullopt;

        std::vector<std::string> names_found;
        names_found.reserve(bvals.size());
        for (const TrackedBValue* bval : bvals) {
          if (bvals_with_default_names.contains(bval)) {
            default_name = name_found_for_bval.at(bval);
            continue;
          }
          names_found.push_back(name_found_for_bval.at(bval));
        }
        if (names_found.empty()) {
          continuation_out.name = default_name.value();
        } else {
          continuation_out.name = absl::StrJoin(names_found, " ");
        }

        static constexpr int64_t max_continuation_name_len = 32;
        if (continuation_out.name.size() > max_continuation_name_len) {
          continuation_out.name =
              continuation_out.name.substr(0, max_continuation_name_len);
        }
      }

      NATIVE_BVAL identity_bval = context().fb->Identity(
          NATIVE_BVAL(continuation_out.output_node, context().fb), loc,
          /*name*/ absl::StrFormat("%s_output", continuation_out.name));

      continuation_out.output_node = identity_bval.node();
      ret_vals.push_back(identity_bval);
    }

    // Unregister all the TrackedBValues that are being continued
    XLSCC_CHECK_GE(current_slice.continuations_out.size(),
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
    OpType after_op_type, GeneratedFunctionSlice& last_slice,
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
    int64_t total_bvals, const xls::SourceInfo& loc) {
  // Create continuation inputs

  absl::flat_hash_map<const ContinuationInput*, TrackedBValue*>
      bvals_by_continuation_input;

  for (ContinuationValue& continuation_out : last_slice.continuations_out) {
    if (!bvalues_by_continuation_output.contains(&continuation_out)) {
      continue;
    }

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

      bvals_by_continuation_input[&new_slice.continuations_in.back()] = bval;

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
  XLSCC_CHECK(bvals_by_continuation_input.size() == total_bvals, loc);
  XLSCC_CHECK_GE(new_slice.continuations_in.size(),
                 bvalues_by_continuation_output.size(), loc);
  XLSCC_CHECK_GE(last_slice.continuations_out.size(),
                 bvalues_by_continuation_output.size(), loc);

  absl::flat_hash_set<TrackedBValue*> bvals_set;

  // Update TrackedBValues
  for (const ContinuationInput& continuation_in : new_slice.continuations_in) {
    XLSCC_CHECK_NE(continuation_in.continuation_out, nullptr, loc);

    TrackedBValue in_bval;

    // Substitute literals to enable unrolling, IO short circuiting, etc.
    // Do not do this for the inputs of pipelined loops, as feedbacks will be
    // added later. Literals can still be substituted for them during
    // optimization.
    if (continuation_in.continuation_out->literal.has_value() &&
        after_op_type != OpType::kLoopBegin) {
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
    } else if (continuation_in.continuation_out->output_node->GetType()
                   ->GetFlatBitCount() == 0) {
      // Zero bit types shouldn't be passed through continuations
      in_bval = context().fb->Literal(
          xls::ZeroOfType(
              continuation_in.continuation_out->output_node->GetType()),
          loc,
          /*name=*/absl::StrFormat("%s_literal", continuation_in.name));
    } else {
      XLSCC_CHECK_EQ(continuation_in.input_node->function_base(),
                     context().fb->function(), loc);
      in_bval = TrackedBValue(continuation_in.input_node, context().fb);
    }
    XLSCC_CHECK(in_bval.valid(), loc);

    TrackedBValue* bval = bvals_by_continuation_input.at(&continuation_in);

    *bval = in_bval;

    XLSCC_CHECK(!bvals_set.contains(bval), loc);
    bvals_set.insert(bval);
  }

  XLSCC_CHECK(bvals_set.size() == total_bvals, loc);

  return absl::OkStatus();
}

std::string Translator::FormatSliceName(std::string_view op_name,
                                        const xls::SourceInfo& loc,
                                        int64_t channel_op_index,
                                        bool create_slice_before,
                                        bool temp_name) {
  std::string_view xls_name =
      xls_names_for_functions_generated_.at(context().sf->clang_decl);

  if (temp_name) {
    XLSCC_CHECK(create_slice_before, loc);
    return absl::StrFormat("%s_slice_before__temp", xls_name);
  }
  return absl::StrFormat("%s_slice_%s_%s_%i", xls_name,
                         create_slice_before ? "before" : "after", op_name,
                         channel_op_index);
}

// The continuation comes before the IO op, and so does not include its input
// parameter
absl::Status Translator::NewContinuation(
    OpType op_type, std::string_view op_name, TrackedBValue op_ret_value,
    const xls::SourceInfo& loc, int64_t channel_op_index,
    bool create_slice_before, bool temp_name) {
  // If there is no first slice, then don't generate any
  if (context().sf->slices.empty()) {
    return absl::OkStatus();
  }

  // ConvertBValuesToContinuationOutputsForCurrentSlice() will invalidate
  // BValues
  const NATIVE_BVAL ret_value_saved = op_ret_value;

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

  int64_t total_bvals = 0;

  XLS_ASSIGN_OR_RETURN(
      std::vector<NATIVE_BVAL> ret_vals,
      ConvertBValuesToContinuationOutputsForCurrentSlice(
          bvalues_by_continuation_output, continuation_outputs_by_bval,
          name_found_for_bval, decls_by_bval_top_context,
          /*total_bvals_out=*/&total_bvals, loc));

  GeneratedFunctionSlice& last_slice = context().sf->slices.back();

  // Only slices before IO operations return IO op conditions.
  //
  // If the next slice, the one this function is about to create,
  // will be an explicit "before IO" slice, then this one, which we are
  // finalizing with its output, doesn't need an IO return.
  //
  // TODO(seanhaskell): Turn into a check when subroutine calls work with new
  // FSM
  if (!create_slice_before) {
    XLSCC_CHECK(ret_value_saved.valid(), loc);
    ret_vals.push_back(ret_value_saved);
  }

  NATIVE_BVAL ret_bval =
      context().fb->Tuple(ret_vals, loc, /*name=*/"continuation_out");

  // Finish building the current slice
  XLS_RETURN_IF_ERROR(FinishSlice(ret_bval, loc));

  // Start building the next slice
  context().sf->slices.push_back(GeneratedFunctionSlice{
      .is_slice_before = create_slice_before,
  });

  xls::BuilderBase* last_builder = context().fb;

  XLSCC_CHECK(functions_in_progress_.contains(context().sf->clang_decl), loc);
  FunctionInProgress& function_in_progress =
      *functions_in_progress_.at(context().sf->clang_decl);

  std::string slice_name = FormatSliceName(op_name, loc, channel_op_index,
                                           create_slice_before, temp_name);

  function_in_progress.builder =
      std::make_unique<TrackedFunctionBuilder>(slice_name, package_);

  // Update xls::FunctionBuilder pointers in TranslationContexts
  for (TranslationContext& context : context_stack_) {
    if (context.fb != last_builder) {
      continue;
    }
    context.fb = function_in_progress.builder->builder();
  }

  GeneratedFunctionSlice& new_slice = context().sf->slices.back();

  XLS_RETURN_IF_ERROR(AddContinuationsToNewSlice(
      op_type, last_slice, new_slice, bvalues_by_continuation_output,
      continuation_outputs_by_bval, name_found_for_bval,
      decls_by_bval_top_context, total_bvals, loc));

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
  auto slice_before_jump_it = slice_after_jump_it;
  --slice_before_jump_it;
  GeneratedFunctionSlice& slice_before_jump = *slice_before_jump_it;

  const IOOp* begin_op = jump_op->loop_op_paired;
  XLSCC_CHECK_NE(begin_op, nullptr, loc);
  std::list<GeneratedFunctionSlice>::iterator slice_after_begin_it =
      slice_iters_by_after_op.at(begin_op);
  GeneratedFunctionSlice& slice_after_begin = *slice_after_begin_it;

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

  for (auto slice_it = slice_after_begin_it; slice_it != slice_after_jump_it;
       ++slice_it) {
    GeneratedFunctionSlice& slice = *slice_it;
    if (!slice.static_values.empty()) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*slice.static_values.front()),
                       "Static values in loop body with new FSM"));
    }
  }

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

  GeneratedFunctionSlice& last_slice = context().sf->slices.back();
  last_slice.function = last_slice_function;

  if (last_slice.do_add_feedbacks) {
    XLS_RETURN_IF_ERROR(AddFeedbacksForSlice(last_slice, loc));
  }

  return absl::OkStatus();
}

absl::Status Translator::RemoveMaskedOpParams(GeneratedFunction& func,
                                              const xls::SourceInfo& loc) {
  for (xls::Param* param : context().sf->masked_op_params_to_remove) {
    XLS_RETURN_IF_ERROR(param
                            ->ReplaceUsesWithNew<xls::Literal>(
                                xls::ZeroOfType(param->GetType()))
                            .status());
    XLS_RETURN_IF_ERROR(param->function_base()->RemoveNode(param));
  }
  context().sf->masked_op_params_to_remove.clear();
  return absl::OkStatus();
}

absl::Status Translator::DecomposeContinuationValues(
    GeneratedFunction& func, bool& changed, const xls::SourceInfo& loc) {
  absl::flat_hash_map<const ContinuationValue*, bool>
      original_output_decomposable;

  absl::flat_hash_map<const xls::Param*, bool> decomposable_params;

  absl::flat_hash_map<const ContinuationValue*, std::vector<const xls::Param*>>
      params_by_output;

  {
    for (GeneratedFunctionSlice& slice : func.slices) {
      // Ensure params_by_output is initialized for all outputs
      for (ContinuationValue& continuation_out : slice.continuations_out) {
        params_by_output[&continuation_out] = {};
      }
    }

    // Ensure decomposable_params is initialized
    for (GeneratedFunctionSlice& slice : func.slices) {
      for (ContinuationInput& continuation_in : slice.continuations_in) {
        params_by_output[continuation_in.continuation_out].push_back(
            continuation_in.input_node);
        decomposable_params[continuation_in.input_node] = true;
      }
    }

    for (GeneratedFunctionSlice& slice : func.slices) {
      for (ContinuationValue& continuation_out : slice.continuations_out) {
        original_output_decomposable[&continuation_out] =
            TypeIsDecomposable(continuation_out.output_node->GetType()) &&
            !continuation_out.direct_in;
      }
    }

    // Iteratively mark things not decomposable
    for (bool iter_changed = true; iter_changed;) {
      iter_changed = false;

      for (GeneratedFunctionSlice& slice : func.slices) {
        for (ContinuationInput& continuation_in : slice.continuations_in) {
          if (!decomposable_params.at(continuation_in.input_node)) {
            continue;
          }
          if (!original_output_decomposable.at(
                  continuation_in.continuation_out)) {
            decomposable_params[continuation_in.input_node] = false;
            iter_changed = true;
          }
        }
      }

      for (GeneratedFunctionSlice& slice : func.slices) {
        for (ContinuationValue& continuation_out : slice.continuations_out) {
          if (!original_output_decomposable.at(&continuation_out)) {
            continue;
          }

          // The output is not decomposable if it feeds any parameter fed by a
          // direct-in
          for (const xls::Param* feeds_param :
               params_by_output.at(&continuation_out)) {
            if (!decomposable_params.at(feeds_param)) {
              original_output_decomposable[&continuation_out] = false;
              iter_changed = true;
              break;
            }
          }
        }
      }
    }
  }

  absl::flat_hash_map<ContinuationValue*,
                      absl::InlinedVector<ContinuationValue*, 1>>
      decomposed_cont_values_by_original_output;

  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::InlinedVector<xls::Node*, 1> new_returns;

    // Last slice has no continuation outputs
    if (&slice == &func.slices.back()) {
      XLSCC_CHECK(slice.continuations_out.empty(), loc);
      break;
    }

    xls::Node* return_tuple = slice.function->return_value();
    XLSCC_CHECK(return_tuple->Is<xls::Tuple>(), loc);
    const int64_t extra_returns =
        return_tuple->operand_count() - slice.continuations_out.size();
    XLSCC_CHECK_GE(extra_returns, 0, loc);

    // NOTE: This iterates over the output continuations while adding more at
    // the end
    std::vector<ContinuationValue*> original_continuations_out;
    original_continuations_out.reserve(slice.continuations_out.size());
    for (ContinuationValue& continuation_out : slice.continuations_out) {
      original_continuations_out.push_back(&continuation_out);
    }

    for (ContinuationValue* continuation_out_ptr : original_continuations_out) {
      ContinuationValue& continuation_out = *continuation_out_ptr;

      // Don't reference the output identity, so that it can be removed later.
      XLSCC_CHECK_EQ(continuation_out.output_node->op(), xls::Op::kIdentity,
                     loc);
      xls::Node* output_source = continuation_out.output_node->operand(0);

      if (!original_output_decomposable.at(&continuation_out)) {
        continue;
      }

      if (!TypeIsDecomposable(output_source->GetType())) {
        continue;
      }

      for (const xls::Param* param : params_by_output.at(&continuation_out)) {
        XLSCC_CHECK(decomposable_params.at(param), loc);
      }

      absl::InlinedVector<xls::Node*, 1> decomposed_nodes;
      XLS_ASSIGN_OR_RETURN(decomposed_nodes, DecomposeTuples(output_source));

      absl::InlinedVector<xls::Value, 1> decomposed_literals;
      if (continuation_out.literal.has_value()) {
        XLS_ASSIGN_OR_RETURN(decomposed_literals,
                             DecomposeValue(output_source->GetType(),
                                            *continuation_out.literal));
      }

      // Create decomposed outputs
      absl::InlinedVector<ContinuationValue*, 1> decomposed_cont_values;

      for (int64_t di = 0; di < decomposed_nodes.size(); ++di) {
        xls::Node* node = decomposed_nodes.at(di);
        ContinuationValue new_continuation_out = continuation_out;

        if (continuation_out.literal.has_value()) {
          new_continuation_out.literal = decomposed_literals.at(di);
        }

        absl::flat_hash_set<DeclLeaf> new_decls;
        for (const DeclLeaf& decl : new_continuation_out.decls) {
          new_decls.insert(DeclLeaf{.decl = decl.decl, .leaf_index = di});
        }
        new_continuation_out.decls = new_decls;

        XLS_ASSIGN_OR_RETURN(
            xls::UnOp * output_identity,
            slice.function->MakeNodeWithName<xls::UnOp>(
                loc, node, xls::Op::kIdentity,
                /*name=*/absl::StrFormat("%s_ident", node->GetName())));
        new_continuation_out.output_node = output_identity;
        new_returns.push_back(new_continuation_out.output_node);
        slice.continuations_out.push_back(new_continuation_out);
        decomposed_cont_values.push_back(&slice.continuations_out.back());
        changed = true;
      }

      decomposed_cont_values_by_original_output[&continuation_out] =
          decomposed_cont_values;
    }

    // Update return value
    if (!new_returns.empty()) {
      std::vector<xls::Node*> all_returns;
      all_returns.insert(all_returns.end(), return_tuple->operands().begin(),
                         return_tuple->operands().begin() +
                             return_tuple->operands().size() - extra_returns);
      all_returns.insert(all_returns.end(), new_returns.begin(),
                         new_returns.end());

      CHECK_EQ(all_returns.size(), slice.continuations_out.size());

      for (int64_t i = return_tuple->operand_count() - extra_returns;
           i < return_tuple->operand_count(); ++i) {
        all_returns.push_back(return_tuple->operand(i));
      }

      XLS_ASSIGN_OR_RETURN(
          xls::Node * new_return_node,
          slice.function->MakeNode<xls::Tuple>(loc, all_returns));
      XLS_RETURN_IF_ERROR(slice.function->set_return_value(new_return_node));
      XLS_RETURN_IF_ERROR(slice.function->RemoveNode(return_tuple));
      changed = true;
    }
  }

  // To accomodate phis, first decompose all slices' outputs, then all slices'
  // inputs
  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::flat_hash_map<const xls::Param*, absl::InlinedVector<xls::Param*, 1>>
        decomposed_params_by_original;
    absl::flat_hash_set<const xls::Param*> all_cont_params;
    std::vector<xls::Param*> all_params_decomposed;
    all_params_decomposed.reserve(slice.continuations_in.size());

    for (ContinuationInput& continuation_in : slice.continuations_in) {
      all_cont_params.insert(continuation_in.input_node->As<xls::Param>());
    }

    // Add decomposed continuation inputs
    // NOTE: This loop iterates over continuation inputs while adding more at
    // the end.
    std::vector<ContinuationInput*> original_continuations_in;
    original_continuations_in.reserve(slice.continuations_in.size());
    for (ContinuationInput& continuation_in : slice.continuations_in) {
      original_continuations_in.push_back(&continuation_in);
    }
    for (ContinuationInput* continuation_in_ptr : original_continuations_in) {
      ContinuationInput& continuation_in = *continuation_in_ptr;
      XLSCC_CHECK_EQ(continuation_in.input_node->op(), xls::Op::kParam, loc);
      xls::Param* input_param = continuation_in.input_node->As<xls::Param>();

      if (!decomposable_params.at(input_param)) {
        continue;
      }

      CHECK(TypeIsDecomposable(input_param->GetType()));
      CHECK(!continuation_in.continuation_out->direct_in);

      CHECK(original_output_decomposable.at(continuation_in.continuation_out));

      const absl::InlinedVector<ContinuationValue*, 1>& decomposed_cont_values =
          decomposed_cont_values_by_original_output.at(
              continuation_in.continuation_out);

      // Create params to replace original param, if not done already
      if (!decomposed_params_by_original.contains(input_param)) {
        decomposed_params_by_original[input_param] = {};

        all_params_decomposed.push_back(input_param);

        for (int64_t di = 0; di < decomposed_cont_values.size(); ++di) {
          const ContinuationValue& decomposed_value =
              *decomposed_cont_values.at(di);
          xls::Type* type = decomposed_value.output_node->GetType();
          XLSCC_CHECK(!TypeIsDecomposable(type), loc);

          std::string decomposed_name =
              absl::StrFormat("%s_%d", input_param->GetName(), di);

          xls::Node* decomposed_param_node =
              slice.function->AddNode(std::make_unique<xls::Param>(
                  loc, decomposed_value.output_node->GetType(),
                  /*name=*/decomposed_name, slice.function));

          xls::Param* decomposed_param =
              decomposed_param_node->As<xls::Param>();

          XLS_RETURN_IF_ERROR(slice.function->MoveParamToIndex(
              decomposed_param, all_cont_params.size()));

          decomposed_params_by_original[input_param].push_back(
              decomposed_param);
          all_cont_params.insert(decomposed_param);

          changed = true;
        }
      }

      // Create inputs to replace original input.
      const absl::InlinedVector<xls::Param*, 1>& decomposed_params =
          decomposed_params_by_original.at(input_param);

      for (int64_t di = 0; di < decomposed_cont_values.size(); ++di) {
        ContinuationValue* decomposed_cont_value =
            decomposed_cont_values.at(di);

        std::string decomposed_name =
            absl::StrFormat("%s_%d", continuation_in.name, di);

        ContinuationInput new_continuation_in = continuation_in;
        new_continuation_in.continuation_out = decomposed_cont_value;
        new_continuation_in.input_node = decomposed_params.at(di);
        new_continuation_in.name = decomposed_name;
        new_continuation_in.decls = decomposed_cont_value->decls;

        slice.continuations_in.push_back(new_continuation_in);
        changed = true;
      }
    }

    // Replace uses of original param with a tuple of new params.
    for (xls::Param* orig_param : all_params_decomposed) {
      const absl::InlinedVector<xls::Param*, 1>& decomposed_params =
          decomposed_params_by_original.at(orig_param);

      absl::InlinedVector<xls::Node*, 1> decomposed_param_nodes;
      decomposed_param_nodes.reserve(decomposed_params.size());
      for (xls::Param* param : decomposed_params) {
        decomposed_param_nodes.push_back(param);
      }

      XLS_ASSIGN_OR_RETURN(
          xls::Node * param_replace_tuple,
          ComposeTuples(orig_param->GetName(), orig_param->GetType(),
                        slice.function, loc, decomposed_param_nodes));

      XLS_RETURN_IF_ERROR(orig_param->ReplaceUsesWith(param_replace_tuple));
      changed = true;
    }
  }

  return absl::OkStatus();
}

absl::Status Translator::FinishLastSlice(TrackedBValue return_bval,
                                         const xls::SourceInfo& loc) {
  XLS_RETURN_IF_ERROR(FinishSlice(return_bval, loc));

  XLS_RETURN_IF_ERROR(RemoveMaskedOpParams(*context().sf, loc));

  // Direct-inness is used in optimization
  OptimizationContext optimization_context;
  XLS_RETURN_IF_ERROR(MarkDirectIns(*context().sf, optimization_context, loc));

  // Set ContinuationInput::choose_in_states from
  // NewFSMState::current_inputs_by_input_param. This allows phi selection order
  // to be preserved through optimization.
  NewFSMGenerator generator(*this, *this, DebugIrTraceFlags_None);
  XLS_RETURN_IF_ERROR(
      GenerateLayoutAndInsertChooseInStates(*context().sf, generator, loc));

  XLS_RETURN_IF_ERROR(
      OptimizeContinuations(*context().sf, optimization_context, loc));

  if (debug_ir_trace_flags_ & DebugIrTraceFlags_FSMStates) {
    LogContinuations(*context().sf);
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

absl::Status RemoveUnusedContinuationInputParams(GeneratedFunction& func,
                                                 OptimizationContext& context,
                                                 bool& changed,
                                                 const xls::SourceInfo& loc) {
  // Multiple inputs can share a parameter in the case of a phi /
  // feedback, so the already deleted parameters are tracked to remove
  // all of the inputs for an unused parameter.
  absl::flat_hash_set<const xls::Param*> deleted_params;

  for (GeneratedFunctionSlice& slice : func.slices) {
    XLS_ASSIGN_OR_RETURN(
        SourcesSetNodeInfo * node_info,
        context.GetSourcesSetNodeInfoForFunction(slice.function));

    ParamSet return_value_from_params =
        node_info->GetSingleInfoForNode(slice.function->return_value());

    for (auto cont_in_it = slice.continuations_in.begin();
         cont_in_it != slice.continuations_in.end();) {
      ContinuationInput& continuation_in = *cont_in_it;

      if (deleted_params.contains(continuation_in.input_node)) {
        cont_in_it = slice.continuations_in.erase(cont_in_it);
        changed = true;
        continue;
      }

      CHECK_EQ(continuation_in.input_node->function_base(), slice.function);
      // The parameter is in use, so skip this input.
      if (return_value_from_params.contains(continuation_in.input_node)) {
        ++cont_in_it;
        continue;
      }
      // There may still be uses like forming a tuple, the element of which is
      // never indexed
      XLS_RETURN_IF_ERROR(
          continuation_in.input_node
              ->ReplaceUsesWithNew<xls::Literal>(
                  xls::ZeroOfType(continuation_in.input_node->GetType()))
              .status());

      XLS_RETURN_IF_ERROR(
          slice.function->RemoveNode(continuation_in.input_node));

      deleted_params.insert(continuation_in.input_node);
      cont_in_it = slice.continuations_in.erase(cont_in_it);
      changed = true;
    }
  }
  return absl::OkStatus();
}

// For each continuation value, finds the whole parameters that feed it.
// For a simple pass through, there will just be one. However, for example,
// in the case of a select, there could be several.
//
// Whole parameter means that the entire value is passed through. For example,
// a tuple parameter's entire tuple value must be passed through, with all
// elements in the original positions.
absl::StatusOr<absl::flat_hash_map<const ContinuationValue*,
                                   absl::flat_hash_set<const xls::Param*>>>
FindPassThroughs(GeneratedFunction& func, OptimizationContext& context) {
  absl::flat_hash_map<const ContinuationValue*,
                      absl::flat_hash_set<const xls::Param*>>
      ret;
  for (GeneratedFunctionSlice& slice : func.slices) {
    XLS_ASSIGN_OR_RETURN(
        SourcesSetTreeNodeInfo * node_sources_info,
        context.GetSourcesSetTreeNodeInfoForFunction(slice.function));

    absl::flat_hash_set<const xls::Param*> continuation_params;
    for (ContinuationInput& continuation_in : slice.continuations_in) {
      continuation_params.insert(continuation_in.input_node);
    }

    for (ContinuationValue& continuation_out : slice.continuations_out) {
      CHECK(continuation_out.output_node->op() == xls::Op::kIdentity);

      const xls::SharedLeafTypeTree<NodeSourceSetPtr>& sources =
          node_sources_info->GetInfo(continuation_out.output_node);

      // First find all the continuation params
      absl::flat_hash_set<xls::Param*> allowed_sources;

      bool disallowed = false;
      XLS_RETURN_IF_ERROR(xls::leaf_type_tree::ForEachIndex(
          sources.AsView(),
          [&](xls::Type* element_type, const NodeSourceSetPtr& source_set,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            for (const xls::NodeSource& source : *source_set) {
              xls::Node* source_node = source.node();

              if (source.tree_index() != tree_index) {
                disallowed = true;
                break;
              }

              if (!source_node->Is<xls::Param>()) {
                disallowed = true;
                break;
              }

              // Check that param has the same number of elements as
              // continuation_out. This avoids marking slices as pass-throughs.
              const xls::LeafTypeTree<std::monostate>& param_tree =
                  context.GetBlankTypeTreeForType(source_node->GetType());
              if (param_tree.elements().size() != sources.elements().size()) {
                disallowed = true;
                break;
              }

              xls::Param* from_param = source_node->As<xls::Param>();
              if (!continuation_params.contains(from_param)) {
                disallowed = true;
                break;
              }
              allowed_sources.insert(from_param);
            }
            return absl::OkStatus();
          }));

      // Ensure that every element contains all the sources
      absl::flat_hash_map<const xls::Param*, xls::LeafTypeTree<std::monostate>>
          source_type_trees;
      XLS_RETURN_IF_ERROR(xls::leaf_type_tree::ForEachIndex(
          sources.AsView(),
          [&](xls::Type* element_type, const NodeSourceSetPtr& source_set,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            for (xls::Param* allowed_source : allowed_sources) {
              std::vector<int64_t> tree_index_vec(tree_index.begin(),
                                                  tree_index.end());
              if (!source_set->contains(
                      xls::NodeSource(allowed_source, tree_index_vec))) {
                disallowed = true;
                break;
              }

              if (!disallowed) {
                // Check the leaf element type, caching the type tree to avoid
                // repeated work.
                auto [it, _] = source_type_trees.try_emplace(
                    allowed_source, allowed_source->GetType());
                CHECK(it->second.AsView(tree_index)
                          .type()
                          ->IsEqualTo(element_type));
              }
            }
            return absl::OkStatus();
          }));

      if (disallowed || allowed_sources.empty()) {
        continue;
      }

      ret[&continuation_out].clear();
      ret[&continuation_out].insert(allowed_sources.begin(),
                                    allowed_sources.end());
    }
  }

  return ret;
}

absl::Status RemovePassThroughs(GeneratedFunction& func, bool& changed,
                                OptimizationContext& context,
                                const xls::SourceInfo& loc,
                                xls::Package* package,
                                xls::OptimizationContext& xls_opt_context) {
  // This pass doesn't actually change the function, so we can calculate this
  // once here
  absl::flat_hash_map<const ContinuationValue*,
                      absl::flat_hash_set<const xls::Param*>>
      pass_throughs;

  absl::flat_hash_map<const xls::Param*, std::vector<ContinuationInput*>>
      continuation_inputs_by_input_node;
  absl::flat_hash_map<const xls::Node*, std::vector<ContinuationInput*>>
      continuation_inputs_by_output_node;
  absl::flat_hash_map<ContinuationInput*, GeneratedFunctionSlice*>
      slice_by_continuation_input;
  absl::flat_hash_map<ContinuationValue*, GeneratedFunctionSlice*>
      slice_by_continuation_output;
  absl::flat_hash_map<GeneratedFunctionSlice*, int64_t> slice_indices;

  XLS_ASSIGN_OR_RETURN(pass_throughs, FindPassThroughs(func, context));

  for (GeneratedFunctionSlice& slice : func.slices) {
    slice_indices[&slice] = slice_indices.size();

    for (ContinuationInput& continuation_in : slice.continuations_in) {
      slice_by_continuation_input[&continuation_in] = &slice;
      continuation_inputs_by_output_node[continuation_in.continuation_out
                                             ->output_node]
          .push_back(&continuation_in);
      continuation_inputs_by_input_node[continuation_in.input_node].push_back(
          &continuation_in);
    }
    for (ContinuationValue& continuation_out : slice.continuations_out) {
      slice_by_continuation_output[&continuation_out] = &slice;
    }
  }

  for (GeneratedFunctionSlice& slice : func.slices) {
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      CHECK(continuation_out.output_node->op() == xls::Op::kIdentity);

      CHECK_GT(continuation_out.output_node->GetType()->GetFlatBitCount(), 0);

      if (!pass_throughs.contains(&continuation_out)) {
        continue;
      }

      const absl::flat_hash_set<const xls::Param*>& pass_in_params =
          pass_throughs.at(&continuation_out);

      if (pass_in_params.size() != 1) {
        continue;
      }

      const xls::Param* pass_in_param = *pass_in_params.begin();

      // If we reach here, then this output is fed directly from an input.
      // Therefore it is safe to redirect the inputs fed from this output
      // to the previous output.
      //
      // Output will get removed by other pass if it is now unused.
      // Input will get removed by other pass now that it is unused.

      // No downstream inputs
      if (!continuation_inputs_by_output_node.contains(
              continuation_out.output_node)) {
        continue;
      }
      // Copy this so that we can mutate the maps
      std::vector<ContinuationInput*> pass_through_to_inputs =
          continuation_inputs_by_output_node.at(continuation_out.output_node);

      CHECK_GE(pass_through_to_inputs.size(), 1);

      const int64_t current_slice_index = slice_indices.at(&slice);

      // In the case of phis, optimization can end up with a slice passing
      // through to itself. Leave these alone.
      bool self_feedback = false;
      for (ContinuationInput* pass_through_to_input : pass_through_to_inputs) {
        GeneratedFunctionSlice* downstream_slice =
            slice_by_continuation_input.at(pass_through_to_input);

        if (downstream_slice != &slice) {
          continue;
        }

        self_feedback = true;
        break;
      }
      if (self_feedback) {
        continue;
      }

      pass_through_to_inputs =
          continuation_inputs_by_output_node.at(continuation_out.output_node);

      // Get all the inputs that use this parameter
      // Copy this so that we can mutate the maps
      const std::vector<ContinuationInput*> this_slice_inputs =
          continuation_inputs_by_input_node.at(pass_in_param);

      CHECK(!this_slice_inputs.empty());

      // Skip if there are multiple upstream inputs currently (let other
      // optimizations apply)
      int64_t num_upstream_inputs_this_param = 0;
      for (const ContinuationInput* this_slice_input : this_slice_inputs) {
        const GeneratedFunctionSlice* upstream_slice =
            slice_by_continuation_output.at(this_slice_input->continuation_out);
        const int64_t upstream_slice_index = slice_indices.at(upstream_slice);
        if (upstream_slice_index < current_slice_index) {
          ++num_upstream_inputs_this_param;
        }
      }
      if (num_upstream_inputs_this_param > 1) {
        continue;
      }

      struct UpstreamInput {
        const ContinuationInput* upstream_input = nullptr;
        bool do_insert_downstream = false;
        absl::btree_set<StateId> choose_in_states;
      };

      // Slice indices will vary, jump states matter
      auto only_jump_states = [](const absl::btree_set<StateId>& states) {
        absl::btree_set<StateId> ret;
        for (const StateId& state : states) {
          ret.insert(StateId{
              .slice_index = -1,
              .from_jump_slice_indices = state.from_jump_slice_indices});
        }
        return ret;
      };

      absl::flat_hash_map<const ContinuationInput*, std::vector<UpstreamInput>>
          upstream_inputs_by_downstream_input;
      absl::flat_hash_map<const ContinuationInput*, absl::btree_set<StateId>>
          missing_states_by_downstream_input;

      // Filter downstream inputs
      for (ContinuationInput* pass_through_to_input_ptr :
           pass_through_to_inputs) {
        absl::btree_set<StateId> all_upstream_jump_states;

        const GeneratedFunctionSlice* pass_through_to_slice =
            slice_by_continuation_input.at(pass_through_to_input_ptr);
        const int64_t pass_through_to_slice_index =
            slice_indices.at(pass_through_to_slice);

        absl::btree_set<StateId> downstream_jump_states =
            only_jump_states(pass_through_to_input_ptr->choose_in_states);

        bool disallow_due_to_feedback = false;

        for (const ContinuationInput* this_slice_input : this_slice_inputs) {
          absl::btree_set<StateId> upstream_jump_states =
              only_jump_states(this_slice_input->choose_in_states);

          const GeneratedFunctionSlice* upstream_slice =
              slice_by_continuation_output.at(
                  this_slice_input->continuation_out);
          const int64_t upstream_slice_index = slice_indices.at(upstream_slice);

          all_upstream_jump_states.insert(upstream_jump_states.begin(),
                                          upstream_jump_states.end());

          // Check feedbacks

          const bool is_feedback =
              current_slice_index >= pass_through_to_slice_index ||
              upstream_slice_index >= current_slice_index;

          const bool will_be_feedback =
              upstream_slice_index >= pass_through_to_slice_index;

          // If it contained a feedback, then it must be a feedback after change
          // This is due to a limitation in the expressiveness of the FSM
          // data flow graphs. Feedback values are seen in the next activation,
          // not the current one. When a feedback becomes a non-feedback,
          // its next value can be seen within the same activation as it was
          // produced, which is incorrect.
          if (is_feedback && !will_be_feedback) {
            disallow_due_to_feedback = true;
            break;
          }
        }

        if (disallow_due_to_feedback) {
          continue;
        }

        // Add choose in states from existing downstream inputs sharing the same
        // parameter. The pass through input may not be the only one on this
        // parameter. Avoids duplicate choose in states
        for (const ContinuationInput& pass_through_to_slice_input :
             pass_through_to_slice->continuations_in) {
          if (pass_through_to_slice_input.input_node !=
                  pass_through_to_input_ptr->input_node ||
              &pass_through_to_slice_input == pass_through_to_input_ptr) {
            continue;
          }
          absl::btree_set<StateId> jump_states =
              only_jump_states(pass_through_to_slice_input.choose_in_states);
          all_upstream_jump_states.insert(jump_states.begin(),
                                          jump_states.end());
        }

        absl::btree_set<StateId> missing_states;
        // Missing states are present downstream but not upstream
        for (const StateId& downstream_jump_state : downstream_jump_states) {
          if (!all_upstream_jump_states.contains(downstream_jump_state)) {
            missing_states.insert(downstream_jump_state);
          }
        }

        // If there is no upstream input, then there's no input to which to
        // assign missing states.
        CHECK_EQ(num_upstream_inputs_this_param, 1);

        missing_states_by_downstream_input[pass_through_to_input_ptr] =
            missing_states;

        for (const ContinuationInput* this_slice_input : this_slice_inputs) {
          upstream_inputs_by_downstream_input[pass_through_to_input_ptr]
              .push_back(UpstreamInput{.upstream_input = this_slice_input});
        }
      }

      // Nothing to do, all downstreams disqualified
      if (upstream_inputs_by_downstream_input.empty()) {
        continue;
      }

      absl::flat_hash_set<const ContinuationInput*>
          pass_through_inputs_found_upstream_input;

      // Filter upstream inputs
      for (auto& [pass_through_to_input, upstream_inputs] :
           upstream_inputs_by_downstream_input) {
        for (UpstreamInput& upstream_input : upstream_inputs) {
          upstream_input.do_insert_downstream = true;

          absl::btree_set<StateId> upstream_jump_states =
              only_jump_states(upstream_input.upstream_input->choose_in_states);
          const absl::btree_set<StateId> pass_through_jump_states =
              only_jump_states(pass_through_to_input->choose_in_states);

          // Assign missing states to upstream input
          const int64_t upstream_slice_index =
              slice_indices.at(slice_by_continuation_output.at(
                  upstream_input.upstream_input->continuation_out));

          if (upstream_slice_index < current_slice_index) {
            CHECK(!pass_through_inputs_found_upstream_input.contains(
                pass_through_to_input));
            pass_through_inputs_found_upstream_input.insert(
                pass_through_to_input);
            const absl::btree_set<StateId>& missing_states =
                missing_states_by_downstream_input.at(pass_through_to_input);
            upstream_jump_states.insert(missing_states.begin(),
                                        missing_states.end());
          }

          // Filter upstream inputs by the set of states for the downstream
          // input.
          absl::btree_set<StateId> state_intersection;
          std::set_intersection(
              upstream_jump_states.begin(), upstream_jump_states.end(),
              pass_through_jump_states.begin(), pass_through_jump_states.end(),
              std::inserter(state_intersection, state_intersection.begin()));

          // Preserve upstream states in new downstream inputs
          // But update target slice index
          const int64_t pass_through_to_slice_index = slice_indices.at(
              slice_by_continuation_input.at(pass_through_to_input));

          upstream_input.choose_in_states.clear();
          for (const StateId& state : state_intersection) {
            upstream_input.choose_in_states.insert(StateId{
                .slice_index = pass_through_to_slice_index,
                .from_jump_slice_indices = state.from_jump_slice_indices,
            });
          }
        }
      }

      // For each downstream input, delete the current downstream input,
      // and replace (in the same place in the list) with new downstream inputs
      // based on the upstream inputs.
      for (auto& [pass_through_to_input_ptr, upstream_inputs] :
           upstream_inputs_by_downstream_input) {
        const bool insert_any_downstream =
            std::any_of(upstream_inputs.begin(), upstream_inputs.end(),
                        [](const UpstreamInput& upstream_input) {
                          return upstream_input.do_insert_downstream;
                        });
        CHECK(insert_any_downstream);

        GeneratedFunctionSlice* downstream_slice =
            slice_by_continuation_input.at(pass_through_to_input_ptr);

        const ContinuationInput pass_through_to_input_org =
            *pass_through_to_input_ptr;

        auto insert_it = downstream_slice->continuations_in.begin();

        // Delete the original downstream input
        bool erased = false;
        for (; insert_it != downstream_slice->continuations_in.end();
             ++insert_it) {
          if (&*insert_it == pass_through_to_input_ptr) {
            auto n_erased =
                std::erase(continuation_inputs_by_input_node.at(
                               pass_through_to_input_ptr->input_node),
                           pass_through_to_input_ptr);
            CHECK_EQ(n_erased, 1);

            n_erased = std::erase(
                continuation_inputs_by_output_node.at(
                    pass_through_to_input_ptr->continuation_out->output_node),
                pass_through_to_input_ptr);
            CHECK_EQ(n_erased, 1);

            slice_by_continuation_input.erase(pass_through_to_input_ptr);

            insert_it = downstream_slice->continuations_in.erase(insert_it);
            erased = true;
            break;
          }
        }

        CHECK(erased);

        changed = true;

        // The first input can simply be forwarded without creating new
        // downstream inputs
        for (const UpstreamInput& upstream_input : upstream_inputs) {
          if (!upstream_input.do_insert_downstream) {
            continue;
          }

          const ContinuationInput* this_slice_input =
              upstream_input.upstream_input;
          CHECK(slice_by_continuation_input.contains(this_slice_input));

          CHECK_NE(pass_through_to_input_org.input_node, pass_in_param);
          CHECK_NE(this_slice_input->continuation_out, &continuation_out);

          ContinuationInput new_input = pass_through_to_input_org;

          new_input.continuation_out = this_slice_input->continuation_out;

          new_input.choose_in_states = upstream_input.choose_in_states;

          CHECK(new_input.continuation_out->output_node->GetType()->IsEqualTo(
              new_input.input_node->GetType()));

          // Update iterator position for this downstream input
          insert_it =
              downstream_slice->continuations_in.insert(insert_it, new_input);

          ContinuationInput* new_input_ptr = &*insert_it;

          slice_by_continuation_input[new_input_ptr] = downstream_slice;
          continuation_inputs_by_input_node[new_input_ptr->input_node]
              .push_back(new_input_ptr);
          continuation_inputs_by_output_node[new_input_ptr->continuation_out
                                                 ->output_node]
              .push_back(new_input_ptr);
          changed = true;
        }
      }
    }  // slice.continuations_out
  }  // func.slices

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
    struct InputSet {
      absl::InlinedVector<std::list<ContinuationInput>::iterator, 2>
          input_iterators;
    };

    absl::btree_map<InputKey, InputSet> inputs_by_key;

    for (auto cont_in_it = slice.continuations_in.begin();
         cont_in_it != slice.continuations_in.end(); ++cont_in_it) {
      const ContinuationInput& continuation_in = *cont_in_it;
      InputKey key = {continuation_in.input_node,
                      continuation_in.continuation_out};
      inputs_by_key[key].input_iterators.push_back(cont_in_it);
    }

    // Delete all but the first
    for (auto& [key, input_set] : inputs_by_key) {
      CHECK_GT(input_set.input_iterators.size(), 0L);
      // Skip extra work if there's already only one input
      if (input_set.input_iterators.size() == 1) {
        continue;
      }
      absl::btree_set<StateId> all_choose_in_states;
      for (std::list<ContinuationInput>::iterator input_it :
           input_set.input_iterators) {
        all_choose_in_states.insert(input_it->choose_in_states.begin(),
                                    input_it->choose_in_states.end());
      }

      CHECK_GT(input_set.input_iterators.size(), 1L);

      bool first = true;
      for (auto input_it : input_set.input_iterators) {
        if (first) {
          first = false;
          input_it->choose_in_states = all_choose_in_states;
          continue;
        }
        slice.continuations_in.erase(input_it);
        changed = true;
      }
    }
  }

  return absl::OkStatus();
}

struct ParamInputs {
  bool multiple_upstream_inputs = false;
  ContinuationInput* upstream_input = nullptr;
  absl::InlinedVector<ContinuationInput*, 1> all_inputs = {};
};

// May include value in the output if it's not a pass-through.
void GetNonPassthroughUpstreamValuesForValue(
    const ContinuationValue* value,
    absl::flat_hash_set<const ContinuationValue*>& non_passthrough_values_out,
    const absl::flat_hash_map<const xls::Param*, ParamInputs>&
        all_continuation_params,
    const absl::flat_hash_map<const ContinuationValue*,
                              absl::flat_hash_set<const xls::Param*>>&
        pass_throughs,
    absl::flat_hash_set<const ContinuationValue*>& visited) {
  if (visited.contains(value)) {
    return;
  }

  visited.insert(value);

  if (!pass_throughs.contains(value)) {
    non_passthrough_values_out.insert(value);
    return;
  }

  for (const xls::Param* pass_through_param : pass_throughs.at(value)) {
    const ParamInputs& param_inputs =
        all_continuation_params.at(pass_through_param);

    for (const ContinuationInput* input : param_inputs.all_inputs) {
      GetNonPassthroughUpstreamValuesForValue(
          input->continuation_out, non_passthrough_values_out,
          all_continuation_params, pass_throughs, visited);
    }
  }
}

// If all feedbacks provide the same value as the feedforward input,
// then they can all be removed.
//
// It is not safe to remove one feedback if another provides a different value,
// as this can change the sequence of values seen by the input parameter.
absl::Status RemovePassthroughFeedbacks(GeneratedFunction& func, bool& changed,
                                        OptimizationContext& context,
                                        const xls::SourceInfo& loc) {
  // This pass doesn't actually change the function, so we can calculate this
  // once here.
  absl::flat_hash_map<const ContinuationValue*,
                      absl::flat_hash_set<const xls::Param*>>
      pass_throughs;
  XLS_ASSIGN_OR_RETURN(pass_throughs, FindPassThroughs(func, context));

  // This pass only modifies continuation inputs, so this can be calculated
  // once here.
  absl::flat_hash_map<const ContinuationValue*, int64_t>
      slice_index_by_continuation_out;
  absl::flat_hash_map<const GeneratedFunctionSlice*, int64_t>
      slice_index_by_slice;

  for (GeneratedFunctionSlice& slice : func.slices) {
    const int64_t slice_index = slice_index_by_slice.size();
    slice_index_by_slice[&slice] = slice_index;
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      slice_index_by_continuation_out[&continuation_out] = slice_index;
    }
  }

  absl::flat_hash_map<const GeneratedFunctionSlice*,
                      absl::flat_hash_set<const xls::Param*>>
      params_per_slice;
  absl::flat_hash_map<const xls::Param*, ParamInputs> all_continuation_params;

  for (GeneratedFunctionSlice& slice : func.slices) {
    params_per_slice[&slice] = {};
    const int64_t slice_index = slice_index_by_slice.at(&slice);

    for (ContinuationInput& continuation_in : slice.continuations_in) {
      params_per_slice[&slice].insert(continuation_in.input_node);

      ParamInputs& param_inputs =
          all_continuation_params[continuation_in.input_node];

      const int64_t upstream_slice_index =
          slice_index_by_continuation_out.at(continuation_in.continuation_out);

      const bool is_feedback = slice_index <= upstream_slice_index;

      param_inputs.all_inputs.push_back(&continuation_in);

      if (!is_feedback) {
        if (param_inputs.upstream_input != nullptr) {
          param_inputs.multiple_upstream_inputs = true;
        }
        param_inputs.upstream_input = &continuation_in;
      }
    }
  }

  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::flat_hash_set<const ContinuationInput*> inputs_to_remove;

    for (const xls::Param* param : params_per_slice.at(&slice)) {
      ParamInputs& param_inputs = all_continuation_params.at(param);
      CHECK(!param_inputs.all_inputs.empty());

      CHECK_NE(param_inputs.upstream_input, nullptr);

      if (param_inputs.multiple_upstream_inputs) {
        continue;
      }
      if (param_inputs.all_inputs.size() == 1) {
        continue;
      }

      absl::flat_hash_set<const ContinuationValue*> non_passthrough_values;

      for (const ContinuationInput* input : param_inputs.all_inputs) {
        if (input == param_inputs.upstream_input) {
          continue;
        }

        absl::flat_hash_set<const ContinuationValue*> visited;
        GetNonPassthroughUpstreamValuesForValue(
            input->continuation_out, non_passthrough_values,
            all_continuation_params, pass_throughs, visited);
      }

      if (non_passthrough_values.size() != 1) {
        continue;
      }

      if (*non_passthrough_values.begin() !=
          param_inputs.upstream_input->continuation_out) {
        continue;
      }

      absl::btree_set<StateId> pass_thru_choose_in_states;

      for (const ContinuationInput* input : param_inputs.all_inputs) {
        if (input == param_inputs.upstream_input) {
          continue;
        }

        for (const StateId& state_id : input->choose_in_states) {
          pass_thru_choose_in_states.insert(state_id);
        }

        inputs_to_remove.insert(input);
      }

      param_inputs.all_inputs = {param_inputs.upstream_input};

      param_inputs.upstream_input->choose_in_states.insert(
          pass_thru_choose_in_states.begin(), pass_thru_choose_in_states.end());
    }

    const int64_t num_inputs_before = slice.continuations_in.size();

    for (auto it = slice.continuations_in.begin();
         it != slice.continuations_in.end();) {
      const ContinuationInput& continuation_in = *it;

      if (!inputs_to_remove.contains(&continuation_in)) {
        ++it;
        continue;
      }

      it = slice.continuations_in.erase(it);
      changed = true;
    }

    const int64_t num_inputs_after = slice.continuations_in.size();

    CHECK_EQ(inputs_to_remove.size(), num_inputs_before - num_inputs_after);
  }
  return absl::OkStatus();
}

absl::Status RemoveDuplicateParams(GeneratedFunction& func, bool& changed,
                                   const xls::SourceInfo& loc) {
  typedef std::tuple<const ContinuationValue*, absl::btree_set<StateId>>
      InputTuple;

  for (GeneratedFunctionSlice& slice : func.slices) {
    absl::flat_hash_map<xls::Param*, absl::btree_set<InputTuple>>
        upstream_values_by_param;

    absl::flat_hash_map<xls::Param*, std::vector<ContinuationInput*>>
        continuation_inputs_by_param;

    absl::btree_map<absl::btree_set<InputTuple>,
                    absl::flat_hash_set<xls::Param*>>
        params_by_upstream_values;

    auto update_maps = [&]() {
      upstream_values_by_param.clear();
      continuation_inputs_by_param.clear();
      params_by_upstream_values.clear();

      for (ContinuationInput& continuation_in : slice.continuations_in) {
        InputTuple new_tuple(continuation_in.continuation_out,
                             continuation_in.choose_in_states);
        upstream_values_by_param[continuation_in.input_node].insert(new_tuple);
        continuation_inputs_by_param[continuation_in.input_node].push_back(
            &continuation_in);
      }

      for (const auto& [param, upstream_values] : upstream_values_by_param) {
        params_by_upstream_values[upstream_values].insert(param);
      }
    };

    update_maps();

    // Determinism is ensured by the ordering of these loops
    absl::flat_hash_set<xls::Param*> params_processed;

    for (ContinuationInput& continuation_in : slice.continuations_in) {
      bool changed_this_input = false;

      xls::Param* this_param = continuation_in.input_node;
      if (params_processed.contains(this_param)) {
        continue;
      }
      const absl::btree_set<InputTuple>& this_param_upstream_values =
          upstream_values_by_param.at(this_param);
      const absl::flat_hash_set<xls::Param*>& params_for_upstream_values =
          params_by_upstream_values.at(this_param_upstream_values);

      CHECK(params_for_upstream_values.contains(this_param));
      if (params_for_upstream_values.size() == 1) {
        continue;
      }
      params_processed.insert(this_param);

      // Rewrite continuation inputs for other params
      for (const xls::Param* other_param : params_for_upstream_values) {
        if (other_param == this_param) {
          continue;
        }
        const std::vector<ContinuationInput*>& continuation_inputs_other_param =
            continuation_inputs_by_param.at(other_param);

        for (ContinuationInput* other_continuation_in :
             continuation_inputs_other_param) {
          CHECK(other_continuation_in->input_node->GetType()->IsEqualTo(
              this_param->GetType()));

          other_continuation_in->input_node = this_param;
          continuation_in.decls.insert(other_continuation_in->decls.begin(),
                                       other_continuation_in->decls.end());

          changed_this_input = true;
        }
      }

      // Replace users of other params
      for (xls::Param* other_param : params_for_upstream_values) {
        if (other_param == this_param) {
          continue;
        }

        CHECK(other_param->GetType()->IsEqualTo(this_param->GetType()));

        XLS_RETURN_IF_ERROR(other_param->ReplaceUsesWith(this_param));

        // Remove the now-unused parameter
        XLS_RETURN_IF_ERROR(
            other_param->function_base()->RemoveNode(other_param));

        params_processed.insert(other_param);
        changed_this_input = true;
      }

      if (changed_this_input) {
        update_maps();
      }

      changed = changed || changed_this_input;
    }
  }

  return absl::OkStatus();
}

}  // namespace

// Note that literals are also propagated as continuations are created
// but none are propagated into pipelined loops, as it isn't known until
// all slices are generated whether or not a phi will be at a given input.
absl::Status Translator::SubstituteLiterals(GeneratedFunction& func,
                                            bool& changed,
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

absl::Status Translator::OptimizeContinuations(GeneratedFunction& func,
                                               OptimizationContext& context,
                                               const xls::SourceInfo& loc) {
  bool changed = true;
  xls::OptimizationContext xls_opt_context;

  do {
    do {
      changed = false;
      XLS_RETURN_IF_ERROR(
          RemoveUnusedContinuationInputParams(func, context, changed, loc));
      XLS_RETURN_IF_ERROR(RemoveUnusedContinuationOutputs(func, changed, loc));
      XLS_RETURN_IF_ERROR(RemovePassThroughs(func, changed, context, loc,
                                             package_, xls_opt_context));
      XLS_RETURN_IF_ERROR(
          RemoveDeadCode(func, changed, package_, xls_opt_context, loc));
      XLS_RETURN_IF_ERROR(RemoveDuplicateInputs(func, changed, loc));
      XLS_RETURN_IF_ERROR(RemoveDuplicateParams(func, changed, loc));
      XLS_RETURN_IF_ERROR(
          RemovePassthroughFeedbacks(func, changed, context, loc));
      XLS_RETURN_IF_ERROR(SubstituteLiterals(func, changed, loc));
    } while (changed);

    // For efficiency's sake, do a round of optimization before decomposing
    // Decompose relies on other passes to clean up
    XLS_RETURN_IF_ERROR(DecomposeContinuationValues(func, changed, loc));
  } while (changed);

  XLS_RETURN_IF_ERROR(ValidateContinuations(func, loc));

  return absl::OkStatus();
}

absl::Status Translator::GetDirectInSourcesForSlice(
    const GeneratedFunctionSlice& slice, bool first_slice,
    absl::flat_hash_set<const xls::Param*>& output) {
  absl::flat_hash_map<const xls::Param*, const ContinuationInput*>
      continuation_inputs_by_param;

  for (const ContinuationInput& continuation_in : slice.continuations_in) {
    continuation_inputs_by_param[continuation_in.input_node] = &continuation_in;
  }

  absl::flat_hash_set<std::string> static_param_names;

  for (const SideEffectingParameter& side_effecting_param :
       slice.side_effecting_parameters) {
    if (side_effecting_param.type == SideEffectingParameterType::kStatic) {
      static_param_names.insert(side_effecting_param.param_name);
    }
  }

  for (int64_t p = 0; p < slice.function->params().size(); ++p) {
    const xls::Param* param = slice.function->params().at(p);

    // Don't mark statics direct-in
    if (static_param_names.contains(param->name())) {
      continue;
    }

    // All first slice inputs are safe to be treated as direct in
    // (this, static, actual direct-ins)
    if (!first_slice) {
      // Continuations then IO op, which should not be treated as direct-in
      if (!continuation_inputs_by_param.contains(param)) {
        continue;
      }

      if (!continuation_inputs_by_param.at(param)
               ->continuation_out->direct_in) {
        continue;
      }
    }

    output.insert(param);
  }
  return absl::OkStatus();
}

absl::Status Translator::MarkDirectIns(GeneratedFunction& func,
                                       OptimizationContext& context,
                                       const xls::SourceInfo& loc) {
  XLSCC_CHECK(!func.slices.empty(), loc);

  absl::flat_hash_set<const xls::Param*> direct_in_sources;

  absl::flat_hash_set<const ContinuationValue*> outputs_to_upstream;

  bool first_slice = true;

  for (GeneratedFunctionSlice& slice : func.slices) {
    XLS_RETURN_IF_ERROR(GetDirectInSourcesForSlice(
        slice, /*first_slice=*/first_slice, direct_in_sources));

    // Input to a slice from itself is feedback
    for (ContinuationInput& continuation_in : slice.continuations_in) {
      outputs_to_upstream.insert(continuation_in.continuation_out);
    }

    for (ContinuationValue& continuation_out : slice.continuations_out) {
      if (outputs_to_upstream.contains(&continuation_out)) {
        continuation_out.direct_in = false;
        continue;
      }

      // Don't count things that don't actually use any direct-ins as direct-in,
      // for example literals. Allow literal substitution pass to prevent these
      // from being stored in state elements.
      XLS_ASSIGN_OR_RETURN(
          continuation_out.direct_in,
          context.CheckNodeSourcesInSet(
              slice.function, continuation_out.output_node, direct_in_sources,
              /*allow_empty_sources_result=*/false));
    }

    first_slice = false;
  }

  return absl::OkStatus();
}

std::string Debug_GenerateSliceGraph(const GeneratedFunction& func) {
  const bool show_choose_in_states = false;
  const bool show_input_debug_info = false;

  // Safe because params aren't changed/removed, nor are continuation outputs
  absl::flat_hash_map<const ContinuationValue*,
                      absl::flat_hash_set<const xls::Param*>>
      pass_throughs;
  OptimizationContext context;
  auto pass_through_ret =
      FindPassThroughs(const_cast<GeneratedFunction&>(func), context);
  CHECK_OK(pass_through_ret);
  pass_throughs = *pass_through_ret;

  // Pointers are used as names, as labels are not always unique
  std::vector<std::string> node_names;
  std::vector<std::string> rank_orders;
  std::vector<std::string> nodes_in_ranks;
  std::vector<std::string> nodes_with_edges;

  std::string last_rank_name = "";

  int64_t slice_index = -1;

  absl::flat_hash_map<int64_t, const GeneratedFunctionSlice*> slice_by_index;
  for (const GeneratedFunctionSlice& slice : func.slices) {
    ++slice_index;
    slice_by_index[slice_index] = &slice;
  }

  slice_index = -1;
  for (const GeneratedFunctionSlice& slice : func.slices) {
    ++slice_index;

    std::string new_rank = "(first)";

    if (slice_index > 0) {
      if (slice.after_op != nullptr) {
        new_rank = "after_" + Debug_OpName(*slice.after_op);
      } else {
        CHECK_LT(slice_index, (func.slices.size() - 1));
        const IOOp* before_op = slice_by_index.at(slice_index + 1)->after_op;
        CHECK_NE(before_op, nullptr);
        new_rank = "before_" + Debug_OpName(*before_op);
      }
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
                            "[%i] %s inputs", slice_index, new_rank))));
    node_names.push_back(
        absl::StrFormat("  %s [label=%s];", rank_output_name,
                        GraphvizEscape(absl::StrFormat(
                            "[%i] %s outputs", slice_index, new_rank))));

    last_rank_name = rank_output_name;

    std::vector<std::string> nodes_in_input_rank = {rank_input_name};
    std::vector<std::string> nodes_in_output_rank = {rank_output_name};

    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      const std::string output_name =
          GraphvizEscape(absl::StrFormat("%p", &continuation_out));
      const std::string type_str = Debug_GenerateReadableTypeName(
          continuation_out.output_node->GetType());

      std::string label = "";
      if (show_input_debug_info) {
        label = absl::StrFormat("%s : %s d %i p %i l %i", continuation_out.name,
                                type_str, (int)continuation_out.direct_in,
                                (int)pass_throughs.contains(&continuation_out),
                                (int)continuation_out.literal.has_value());
      } else {
        label = absl::StrFormat("%s : %s", continuation_out.name, type_str);
      }
      node_names.push_back(absl::StrFormat("  %s [label=%s];", output_name,
                                           GraphvizEscape(label)));

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
          "  %s -> %s [label=\"%s\"]",
          GraphvizEscape(
              absl::StrFormat("%p", continuation_in.continuation_out)),
          GraphvizEscape(absl::StrFormat("%p", continuation_in.input_node)),
          show_choose_in_states
              ? absl::StrJoin(continuation_in.choose_in_states, "; ",
                              [](std::string* out, const StateId& t) {
                                absl::StrAppend(out, t.ToString());
                              })
              : ""));
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

    // Continuation params come first
    absl::flat_hash_map<const xls::Param*, int64_t>
        continuation_in_count_by_param;

    for (const ContinuationInput& continuation_in : slice.continuations_in) {
      ++continuation_in_count_by_param[continuation_in.input_node];
    }

    auto continuation_in_it = slice.continuations_in.begin();
    for (int64_t i = 0; continuation_in_it != slice.continuations_in.end();
         ++i, ++continuation_in_it) {
      XLSCC_CHECK_EQ(slice.function->params().at(i),
                     continuation_in_it->input_node, loc);

      TrackedBValue prev_slice_val =
          prev_slice_ret.at(continuation_in_it->continuation_out);
      XLSCC_CHECK(prev_slice_val.valid(), loc);

      XLSCC_CHECK(
          continuation_in_it->input_node->GetType()->IsEqualTo(
              continuation_in_it->continuation_out->output_node->GetType()),
          loc);
      XLSCC_CHECK(continuation_in_it->input_node->GetType()->IsEqualTo(
                      prev_slice_val.GetType()),
                  loc);

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

    XLSCC_CHECK_EQ(args.size(), slice.function->params().size(), loc);

    TrackedBValue slice_ret =
        builder->Invoke(ToNativeBValues(args), slice.function, loc);
    XLSCC_CHECK(slice_ret.valid(), loc);

    int64_t output_idx = 0;
    for (const ContinuationValue& continuation_out : slice.continuations_out) {
      XLSCC_CHECK(slice_ret.GetType()->IsTuple(), loc);
      XLSCC_CHECK_LT(output_idx, slice_ret.GetType()->AsTupleOrDie()->size(),
                     loc);

      TrackedBValue out_value =
          builder->TupleIndex(slice_ret, output_idx++, loc,
                              /*name=*/continuation_out.name);

      XLSCC_CHECK(out_value.GetType()->IsEqualTo(
                      continuation_out.output_node->GetType()),
                  loc);

      prev_slice_ret[&continuation_out] = out_value;
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
