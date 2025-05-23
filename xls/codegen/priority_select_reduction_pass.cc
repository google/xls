// Copyright 2024 The XLS Authors
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

#include "xls/codegen/priority_select_reduction_pass.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

absl::StatusOr<bool> PrioritySelectReductionPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!context.HasMetadataForBlock(block.get())) {
      continue;
    }
    CodegenMetadata& metadata = context.GetMetadataForBlock(block.get());

    BddQueryEngine query_engine(BddQueryEngine::kDefaultPathLimit);
    XLS_RETURN_IF_ERROR(query_engine.Populate(block.get()).status());
    for (Node* node : ReverseTopoSort(block.get())) {
      if (!node->Is<PrioritySelect>()) {
        continue;
      }
      PrioritySelect* sel = node->As<PrioritySelect>();
      Node* selector = sel->selector();
      XLS_RET_CHECK(selector->GetType()->IsBits());
      if (selector->BitCountOrDie() == 1) {
        // This can be converted to a select, which converts to neater Verilog.
        VLOG(2) << absl::StreamFormat("Rewriting to one-bit select: %v", *node);
        XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<Select>(
                                   selector,
                                   std::vector<Node*>{sel->default_value(),
                                                      sel->cases().front()},
                                   /*default_value=*/std::nullopt)
                                .status());
        changed = true;
        continue;
      }

      if (!query_engine.AtMostOneBitTrue(selector)) {
        // Selector may not be one-hot, so we need to be able to reject the
        // lower-priority cases.
        continue;
      }
      if (!query_engine.AtLeastOneBitTrue(selector) &&
          !query_engine.IsAllZeros(sel->default_value())) {
        // The default case may be used and may be nonzero, so we can't rewrite
        // this to a one-hot select without adding a new case.
        continue;
      }
      VLOG(2) << absl::StreamFormat("Rewriting to one-hot select: %v", *node);
      XLS_ASSIGN_OR_RETURN(Node * one_hot_selector,
                           sel->function_base()->MakeNode<OneHot>(
                               sel->loc(), selector, LsbOrMsb::kLsb));
      XLS_ASSIGN_OR_RETURN(
          Node * one_hot_original_bits,
          sel->function_base()->MakeNode<BitSlice>(
              sel->loc(), one_hot_selector, /*start=*/0,
              /*width=*/one_hot_selector->BitCountOrDie() - 1));
      XLS_ASSIGN_OR_RETURN(
          Node * selector_is_one_hot,
          sel->function_base()->MakeNode<CompareOp>(
              sel->loc(), selector, one_hot_original_bits, Op::kEq));
      XLS_ASSIGN_OR_RETURN(Node * tkn,
                           sel->function_base()->MakeNode<xls::Literal>(
                               sel->loc(), Value::Token()));
      XLS_ASSIGN_OR_RETURN(
          Node * one_hot_assert,
          sel->function_base()->MakeNode<Assert>(
              sel->loc(), tkn, selector_is_one_hot,
              absl::StrFormat(
                  "Selector %s was expected to be one-hot, and is not.",
                  selector->GetName()),
              /*label=*/
              SanitizeVerilogIdentifier(
                  absl::StrCat(sel->GetName(), "_selector_one_hot_A")),
              /*original_label=*/std::nullopt));
      XLS_ASSIGN_OR_RETURN(
          Node * new_sel,
          sel->ReplaceUsesWithNew<OneHotSelect>(selector, sel->cases()));

      absl::flat_hash_map<Node*, Stage>& node_to_stage_map =
          metadata.streaming_io_and_pipeline.node_to_stage_map;
      if (auto it = node_to_stage_map.find(sel);
          it != node_to_stage_map.end()) {
        const Stage stage = it->second;
        node_to_stage_map[one_hot_selector] = stage;
        node_to_stage_map[one_hot_original_bits] = stage;
        node_to_stage_map[selector_is_one_hot] = stage;
        node_to_stage_map[tkn] = stage;
        node_to_stage_map[one_hot_assert] = stage;
        node_to_stage_map[new_sel] = stage;
      }
      changed = true;
    }
  }
  return changed;
}
}  // namespace xls::verilog
