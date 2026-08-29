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

#include "xls/passes/concat_select_removal_pass.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {
struct ConcatSelect {
  // The actual concat we are thinking of replacing.
  Concat* real_node;
  // The pieces of the concat. Either a select or a node.
  std::vector<std::variant<Node*, GenericSelect>> pieces;
};

// Check if we can merge the two selects.
//
// NB Because the selectors are known to be equal we don't need to check for the
// number of cases since the possibilities are:
//
// 1) Regular select -> either they both have full cases or at least one has a
//    default.
//
// 2) Priority Select -> Same number of cases.
//
// 3) One-hot select -> Same number of cases.
bool CanMerge(const QueryEngine& qe, const GenericSelect& a,
              const GenericSelect& b) {
  if (!(a.kind() == b.kind() &&
        qe.NodesKnownUnsignedEquals(a.selector(), b.selector()))) {
    return false;
  }
  // If we're actually running as a scheduled-ir pass we have a few more checks.
  if (!a.AsNode()->function_base()->IsScheduled()) {
    return true;
  }
  FunctionBase* fb = a.AsNode()->function_base();
  if (!fb->IsStaged(a.AsNode()) || !fb->IsStaged(b.AsNode())) {
    // We're a scheduled function but the nodes are not staged. Just fall back
    // to doing nothing.
    return false;
  }
  auto a_stage = fb->GetStageIndex(a.AsNode());
  auto b_stage = fb->GetStageIndex(b.AsNode());
  CHECK_OK(a_stage.status());
  CHECK_OK(b_stage.status());
  // Only merge if we are in the same stage.
  return *a_stage == *b_stage;
}

// Applicability guard.
//
// Returns a list of all the concat nodes in the function that have at least 2
// adjacent selects with the same selector as operands.
absl::StatusOr<std::vector<ConcatSelect>> FindConcatSelects(
    FunctionBase* f, const QueryEngine& qe, OptimizationContext& context) {
  std::vector<ConcatSelect> concat_selects;
  XLS_ASSIGN_OR_RETURN(auto nodes, context.TopoSort(f));
  for (Node* node : nodes) {
    if (!node->Is<Concat>()) {
      continue;
    }
    Concat* concat = node->As<Concat>();
    std::optional<GenericSelect> prev_select;
    int64_t select_count = 0;
    ConcatSelect concat_select{.real_node = concat};
    int64_t max_select_count = 0;
    auto add_non_select = [&]() {
      select_count = 0;
      prev_select = std::nullopt;
    };
    auto add_select = [&](const GenericSelect& sel) {
      if (prev_select && CanMerge(qe, *prev_select, sel)) {
        // This one joins the existing list of mergeable selects
        select_count++;
      } else {
        // If there is a previous select but it can't be merged then we need to
        // start our count from 1 again.
        select_count = 1;
      }
      prev_select = sel;
      if (select_count > max_select_count) {
        max_select_count = select_count;
      }
    };
    for (Node* operand : concat->operands()) {
      if (std::optional<GenericSelect> select =
              GenericSelect::TryFrom(operand)) {
        concat_select.pieces.push_back(*select);
        add_select(*select);
      } else {
        add_non_select();
        concat_select.pieces.push_back(operand);
      }
    }
    if (max_select_count >= 2) {
      // Can't do any transform unless there are at least 2 selects.
      VLOG(3) << "Found concat select: " << concat->ToString();
      concat_selects.push_back(concat_select);
    }
  }
  return concat_selects;
}

struct TransformTarget {
  Node* real_node;
  std::vector<std::variant<Node*, std::vector<GenericSelect>>> pieces;
};

// Profitability guard.
//
// Get the nodes we want to create.
//
// Just null profitability, Anything we can do we will do.
absl::StatusOr<std::optional<TransformTarget>> PickTransform(
    const ConcatSelect& concat_select, const QueryEngine& qe) {
  using SelectList = std::vector<GenericSelect>;
  TransformTarget res{.real_node = concat_select.real_node};
  for (const auto& piece : concat_select.pieces) {
    std::visit(
        Visitor{
            [&](Node* node) { res.pieces.push_back(node); },
            [&](const GenericSelect& select) {
              if (!res.pieces.empty() &&
                  std::holds_alternative<SelectList>(res.pieces.back()) &&
                  CanMerge(qe, std::get<SelectList>(res.pieces.back()).back(),
                           std::get<GenericSelect>(piece))) {
                std::get<SelectList>(res.pieces.back())
                    .push_back(std::get<GenericSelect>(piece));
              } else {
                res.pieces.push_back(
                    std::vector<GenericSelect>{std::get<GenericSelect>(piece)});
              }
            },
        },
        piece);
  }
  for (auto& piece : res.pieces) {
    if (std::holds_alternative<SelectList>(piece) &&
        std::get<SelectList>(piece).size() == 1) {
      VLOG(2) << "Simplifying select list of size 1: "
              << std::get<SelectList>(piece).front().AsNode()->ToString();
      piece = std::get<SelectList>(piece).front().AsNode();
    }
  }
  return res;
}

absl::StatusOr<Node*> TransformToSelectOfConcats(
    FunctionBase* fb, const std::vector<GenericSelect>& selects,
    const SourceInfo& loc = SourceInfo()) {
  auto get_case = [&](const GenericSelect& select,
                      int64_t i) -> absl::StatusOr<Node*> {
    if (i < select.cases().size()) {
      return select.cases()[i];
    }
    XLS_RET_CHECK(select.default_value().has_value())
        << "Asked to get case " << i << " from select with only "
        << select.cases().size()
        << " cases and no default value: " << select.AsNode()->ToString();
    return *select.default_value();
  };
  auto largest_sel = absl::c_max_element(
      selects, [](const GenericSelect& a, const GenericSelect& b) {
        return a.cases().size() < b.cases().size();
      });
  int64_t num_cases = largest_sel->cases().size();
  bool has_default = largest_sel->default_value().has_value();
  std::optional<int64_t> stage;
  if (fb->IsScheduled()) {
    XLS_ASSIGN_OR_RETURN(stage, fb->GetStageIndex(selects.front().AsNode()));
  }
  std::vector<Node*> new_cases;
  new_cases.reserve(num_cases);
  auto collect_cases = [&](int64_t case_idx) -> absl::StatusOr<Node*> {
    std::vector<Node*> case_vals;
    case_vals.reserve(selects.size());
    for (const GenericSelect& select : selects) {
      XLS_ASSIGN_OR_RETURN(Node * case_node, get_case(select, case_idx));
      case_vals.push_back(case_node);
    }
    XLS_ASSIGN_OR_RETURN(Node * concat, fb->MakeNode<Concat>(loc, case_vals));
    if (stage.has_value()) {
      XLS_RETURN_IF_ERROR(fb->AddNodeToStage(*stage, concat).status());
    }
    return concat;
  };
  for (int64_t case_idx = 0; case_idx < num_cases; ++case_idx) {
    XLS_ASSIGN_OR_RETURN(Node * case_vals, collect_cases(case_idx));
    new_cases.push_back(case_vals);
  }
  std::optional<Node*> default_val;
  if (has_default) {
    XLS_ASSIGN_OR_RETURN(default_val, collect_cases(num_cases));
  }
  XLS_ASSIGN_OR_RETURN(Node * new_select, largest_sel->CloneSelectLike(
                                              largest_sel->selector(),
                                              new_cases, default_val, loc, ""));
  if (stage.has_value()) {
    XLS_RETURN_IF_ERROR(fb->AddNodeToStage(*stage, new_select).status());
  }
  return new_select;
}

absl::Status DoTransform(const TransformTarget& target) {
  std::vector<Node*> new_operands;
  new_operands.reserve(target.pieces.size());
  for (const auto& piece : target.pieces) {
    XLS_RETURN_IF_ERROR(std::visit(
        Visitor{
            [&](Node* node) -> absl::Status {
              new_operands.push_back(node);
              return absl::OkStatus();
            },
            [&](const std::vector<GenericSelect>& selects) -> absl::Status {
              XLS_ASSIGN_OR_RETURN(
                  Node * new_node,
                  TransformToSelectOfConcats(target.real_node->function_base(),
                                             selects, target.real_node->loc()));
              new_operands.push_back(new_node);
              return absl::OkStatus();
            },
        },
        piece));
  }
  if (new_operands.size() == 1) {
    XLS_RETURN_IF_ERROR(
        target.real_node->ReplaceUsesWith(new_operands.front()));
    return absl::OkStatus();
  }
  XLS_RETURN_IF_ERROR(
      target.real_node->ReplaceUsesWithNew<Concat>(new_operands).status());
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ConcatSelectRemovalPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  StatelessQueryEngine qe;
  XLS_RETURN_IF_ERROR(qe.Populate(f).status());

  // Applicability
  XLS_ASSIGN_OR_RETURN(std::vector<ConcatSelect> concat_selects,
                       FindConcatSelects(f, qe, context));

  VLOG(2) << "Found " << concat_selects.size() << " concat selects in "
          << f->name();

  // Profitability
  std::vector<TransformTarget> targets;
  targets.reserve(concat_selects.size());
  for (const ConcatSelect& concat_select : concat_selects) {
    VLOG(3) << "Considering concat select: "
            << concat_select.real_node->ToString();
    XLS_ASSIGN_OR_RETURN(std::optional<TransformTarget> target,
                         PickTransform(concat_select, qe),
                         _ << "Failed to pick transform for "
                           << concat_select.real_node->ToString());
    if (target) {
      VLOG(3) << "Will transform target: " << target->real_node->ToString()
              << " with " << target->pieces.size() << " pieces.";
      targets.push_back(*std::move(target));
    }
  }
  if (targets.empty()) {
    return false;
  }
  // Do the transform.
  for (const TransformTarget& target : targets) {
    XLS_RETURN_IF_ERROR(DoTransform(target))
        << "Failed to transform " << target.real_node->ToString();
  }
  return true;
}

}  // namespace xls
