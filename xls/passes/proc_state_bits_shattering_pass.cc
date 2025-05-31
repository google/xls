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

#include "xls/passes/proc_state_bits_shattering_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

class TuplifyFlatStateElement : public Proc::StateElementTransformer {
 public:
  explicit TuplifyFlatStateElement(std::vector<int64_t> split_ends)
      : split_ends_(std::move(split_ends)) {}

  absl::StatusOr<Node*> TransformStateRead(Proc* proc,
                                           StateRead* new_state_read,
                                           StateRead* old_state_read) override {
    CHECK_GT(split_ends_.size(), 1);

    BitsType* old_type = old_state_read->GetType()->AsBitsOrDie();
    CHECK_EQ(split_ends_.back(), old_type->bit_count());

    TupleType* new_type = new_state_read->GetType()->AsTupleOrDie();
    CHECK_EQ(new_type->size(), split_ends_.size());

    std::vector<Node*> elements;
    elements.reserve(new_type->size());
    for (int64_t i = 0; i < new_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          proc->MakeNode<TupleIndex>(old_state_read->loc(), new_state_read, i));
      elements.push_back(element);
    }
    absl::c_reverse(elements);
    return proc->MakeNode<Concat>(old_state_read->loc(), elements);
  }

  absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                           StateRead* new_state_read,
                                           Next* old_next) override {
    CHECK_GT(split_ends_.size(), 1);

    Node* old_value = old_next->value();
    BitsType* old_type = old_value->GetType()->AsBitsOrDie();
    CHECK_EQ(split_ends_.back(), old_type->bit_count());

    TupleType* new_type = new_state_read->GetType()->AsTupleOrDie();
    CHECK_EQ(new_type->size(), split_ends_.size());

    std::vector<Node*> elements;
    elements.reserve(new_type->size());
    int64_t start = 0;
    for (int64_t i = 0; i < new_type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          proc->MakeNode<BitSlice>(old_next->loc(), old_value, /*start=*/start,
                                   /*width=*/split_ends_[i] - start));
      elements.push_back(element);
      start = split_ends_[i];
    }
    return proc->MakeNode<Tuple>(old_next->loc(), elements);
  }

 private:
  std::vector<int64_t> split_ends_;
};

absl::StatusOr<bool> MaybeSplitStateElements(
    Proc* proc, const OptimizationPassOptions& options) {
  bool changed = false;

  std::vector<int64_t> to_remove;
  std::vector<StateElement*> state_elements(proc->StateElements().begin(),
                                            proc->StateElements().end());
  for (StateElement* state_element : state_elements) {
    if (!state_element->type()->IsBits()) {
      continue;
    }

    // We'll keep `split_ends` in increasing order at all times, making it easy
    // to use STL set intersection algorithms.
    std::vector<int64_t> split_ends;
    bool could_benefit_from_splitting = false;
    StateRead* state_read = proc->GetStateRead(state_element);
    for (Next* next : proc->next_values(state_read)) {
      if (next->value() == state_read) {
        // This is a no-op next-value; it doesn't affect whether or not it's
        // beneficial to split the state element, since it'll convert to a no-op
        // next-value anyway.
        continue;
      }
      if (!next->value()->Is<Concat>()) {
        split_ends = {next->value()->BitCountOrDie()};
        break;
      }
      Concat* concat = next->value()->As<Concat>();

      // `concat_splits` is always sorted in increasing order by construction,
      // making it easy to use STL set intersection algorithms.
      std::vector<int64_t> concat_splits;
      concat_splits.reserve(concat->operand_count());
      if (absl::c_any_of(concat->operands(), [&](Node* operand) {
            if (operand->Is<PrioritySelect>()) {
              return true;
            }
            if (operand->Is<Select>()) {
              return options.split_next_value_selects.has_value() &&
                     operand->As<Select>()->cases().size() <=
                         *options.split_next_value_selects;
            }
            return false;
          })) {
        could_benefit_from_splitting = true;
      }
      int64_t start = 0;
      for (Node* operand : iter::reversed(concat->operands())) {
        start += operand->BitCountOrDie();
        concat_splits.push_back(start);
      }

      if (split_ends.empty()) {
        // First iteration, so the intersection is just `concat_splits`.
        split_ends = std::move(concat_splits);
      } else {
        // Since `split_ends` and `concat_splits` are sorted in increasing
        // order, we can use `absl::c_set_intersection` to set `split_ends`
        // equal to their intersection.
        std::vector<int64_t> intersection;
        intersection.reserve(std::min(split_ends.size(), concat_splits.size()));
        absl::c_set_intersection(split_ends, concat_splits,
                                 std::back_inserter(intersection));
        split_ends = std::move(intersection);
      }
      if (split_ends.size() <= 1) {
        break;
      }
    }
    if (!could_benefit_from_splitting || split_ends.size() <= 1) {
      // No splitting needed for this state element.
      continue;
    }

    std::vector<Value> initial_value_parts;
    initial_value_parts.reserve(split_ends.size());
    Bits old_initial_value = state_element->initial_value().bits();
    int64_t start = 0;
    for (int64_t split_end : split_ends) {
      initial_value_parts.push_back(
          Value(old_initial_value.Slice(start, split_end - start)));
      start = split_end;
    }
    Value initial_value = Value::Tuple(initial_value_parts);

    TuplifyFlatStateElement transformer(std::move(split_ends));
    XLS_RETURN_IF_ERROR(
        proc->TransformStateElement(state_read, initial_value, transformer)
            .status());
    changed = true;
    XLS_ASSIGN_OR_RETURN(int64_t state_index,
                         proc->GetStateElementIndex(state_element));
    to_remove.push_back(state_index);
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ProcStateBitsShatteringPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  XLS_ASSIGN_OR_RETURN(bool split_state_elements,
                       MaybeSplitStateElements(proc, options));
  return split_state_elements;
}

REGISTER_OPT_PASS(ProcStateBitsShatteringPass);

}  // namespace xls
