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

#include "xls/passes/proc_state_narrowing_pass.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {
// Struct which transforms a param into a slice of its trailing bits.
struct ProcStateNarrowTransform : public Proc::StateElementTransformer {
 public:
  explicit ProcStateNarrowTransform(Bits known_leading)
      : Proc::StateElementTransformer(),
        known_leading_(std::move(known_leading)) {}

  absl::StatusOr<Node*> TransformParamRead(Proc* proc, Param* new_param,
                                           Param* old_param) final {
    XLS_RET_CHECK_EQ(
        new_param->GetType()->GetFlatBitCount() + known_leading_.bit_count(),
        old_param->GetType()->GetFlatBitCount());
    XLS_ASSIGN_OR_RETURN(
        Node * leading,
        proc->MakeNodeWithName<Literal>(
            old_param->loc(), Value(known_leading_),
            absl::StrFormat("leading_bits_%s", old_param->name())));
    return proc->MakeNodeWithName<Concat>(
        new_param->loc(), std::array<Node*, 2>{leading, new_param},
        absl::StrFormat("extended_%s", old_param->name()));
  }
  absl::StatusOr<Node*> TransformNextValue(Proc* proc, Param* new_param,
                                           Next* old_next) final {
    XLS_RET_CHECK_EQ(
        new_param->GetType()->GetFlatBitCount() + known_leading_.bit_count(),
        old_next->param()->GetType()->GetFlatBitCount());
    return proc->MakeNodeWithName<BitSlice>(
        old_next->loc(), old_next->value(), /*start=*/0,
        /*width=*/new_param->GetType()->GetFlatBitCount(),
        absl::StrFormat("unexpand_for_%s", old_next->GetName()));
  }

 private:
  Bits known_leading_;
};
}  // namespace

absl::StatusOr<bool> ProcStateNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Find basic ternary limits
  TernaryQueryEngine tqe;
  XLS_RETURN_IF_ERROR(tqe.Populate(proc).status());

  // List of all the next instructions that change the param for each param.
  absl::flat_hash_map<Param*, std::vector<Next*>> modifying_nexts_for_param;
  for (Param* param : proc->StateParams()) {
    // TODO(allight): Being able to narrow inside a compound value would be
    // nice. Since we unpack tuple state elements in other passes however the
    // actual impact would likely be negligible so no reason to bother with it
    // for now.
    if (!param->GetType()->IsBits()) {
      continue;
    }
    std::vector<Next*>& nexts = modifying_nexts_for_param[param];
    for (Next* n : proc->next_values(param)) {
      // TODO(allight): We might want to use data-flow to better track whether
      // things have changed. This should probably be good enough in practice
      // however.
      if (n->param() != n->value()) {
        nexts.push_back(n);
      }
    }
  }
  bool made_changes = false;
  // To avoid issues where changes to the param values leads to invalidating the
  // TernaryQueryEngine we do all the modifications at the end.
  struct ToTransform {
    Param* orig_param;
    Value new_init_value;
    ProcStateNarrowTransform transformer;
  };
  std::vector<ToTransform> transforms;
  for (const auto& [orig_param, updates] : modifying_nexts_for_param) {
    if (updates.empty()) {
      // The state only has identity updates? Strange but this will be cleaned
      // up by NextValueOptimizationPass so we can ignore it.
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Value orig_init_value, proc->GetInitValue(orig_param));
    TernaryVector possible_values =
        ternary_ops::BitsToTernary(orig_init_value.bits());
    for (Next* next : updates) {
      possible_values = ternary_ops::Intersection(
          possible_values, tqe.GetTernary(next->value()).Get({}));
    }
    int64_t initial_width = possible_values.size();
    int64_t known_leading =
        ternary_ops::ToKnownBits(possible_values).CountLeadingOnes();
    if (known_leading == 0) {
      continue;
    }

    transforms.push_back(
        {.orig_param = orig_param,
         // Remove the known leading bits from the proc state.
         .new_init_value = Value(
             orig_init_value.bits().Slice(0, initial_width - known_leading)),
         .transformer = ProcStateNarrowTransform(ternary_ops::ToKnownBitsValues(
             absl::MakeSpan(possible_values)
                 .subspan(initial_width - known_leading)))});
  }
  for (auto t : std::move(transforms)) {
    XLS_RETURN_IF_ERROR(proc->TransformStateElement(
                                t.orig_param, t.new_init_value, t.transformer)
                            .status());
    made_changes = true;
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateNarrowingPass);

}  // namespace xls
