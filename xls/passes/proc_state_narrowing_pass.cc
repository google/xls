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
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/proc_state_range_query_engine.h"

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

absl::Status RemoveLeadingBits(Param* param, const Value& orig_init_value,
                               const Bits& known_leading) {
  Value new_init_value(orig_init_value.bits().Slice(
      0, orig_init_value.bits().bit_count() - known_leading.bit_count()));
  ProcStateNarrowTransform transform(known_leading);
  return param->function_base()
      ->AsProcOrDie()
      ->TransformStateElement(param, new_init_value, transform)
      .status();
}

}  // namespace

// TODO(allight): Technically we'd probably want to run this whole pass to fixed
// point (incorporating the results into later runs) to get optimal results.
// It's not clear how much we'd gain there though. For now we will just run it
// once assuming that params are relatively independent of one
// another/additional information won't reveal more opportunities.
absl::StatusOr<bool> ProcStateNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  XLS_RET_CHECK(ProcStateRangeQueryEngine::CanAnalyzeProcStateEvolution(proc))
      << "Unable to analyze the proc: " << proc;
  ProcStateRangeQueryEngine qe;
  XLS_RETURN_IF_ERROR(qe.Populate(proc).status());

  bool made_changes = false;
  // NB Since we are adding params we need to copy the state param list.
  for (Param* param :
       std::vector(proc->StateParams().begin(), proc->StateParams().end())) {
    if (!param->GetType()->IsBits()) {
      VLOG(3) << "Unable to narrow compound value param " << param;
      continue;
    }
    std::optional<LeafTypeTree<TernaryVector>> ternary = qe.GetTernary(param);
    if (!ternary) {
      continue;
    }
    int64_t known_leading =
        ternary_ops::ToKnownBits(ternary->Get({})).CountLeadingOnes();
    if (known_leading == 0) {
      // TODO(allight): We could also narrow internal/trailing bits.
      VLOG(2) << "Unable to narrow " << param
              << " due to finding that no leading bits are known.";
      continue;
    }
    TernarySpan known_leading_tern =
        absl::MakeConstSpan(ternary->Get({})).last(known_leading);
    XLS_RET_CHECK(ternary_ops::IsFullyKnown(known_leading_tern));
    XLS_ASSIGN_OR_RETURN(Value orig_init_value, proc->GetInitValue(param));
    VLOG(2) << "Narrowing param " << param << " from " << param->BitCountOrDie()
            << " to " << (param->BitCountOrDie() - known_leading)
            << " bits (removing " << known_leading << " bits).";
    XLS_RETURN_IF_ERROR(
        RemoveLeadingBits(param, orig_init_value,
                          ternary_ops::ToKnownBitsValues(known_leading_tern)));
    made_changes = true;
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateNarrowingPass);

}  // namespace xls
