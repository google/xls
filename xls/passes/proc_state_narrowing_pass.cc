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
#include "xls/ir/interval_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/proc_state_range_query_engine.h"

namespace xls {
namespace {

// Struct which transforms a state element into a slice of its trailing bits.
struct ProcStateNarrowTransform : public Proc::StateElementTransformer {
 public:
  explicit ProcStateNarrowTransform(int64_t known_leading)
      : Proc::StateElementTransformer(), known_leading_(known_leading) {}

  int64_t known_leading() const { return known_leading_; }
  absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                           StateRead* new_state_read,
                                           Next* old_next) final {
    XLS_RET_CHECK_EQ(
        new_state_read->GetType()->GetFlatBitCount() + known_leading_,
        old_next->state_read()->GetType()->GetFlatBitCount());
    return proc->MakeNodeWithName<BitSlice>(
        old_next->loc(), old_next->value(), /*start=*/0,
        /*width=*/new_state_read->GetType()->GetFlatBitCount(),
        absl::StrFormat("unexpand_for_%s", old_next->GetName()));
  }

 private:
  int64_t known_leading_;
};

class ProcStateConcatNarrowTransform : public ProcStateNarrowTransform {
 public:
  explicit ProcStateConcatNarrowTransform(Bits leading_bits)
      : ProcStateNarrowTransform(leading_bits.bit_count()),
        leading_bits_(std::move(leading_bits)) {}

  absl::StatusOr<Node*> TransformStateRead(Proc* proc,
                                           StateRead* new_state_read,
                                           StateRead* old_state_read) final {
    XLS_RET_CHECK_EQ(
        new_state_read->GetType()->GetFlatBitCount() + known_leading(),
        old_state_read->GetType()->GetFlatBitCount());
    XLS_ASSIGN_OR_RETURN(
        Node * leading,
        proc->MakeNodeWithName<Literal>(
            old_state_read->loc(), Value(leading_bits_),
            absl::StrFormat("leading_bits_%s",
                            old_state_read->state_element()->name())));
    return proc->MakeNodeWithName<Concat>(
        new_state_read->loc(), std::array<Node*, 2>{leading, new_state_read},
        absl::StrFormat("extended_%s",
                        old_state_read->state_element()->name()));
  }

 private:
  Bits leading_bits_;
};

absl::Status RemoveLeadingBits(StateRead* state_read,
                               const Value& orig_init_value,
                               const Bits& known_leading) {
  Value new_init_value(orig_init_value.bits().Slice(
      0, orig_init_value.bits().bit_count() - known_leading.bit_count()));
  ProcStateConcatNarrowTransform transform(known_leading);
  return state_read->function_base()
      ->AsProcOrDie()
      ->TransformStateElement(state_read, new_init_value, transform)
      .status();
}

class ProcStateSignExtendNarrowTransform : public ProcStateNarrowTransform {
 public:
  explicit ProcStateSignExtendNarrowTransform(int64_t known_leading)
      : ProcStateNarrowTransform(known_leading) {}

  absl::StatusOr<Node*> TransformStateRead(Proc* proc,
                                           StateRead* new_state_read,
                                           StateRead* old_state_read) final {
    return proc->MakeNodeWithName<ExtendOp>(
        new_state_read->loc(), new_state_read, old_state_read->BitCountOrDie(),
        Op::kSignExt,
        absl::StrFormat("extended_%s",
                        old_state_read->state_element()->name()));
  }
};

absl::Status RemoveSignBits(StateRead* state_read, const Value& orig_init_value,
                            int64_t real_size) {
  Value new_init_value(orig_init_value.bits().Slice(0, real_size));
  ProcStateSignExtendNarrowTransform transform(state_read->BitCountOrDie() -
                                               real_size);
  return state_read->function_base()
      ->AsProcOrDie()
      ->TransformStateElement(state_read, new_init_value, transform)
      .status();
}

}  // namespace

// TODO(allight): Technically we'd probably want to run this whole pass to fixed
// point (incorporating the results into later runs) to get optimal results.
// It's not clear how much we'd gain there though. For now we will just run it
// once assuming that state elements are relatively independent of one
// another/additional information won't reveal more opportunities.
absl::StatusOr<bool> ProcStateNarrowingPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext* context) const {
  XLS_RET_CHECK(ProcStateRangeQueryEngine::CanAnalyzeProcStateEvolution(proc))
      << "Unable to analyze the proc: " << proc;
  ProcStateRangeQueryEngine qe;
  XLS_RETURN_IF_ERROR(qe.Populate(proc).status());

  bool made_changes = false;
  // NB Since we are adding state elements we need to copy the list.
  for (StateElement* state_element : std::vector(proc->StateElements().begin(),
                                                 proc->StateElements().end())) {
    if (!state_element->type()->IsBits()) {
      VLOG(3) << "Unable to narrow compound value state element "
              << state_element->name() << ": "
              << state_element->type()->ToString();
      continue;
    }
    StateRead* state_read = proc->GetStateRead(state_element);
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        qe.GetTernary(state_read);
    if (!ternary) {
      continue;
    }
    int64_t known_leading =
        ternary_ops::ToKnownBits(ternary->Get({})).CountLeadingOnes();
    if (known_leading != 0) {
      // TODO(allight): We could also narrow internal/trailing bits.
      TernarySpan known_leading_tern =
          absl::MakeConstSpan(ternary->Get({})).last(known_leading);
      XLS_RET_CHECK(ternary_ops::IsFullyKnown(known_leading_tern));
      Value orig_init_value = state_element->initial_value();
      VLOG(2) << "Narrowing state_read " << state_read << " from "
              << state_read->BitCountOrDie() << " to "
              << (state_read->BitCountOrDie() - known_leading)
              << " bits (removing " << known_leading
              << " bits) using known-leading bits.";
      XLS_RETURN_IF_ERROR(RemoveLeadingBits(
          state_read, orig_init_value,
          ternary_ops::ToKnownBitsValues(known_leading_tern)));
      made_changes = true;
      continue;
    }
    int64_t signed_bits = interval_ops::MinimumSignedBitCount(
        qe.GetIntervals(state_read).Get({}));
    int64_t signed_bits_removed = state_read->BitCountOrDie() - signed_bits;
    if (signed_bits_removed != 0) {
      Value orig_init_value = state_element->initial_value();
      VLOG(2) << "Narrowing state_read " << state_read << " from "
              << state_read->BitCountOrDie() << " to "
              << (state_read->BitCountOrDie() - known_leading)
              << " bits (removing " << known_leading
              << " bits) using sign-extend.";
      XLS_RETURN_IF_ERROR(RemoveSignBits(state_read, orig_init_value,
                                         /*real_size=*/signed_bits));
      made_changes = true;
    }
    VLOG(2) << "Unable to narrow " << state_read
            << " due to finding that no leading bits are known and signed "
               "interval is not narrowable.";
  }

  return made_changes;
}

REGISTER_OPT_PASS(ProcStateNarrowingPass);

}  // namespace xls
