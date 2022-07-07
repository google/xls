// Copyright 2021 The XLS Authors
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

#ifndef XLS_PASSES_PROC_LOOP_FOLDING_H_
#define XLS_PASSES_PROC_LOOP_FOLDING_H_

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// This pass will act upon a Proc that contains a single CountedFor node. The
// goal is to move the state for the CountedFor node into the Proc state, and
// adjusting the data flow to allow the CountedFor loop (and hence the proc) to
// iterate the required number of times before placing any result on the output
// channel. The driving example is the proc_fir_filter found in xls/examples.

// In the FIR filter's case, the CountedFor is responsible for the convolution
// (multiply-accumulate). Left alone, the CountedFor would be unrolled by
// trip_count times. This translates into that many duplicate blocks of
// hardware. As multipliers are particularly large and expensive IPs, it is
// often good to minimize how many there are. By rolling the CountedFor into
// the proc state, we end up with only a single multiplier, which can be a
// nice area and power consumption optimization.
class RollIntoProcPass : public ProcPass {
 public:
  RollIntoProcPass(std::optional<int64_t> unroll_factor = absl::nullopt);
  ~RollIntoProcPass() override {}

 protected:
  absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const PassOptions& options,
      PassResults* results) const override;

  // Unroll the CountedFor function body. This will make it easier to do this
  // kind of optimization in this pass.
  absl::StatusOr<CountedFor*> UnrollCountedForBody(Proc* proc,
                                                   CountedFor* countedfor,
                                                   int64_t unroll_factor) const;

  // Create the new proc initial state that will be used for the folded loop.
  absl::StatusOr<Value> CreateInitialState(Proc* proc, CountedFor* countedfor,
                                           Receive* recv) const;

  // We will need to clone the CountedFor loop body into the proc. This function
  // takes care of that for us.
  absl::StatusOr<Node*> CloneCountedFor(Proc* proc, CountedFor* countedfor,
                                        Node* loop_induction_variable,
                                        Node* loop_carry) const;

  // The Receive node needs to be replaced with a "ReceiveIf" node. Since this
  // required a few steps (including muxing with one of the proc state Tuple
  // elements), move this to a separate function.
  absl::StatusOr<Node*> ReplaceReceiveWithConditionalReceive(
      Proc* proc, Receive* original_receive, Node* receive_condition,
      Node* on_condition_false) const;

  // Build a set of muxes that all use the same selector input.
  absl::StatusOr<std::vector<Node*>> SelectBetween(
      Proc* proc, CountedFor* countedfor, Node* selector,
      absl::Span<Node* const> on_false, absl::Span<Node* const> on_true) const;

 private:
  std::optional<int64_t> unroll_factor_;
};

}  // namespace xls

#endif  // XLS_PASSES_PROC_LOOP_FOLDING_H_
