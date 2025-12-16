// Copyright 2023 The XLS Authors
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

#ifndef XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_
#define XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass that legalizes multiple send/receive operations per channel.
//
// This pass adds cross-activation tokens to guarantee that later activations of
// a proc cannot send or receive on a channel until all previous activations
// have completed working with that channel.
//
// The `ChannelLegalizationPass` is an optimization pass designed to ensure the
// correctness and determinism of hardware designs that contain multiple `send`
// or `receive` operations targeting the same channel within a single Proc
// activation. In hardware, simultaneous, unordered access to a shared channel
// can lead to non-deterministic behavior, race conditions, or resource
// conflicts. This pass addresses these issues by introducing explicit token
// dependencies to enforce a well-defined ordering of these operations.
//
// The pass operates by analyzing the communication patterns within Procs and
// enforcing strictness rules based on the channel's `strictness` attribute.
// This attribute, defined in the channel configuration, dictates how access to
// a shared channel should be managed:
//
// *   **`kTotalOrder`**: All operations on the channel must adhere to a strict
//     total order. The pass verifies if the existing token graph already
//     enforces this order; if not, it introduces mechanisms (cross-activation
//     tokens) to explicitly enforce it.
//
// *   **`kRuntimeOrdered`**: Similar to `kTotalOrder`, but allows for dynamic
//     ordering at runtime as long as correctness is maintained. The pass
//     inserts assertions to verify this at runtime if violations occur.
//
// *   **`kProvenMutuallyExclusive`**: Operations on the channel are expected to
//     be mutually exclusive at compile time (i.e., only one operation can be
//     active in any given activation). The pass uses a Z3 solver to formally
//     prove this mutual exclusivity; if it cannot be proven, a compile-time
//     error is raised.
//
// *   **`kRuntimeMutuallyExclusive`**: Operations are expected to be mutually
//     exclusive at runtime. The pass adds runtime assertions that will trigger
//     an error during simulation or execution if simultaneous access occurs.
//
// *   **`kArbitraryStaticOrder`**: Allows for any static ordering of
//     operations. The pass will introduce token dependencies to impose an
//     arbitrary but deterministic static order if multiple operations are
//     detected.
//
// The core mechanism for enforcing ordering when multiple operations on a
// channel exist is by introducing "cross-activation tokens." For each `send` or
// `receive` operation that is part of a multiple-access group on a given
// channel:
//
// 1.  **New State Elements for Tokens**: A new `token`-typed `StateElement` is
//     added to the Proc for each `send` or `receive` operation that requires
//     legalization. These new state elements act as implicit tokens, tracking
//     the completion of operations across different activations of the Proc.
//
// 2.  **`Next` Operations for New Tokens**: For each newly added `token`
//     state element, a `next_value` operation is introduced. This `next_value`
//     is
//     configured to update the token state with the token output of the
//     corresponding `send` or `receive` operation, conditioned by the
//     `send`/`receive` predicate (if it exists). This ensures that the token
//     for the *next* activation is only available after the current
//     activation's operation has completed.
//
// 3.  **Modifying Original Tokens**: The original `token` input to each
//     `send` or `receive` operation is replaced with an `after_all` operation.
//     This
//     `after_all` operation combines the original incoming token with all the
//     newly created implicit tokens from *other* operations on the *same
//     channel*. This effectively creates a dependency chain, ensuring that a
//     given `send`/`receive` cannot proceed until all other `send`/`receive`
//     operations on that channel from the *previous* activation have completed,
//     and its own incoming token is ready.
//
// This entire process ensures that even when multiple `send`/`receive`
// operations target the same channel, their execution is properly sequenced,
// preventing data hazards and ensuring deterministic behavior in the generated
// hardware. The pass also includes checks to ensure that channels of type
// `kStreaming` (the default) are legalized, while other channel kinds are
// skipped.
//
// Example (simplified conceptual view for multiple receives on channel `in`
// with `kTotalOrder` strictness, where no explicit token dependency exists
// initially):
//
//
// ```
// // Original IR snippet for a proc with multiple receives on 'in'
// chan in(bits[32], id=0, kind=streaming, ops=receive_only,
//         flow_control=ready_valid, strictness=total_order)
//
// top proc my_proc() {
//   tok: token = literal(value=token)
//   recv0: (token, bits[32]) = receive(tok, channel=in)
//   recv0_tok: token = tuple_index(recv0, index=0)
//   recv0_data: bits[32] = tuple_index(recv0, index=1)
//
//   recv1: (token, bits[32]) = receive(tok, channel=in)
//   // Also uses original 'tok', creating a conflict
//   recv1_tok: token = tuple_index(recv1, index=0)
//   recv1_data: bits[32] = tuple_index(recv1, index=1)
//
//   // ... other logic and next state
// }
// ```
//
//
// For this scenario, where `recv0` and `recv1` initially use the same `tok`
// without explicit ordering, `ChannelLegalizationPass` would perform
// transformations similar to:
//
//
// ```
// // Optimized IR snippet (simplified, showing key changes)
// chan in(bits[32], id=0, kind=streaming, ops=receive_only,
//         flow_control=ready_valid, strictness=total_order)
//
// top proc my_proc() {
//   original_tok_input: token = literal(value=token) // Original token input
//
//   // New state elements to track tokens across activations
//   implicit_token__recv0_state: token = state_element(init=token)
//   implicit_token__recv1_state: token = state_element(init=token)
//
//   // recv0 now waits on implicit_token__recv1_state from previous activation
//   recv0: (token, bits[32]) =
//          receive(
//            after_all(
//              original_tok_input,
//              state_read(implicit_token__recv1_state)),
//            channel=in)
//   recv0_tok: token = tuple_index(recv0, index=0)
//   recv0_data: bits[32] = tuple_index(recv0, index=1)
//   // Update implicit_token__recv0_state
//   next (implicit_token__recv0_state, recv0_tok)
//
//   // recv1 now waits on implicit_token__recv0_state from previous activation
//   recv1: (token, bits[32]) =
//          receive(
//            after_all(
//              original_tok_input,
//              state_read(implicit_token__recv0_state)),
//            channel=in)
//   recv1_tok: token = tuple_index(recv1, index=0)
//   recv1_data: bits[32] = tuple_index(recv1, index=1)
//   // Update implicit_token__recv1_state
//   next (implicit_token__recv1_state, recv1_tok)
//
//   // ... other logic and next state
// }
// ```
//
// This introduces a circular dependency through the state, ensuring that
// `recv0` in activation `N-1` must complete before `recv1` in activation `N`
// can proceed, and vice-versa, thereby enforcing a total order of access across
// activations and preventing simultaneous access.
class ChannelLegalizationPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "channel_legalization";
  ChannelLegalizationPass()
      : OptimizationPass(kName, "Legalize multiple send/recvs per channel") {}
  ~ChannelLegalizationPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CHANNEL_LEGALIZATION_PASS_H_
