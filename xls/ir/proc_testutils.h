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

#ifndef XLS_IR_PROC_TESTUTILS_H_
#define XLS_IR_PROC_TESTUTILS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {

// A struct which represents a single IO action, either a send or a receive.
struct IOAction {
  // Whether this action is a send or a receive.
  bool is_send;
  // The data that was sent or received.
  BValue data;
  // A boolean value which is whether the action is attempted (eg for a
  // send_if).
  BValue executed;
  // A boolean value which is whether the action completes, whether or not a
  // value is actually returned/sent. If false then the action is blocked due to
  // fifo full/empty.
  BValue completes;

  // Additional optional BValue that the input/output gen can fill in for use
  // wherever it is required.
  std::optional<BValue> context;
};

struct ActivationAction {
  // What activation we are on. The first activation is index 0.
  int64_t activation_index;
  // State vector after this activation. Values are only valid if
  // 'activation_complete' is true.
  absl::flat_hash_map<StateElement*, BValue> end_state;
  // Values which are sent out from this activation.
  absl::flat_hash_map<SendChannelRef, IOAction> sent_values;
  // Map from a receive node to the IO action describing the receive.
  absl::flat_hash_map<ReceiveChannelRef, IOAction> receives_finished;
  // Boolean value which is true if the activation completed and false if it
  // blocked on a send/recv.
  //
  // This is the and of all the IOAction.completes values.
  BValue activation_complete;

  // Mapping from original proc Node* to its unrolled BValue in this activation.
  absl::flat_hash_map<Node*, BValue> node_values;
};

// A struct which represents the unrolled proc.
struct UnrolledProc {
  // The unrolled function created in the proc's package.
  Function* function = nullptr;
  // State at the end of each activation.
  std::vector<ActivationAction> activations;
  // Initial state of the proc.
  absl::flat_hash_map<StateElement*, BValue> initial_state;
};

// Helper to convert a proc into a function which performs 'activation_count'
// activations. Input and output channels never block (though if non-blocking
// may skip the send/recv).
//
// Tokens are replaced with literals of the given 'token_value'. NB this should
// be a non-zero-length value because z3 doesn't like zero-length values with
// uses.
//
// This function adds the function version to the input procs package.
//
// The state elements are considered to start at their initial values.
//
// Each channel may only be sent on once per activation (though more than one
// send may be present).
// TODO(allight): Support sending on a single channel multiple times.
//
// Each channel will receive only one value per activation (though more than one
// receive may be present).
// TODO(allight): Support receiving on a single channel multiple times.
//
// StateReads which are predicated-off have a value of 0. The mutex and
// unobservability of these reads is not explicitly checked.
//
// The return type of the function depends on the lexicographic ordering of
// channels so this should not be messed with.
//
// This is only intended for use with testing tools such as z3.
// Helper to unroll a proc 'activation_count' times and return the complete
// UnrolledProc data structure, including the unrolled Function* as well as the
// intermediate node_values for every activation.
//
// If 'cleanup' is true, runs DCE and inlining on the unrolled function. When
// checking intermediate non-IO node constancy, set 'cleanup' to false so
// intermediate nodes are not eliminated by DCE.
absl::StatusOr<Function*> UnrollProcToFunction(
    Proc* p, int64_t activation_count, bool include_state,
    const Value& token_value = Value::Tuple({Value(UBits(0xdeadbeef, 32))}));

// Version of `UnrollProcToFunction` that also returns the intermediate
// node_values for every activation.
absl::StatusOr<UnrolledProc> UnrollProc(
    Proc* p, int64_t activation_count, bool include_state,
    const Value& token_value = Value::Tuple({Value(UBits(0xdeadbeef, 32))}),
    bool cleanup = true);

// Helper to convert a proc into a function which performs 'activation_count'
// activations consuming up to 'output_value_count' values and producing up to
// 'output_value_count' values.
//
// Input values are not necessarily consumed and output values beyond the
// 'output_value_count' are ignored.
//
// Channel input and output order is alphabetical by the channel name.
//
// In the unrolled proc execution can only occur at the granularity of an entire
// activation. If any send/receive would block due to full/empty channel FIFOs
// no observable progress is made.
//
// Non-blocking reads/writes are only skipped if the buffer is empty/full
// respectively.
//
// Each channel may only be sent on once per activation (though more than one
// send may be present).
// TODO(allight): Support sending on a single channel multiple times.
//
// Each channel will receive only one value per activation (though more than one
// receive may be present).
// TODO(allight): Support receiving on a single channel multiple times.
//
// StateReads which are predicated-off have a value of 0. The mutex and
// unobservability of these reads is not explicitly checked.
//
// This function adds the function version to the input procs package.
//
// The state elements are considered to start at their initial values.
//
// The return type of the function depends on the alphabetical ordering of
// channels so this should not be messed with.
//
// If count_recvs is true, the function will return the number of times each
// receive channel was received on in addition to the values sent on output
// channels.
//
// This is only intended for use with testing tools such as z3.
absl::StatusOr<Function*> UnrollProcToUntimedFunction(
    Proc* p, int64_t activation_count, int64_t input_value_count,
    int64_t output_value_count, bool count_recvs = true);

// Struct which holds what happens on a channel for each activation.
struct ChannelActions {
  struct SingleAction {
    // The value consumed/produced by the channel. For output channels the
    // values are the values sent. For input channels the values are the values
    // received.
    Value value;
    // What activation this value was sent/received on.
    int64_t activation_number;
  };

  // The actions taken by the channel. This must always be in increasing
  // activation-number order and each activation number must be used at most
  // once.
  std::vector<SingleAction> actions;
};

template <typename Sink>
void AbslStringify(Sink& sink, const ChannelActions& actions) {
  absl::Format(&sink, "[%v]", absl::StrJoin(actions.actions, ",\t"));
}
template <typename Sink>
void AbslStringify(Sink& sink, const ChannelActions::SingleAction& actions) {
  absl::Format(&sink, "%s@%d", actions.value.ToHumanString(),
               actions.activation_number);
}

inline bool operator==(const ChannelActions::SingleAction& lhs,
                       const ChannelActions::SingleAction& rhs) {
  return lhs.value == rhs.value &&
         lhs.activation_number == rhs.activation_number;
}
inline bool operator!=(const ChannelActions::SingleAction& lhs,
                       const ChannelActions::SingleAction& rhs) {
  return !(lhs == rhs);
}
inline bool operator==(const ChannelActions& lhs, const ChannelActions& rhs) {
  return lhs.actions == rhs.actions;
}
inline bool operator!=(const ChannelActions& lhs, const ChannelActions& rhs) {
  return !(lhs == rhs);
}

// Struct which holds the results of running a proc for a small number of
// activations.
struct ProcResults {
  // What each channel received/sent on each activation.
  absl::flat_hash_map<ChannelRef, ChannelActions> channel_actions;
  // The value of each node on each activation.
  absl::flat_hash_map<Node*, std::vector<Value>> node_values;
};

// Helper to get info about what an (isolated) proc does when run for a certain
// number of activations. Given the number of activations, lists of inputs
// available and the size of the output-channels it will run the proc until
// either the activations are executed or any channel blocks and returns
// information about what every node is on each activation and what values on
// what activation each channel produces/consumes.
//
// This should only be used on isolated procs (ie procs which do not communicate
// with other procs).
absl::StatusOr<ProcResults> GetProcResults(
    Proc* p, int64_t activation_count, int64_t output_value_count,
    const absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>>& inputs);

// IR Annotator that adds ProcResults to the IR.
//
// It shows results like:
//   node | [val1, val2, ...]
//   chan | [val1@<act>, val2@<act>, ...]
class ProcResultsAnnotator : public IrAnnotator {
 public:
  explicit ProcResultsAnnotator(ProcResults results)
      : results_(std::move(results)) {}
  Annotation NodeAnnotation(Node* node) const override;
  Annotation ChannelAnnotation(Channel* chan) const override;
  Annotation ChannelInterfaceAnnotation(
      const ChannelInterface* iface) const override;

 private:
  ProcResults results_;
};

// Helper to create a proc-results annotator (which ignores failures)
class ProcResultsAnnotatorOrNothing final : public IrAnnotator {
 public:
  explicit ProcResultsAnnotatorOrNothing(
      Proc* p, int64_t activation_count, int64_t output_value_count,
      const absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>>&
          inputs) {
    auto proc_results =
        GetProcResults(p, activation_count, output_value_count, inputs);
    if (proc_results.ok()) {
      annotator_.emplace(std::move(proc_results.value()));
    } else {
      LOG(WARNING) << "Failed to get proc results: " << proc_results.status();
    }
  }
  Annotation NodeAnnotation(Node* node) const override {
    if (annotator_) {
      return annotator_->NodeAnnotation(node);
    }
    return {};
  }
  Annotation ChannelAnnotation(Channel* chan) const override {
    if (annotator_) {
      return annotator_->ChannelAnnotation(chan);
    }
    return {};
  }
  Annotation ChannelInterfaceAnnotation(
      const ChannelInterface* chan) const override {
    if (annotator_) {
      return annotator_->ChannelInterfaceAnnotation(chan);
    }
    return {};
  }

 private:
  std::optional<ProcResultsAnnotator> annotator_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_TESTUTILS_H_
