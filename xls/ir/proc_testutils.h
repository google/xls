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

#include "absl/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls {

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
absl::StatusOr<Function*> UnrollProcToFunction(
    Proc* p, int64_t activation_count, bool include_state,
    const Value& token_value = Value::Tuple({Value(UBits(0xdeadbeef, 32))}));

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

}  // namespace xls

#endif  // XLS_IR_PROC_TESTUTILS_H_
