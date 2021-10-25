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
// See the License for the specific language governing permissions and
// limitations under the License.


#include "xls/examples/proc_fir_filter.h"
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value_helpers.h"

namespace xls {

// Builds a proc to implement an FIR filter.
absl::StatusOr<Proc*>
CreateFirFilter(absl::string_view name,
                               const Value& kernel_value,
                               StreamingChannel* input_channel,
                               StreamingChannel* output_channel,
                               Package* package) {
  // Build initial array of zeros for the input vector x.

  if (!kernel_value.IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Kernel must be an array is type: %s",
                        package->GetTypeForValue(kernel_value)->ToString()));
  }

  ArrayType* kernel_type =
      package->GetTypeForValue(kernel_value)->AsArrayOrDie();

  if (!kernel_type->element_type()->IsBits()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Kernel must be an array of bits, is type: %s",
                       kernel_type->ToString()));
  }

  Value shiftreg_init = ZeroOfType(kernel_type);

  // Create ProcBuilder and assign two channels.
  // The proc state is a Tuple. Element 0 is the result of the current filter
  // convolution. Element 1 is the current input array x.
  ProcBuilder pb(name,
                 Value::Tuple({Value(UBits(0, 32)), shiftreg_init}),
                 absl::StrFormat("%s_token", name),
                 absl::StrFormat("%s_state", name),
                 package);

  if (output_channel->type() != kernel_type->element_type()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Output channel must be type %s, is type %s",
                        kernel_type->element_type()->ToString(),
                        output_channel->type()->ToString()));
  }

  if (input_channel->type() != kernel_type->element_type()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Input channel must be type %s, is type %s",
                        kernel_type->element_type()->ToString(),
                        input_channel->type()->ToString()));
  }

  // The output channel gives us the output of the FIR filter.
  BValue out = pb.Send(output_channel, pb.GetTokenParam(),
                       pb.TupleIndex(pb.GetStateParam(), 0));

  // The input channel gives us the next value to include in x.
  BValue in = pb.Receive(input_channel, pb.GetTokenParam());

  BValue kernel = pb.Literal(kernel_value, absl::nullopt, "kernel");

  // The input array x is essentially a shift register, so we can use array
  // operations to remove the oldest element in x and append the new element.
  BValue shiftreg_slicer = pb.Literal(UBits(0, 32), absl::nullopt,
                                      "shiftreg_slicer");
  BValue index = pb.TupleIndex(pb.GetStateParam(), 1, absl::nullopt, "index");
  BValue x_slice = pb.ArraySlice(index, shiftreg_slicer, kernel_value.size()-1);
  BValue new_x = pb.Array({pb.TupleIndex(in, 1)}, kernel_type->element_type(),
                          absl::nullopt, "new_x");
  BValue x = pb.ArrayConcat({new_x, x_slice}, absl::nullopt, "x");

  BValue accumulator = pb.Literal(ZeroOfType(kernel_type->element_type()));

  // Perform the convolution of x and the kernel.
  for (int fir_idx = 0; fir_idx < kernel_value.size(); fir_idx++) {
    // Create an indexer node for x and kernel.
    BValue indexer = pb.Literal(UBits(fir_idx, 32), absl::nullopt,
                                absl::StrFormat("indexer_%d", fir_idx));
    BValue multiplier = pb.UMul(pb.ArrayIndex(kernel, {indexer}),
                                pb.ArrayIndex(x, {indexer}));
    accumulator = pb.Add(accumulator, multiplier);
  }

  BValue after_all = pb.AfterAll({out, pb.TupleIndex(in, 0)});
  BValue next_state = pb.Tuple({accumulator, x});
  XLS_ASSIGN_OR_RETURN(Proc* proc, pb.Build(after_all, next_state));
  return proc;
}

}  // namespace xls
