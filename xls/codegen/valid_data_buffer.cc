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

#include "xls/codegen/valid_data_buffer.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls::verilog {

absl::StatusOr<Block*> CreateValidDataBuffer(
    Package* package, std::string_view name, Type* data_type, int64_t depth,
    const ValidDataBufferOptions& options) {
  if (depth < 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Valid-data buffer `%s` must have a depth of at least 1; got %d.", name,
        depth));
  }

  BlockBuilder bb(name, package);
  bb.AddClockPort(options.clock_name);
  std::optional<BValue> reset;
  if (options.reset.has_value()) {
    reset = bb.ResetPort(options.reset->name, options.reset->behavior);
  }

  Type* u1 = package->GetBitsType(1);
  BValue push_data = bb.InputPort(kValidDataBufferPushDataPort, data_type);
  BValue push = bb.InputPort(kValidDataBufferPushValidPort, u1);
  BValue pop_ready = bb.InputPort(kValidDataBufferPopReadyPort, u1);

  // Entry 0 holds the oldest stored value (if any), entry 1 the next-oldest,
  // and so on; an entry can only be occupied if all entries before it are, too.
  auto add_register = [&](std::string_view reg_name,
                          Type* type) -> absl::StatusOr<Register*> {
    return bb.block()->AddRegister(reg_name, type,
                                   reset.has_value()
                                       ? std::make_optional(ZeroOfType(type))
                                       : std::nullopt);
  };
  std::vector<Register*> valid_regs;
  std::vector<Register*> data_regs;
  std::vector<BValue> valid;
  std::vector<BValue> data;
  for (int64_t i = 0; i < depth; ++i) {
    XLS_ASSIGN_OR_RETURN(
        Register * valid_reg,
        add_register(absl::StrFormat("entry_%d_valid", i), u1));
    XLS_ASSIGN_OR_RETURN(
        Register * data_reg,
        add_register(absl::StrFormat("entry_%d_data", i), data_type));
    valid_regs.push_back(valid_reg);
    data_regs.push_back(data_reg);
    valid.push_back(
        bb.RegisterRead(valid_reg, SourceInfo(), valid_reg->name()));
    data.push_back(bb.RegisterRead(data_reg, SourceInfo(), data_reg->name()));
  }

  // The consumer sees the head entry or, if the buffer is empty, the value
  // arriving in this cycle.
  bb.OutputPort(kValidDataBufferPopValidPort, bb.Or(valid[0], push));
  bb.OutputPort(kValidDataBufferPopDataPort,
                bb.Select(valid[0], /*on_true=*/data[0],
                          /*on_false=*/push_data));

  // If the buffer isn't empty, a pop removes the head entry's value, and every
  // stored value shifts one entry toward the head. Otherwise, a popped value
  // falls through from the push side, and is never stored.
  BValue shift = bb.And(valid[0], pop_ready, SourceInfo(), "shift");
  BValue store =
      bb.And(push, bb.Or(valid[0], bb.Not(pop_ready)), SourceInfo(), "store");

  for (int64_t i = 0; i < depth; ++i) {
    const bool is_last = i == depth - 1;
    // When values shift, entry i takes entry i + 1's value, or (if entry i was
    // the last occupied entry) the arriving value.
    BValue occupied_after_shift = bb.And(push, valid[i]);
    if (!is_last) {
      occupied_after_shift = bb.Or(valid[i + 1], occupied_after_shift);
    }
    // Otherwise, a stored value goes into the first unoccupied entry.
    BValue first_unoccupied = bb.Not(valid[i]);
    if (i > 0) {
      first_unoccupied = bb.And(valid[i - 1], first_unoccupied);
    }

    bb.RegisterWrite(
        valid_regs[i],
        bb.Select(shift, /*on_true=*/occupied_after_shift,
                  /*on_false=*/bb.Or(valid[i], bb.And(store, first_unoccupied)),
                  SourceInfo(), absl::StrFormat("entry_%d_next_valid", i)),
        /*load_enable=*/std::nullopt, reset);
    bb.RegisterWrite(data_regs[i],
                     is_last
                         ? push_data
                         : bb.Select(valid[i + 1], /*on_true=*/data[i + 1],
                                     /*on_false=*/push_data, SourceInfo(),
                                     absl::StrFormat("entry_%d_next_data", i)),
                     /*load_enable=*/
                     bb.Or(bb.And(shift, occupied_after_shift),
                           bb.And({bb.Not(shift), store, first_unoccupied}),
                           SourceInfo(), absl::StrFormat("entry_%d_load", i)),
                     reset);
  }

  if (options.add_overflow_assert) {
    // A value that arrives while every entry is occupied, with nothing popped,
    // has nowhere to go.
    BValue overflow = bb.And({push, valid[depth - 1], bb.Not(pop_ready)});
    bb.Assert(
        bb.Literal(Value::Token()), bb.Not(overflow),
        absl::StrFormat("Value pushed into valid-data buffer `%s` "
                        "while it was full; its producer must never do "
                        "that unless the consumer pops in the same cycle.",
                        name),
        /*label=*/
        SanitizeVerilogIdentifier(
            absl::StrCat("__xls_invariant_", name, "_no_overflow_A")));
  }

  return bb.Build();
}

}  // namespace xls::verilog
