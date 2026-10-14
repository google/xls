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

#ifndef XLS_CODEGEN_VALID_DATA_BUFFER_H_
#define XLS_CODEGEN_VALID_DATA_BUFFER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/type.h"

namespace xls::verilog {

// Port names of the blocks created by `CreateValidDataBuffer`.
inline constexpr std::string_view kValidDataBufferPushDataPort = "push_data";
inline constexpr std::string_view kValidDataBufferPushValidPort = "push_valid";
inline constexpr std::string_view kValidDataBufferPopReadyPort = "pop_ready";
inline constexpr std::string_view kValidDataBufferPopDataPort = "pop_data";
inline constexpr std::string_view kValidDataBufferPopValidPort = "pop_valid";

struct ValidDataBufferOptions {
  // The name of the block's clock port.
  std::string clock_name = "clk";

  struct Reset {
    std::string name;
    ResetBehavior behavior;
  };
  // The block's reset port, if any. Without one, the buffer's registers have no
  // reset values.
  std::optional<Reset> reset;

  // If true, the block asserts that nothing is pushed while it's full (unless
  // something is popped in the same cycle), since the pushed value would be
  // lost.
  bool add_overflow_assert = false;
};

// Creates a block (named `name`) that buffers a valid-data stream (one with no
// ready signal) for a ready/valid consumer, holding up to `depth` values of
// type `data_type`, in order, until the consumer is ready for them. `depth`
// must be at least 1.
//
// Since the push side has no `push_ready` output, the producer must never push
// while the buffer is full, unless the consumer pops in the same cycle. This
// suits producers that can't be stalled but can bound how many of their values
// are outstanding at once, such as a RAM's read port.
//
// The buffer has zero latency (first-word fall-through): while it's empty, a
// pushed value is presented to the consumer in the same cycle, and isn't stored
// if it's popped right away. Stored values shift toward the head entry as
// values are popped, so `pop_data` always comes from the head entry (or the
// arriving value) through a single 2:1 mux, at the cost of moving every stored
// value on each pop; this is intended for small depths.
//
// The block's ports (besides clock & reset) are:
//
//   push_data, push_valid (inputs): the value arriving in this cycle, if
//       push_valid is asserted.
//   pop_ready (input): whether the consumer is ready for a value.
//   pop_data, pop_valid (outputs): the oldest buffered value or, if the buffer
//       is empty, the value arriving in this cycle (if any). A value is popped
//       in any cycle where pop_valid & pop_ready are both asserted.
absl::StatusOr<Block*> CreateValidDataBuffer(
    Package* package, std::string_view name, Type* data_type, int64_t depth,
    const ValidDataBufferOptions& options = {});

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_VALID_DATA_BUFFER_H_
