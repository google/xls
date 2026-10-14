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
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Optional;

constexpr int64_t kDataWidth = 32;
constexpr std::string_view kResetName = "rst";

// Simulates a valid-data buffer block with the block interpreter.
class BufferSim {
 public:
  static absl::StatusOr<BufferSim> Create(Block* block) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<BlockContinuation> continuation,
                         kInterpreterBlockEvaluator.NewContinuation(block));
    BufferSim sim(std::move(continuation), block->GetResetPort().has_value());
    if (sim.has_reset_) {
      XLS_RETURN_IF_ERROR(sim.Cycle(std::nullopt, /*pop_ready=*/false,
                                    /*reset=*/true)
                              .status());
    }
    return sim;
  }

  // Runs one cycle, in which `push` (if any) arrives and the consumer is ready
  // if `pop_ready`. Returns the value presented to the consumer, if any.
  absl::StatusOr<std::optional<int64_t>> Cycle(std::optional<int64_t> push,
                                               bool pop_ready,
                                               bool reset = false) {
    absl::flat_hash_map<std::string, Value> inputs = {
        {std::string(kValidDataBufferPushDataPort),
         Value(UBits(push.value_or(0xBAD0BAD0), kDataWidth))},
        {std::string(kValidDataBufferPushValidPort),
         Value(UBits(push.has_value() ? 1 : 0, 1))},
        {std::string(kValidDataBufferPopReadyPort),
         Value(UBits(pop_ready ? 1 : 0, 1))},
    };
    if (has_reset_) {
      inputs[std::string(kResetName)] = Value(UBits(reset ? 1 : 0, 1));
    }
    XLS_RETURN_IF_ERROR(continuation_->RunOneCycle(inputs));
    const absl::flat_hash_map<std::string, Value>& outputs =
        continuation_->output_ports();
    if (outputs.at(kValidDataBufferPopValidPort).bits().IsZero()) {
      return std::nullopt;
    }
    XLS_ASSIGN_OR_RETURN(
        int64_t data, outputs.at(kValidDataBufferPopDataPort).bits().ToInt64());
    return data;
  }

  // The messages of the assertions that failed in the last cycle.
  std::vector<std::string> FailedAssertions() {
    return continuation_->events().GetAssertMessages();
  }

 private:
  BufferSim(std::unique_ptr<BlockContinuation> continuation, bool has_reset)
      : continuation_(std::move(continuation)), has_reset_(has_reset) {}

  std::unique_ptr<BlockContinuation> continuation_;
  bool has_reset_;
};

class ValidDataBufferTest : public IrTestBase {
 protected:
  absl::StatusOr<Block*> CreateBuffer(int64_t depth,
                                      bool add_overflow_assert = true) {
    return CreateValidDataBuffer(
        package_.get(), "buffer", package_->GetBitsType(kDataWidth), depth,
        {.reset =
             ValidDataBufferOptions::Reset{
                 .name = std::string(kResetName),
                 .behavior = {.asynchronous = false, .active_low = false}},
         .add_overflow_assert = add_overflow_assert});
  }

  std::unique_ptr<Package> package_ = CreatePackage();
};

TEST_F(ValidDataBufferTest, RejectsZeroDepth) {
  EXPECT_THAT(CreateBuffer(/*depth=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("depth of at least 1")));
}

TEST_F(ValidDataBufferTest, FallsThroughWhileEmpty) {
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, CreateBuffer(/*depth=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));

  // A value that's popped as it arrives isn't stored.
  EXPECT_THAT(sim.Cycle(1, /*pop_ready=*/true), IsOkAndHolds(Optional(1)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));

  // ... but a value that isn't is presented until it's popped.
  EXPECT_THAT(sim.Cycle(2, /*pop_ready=*/false), IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/false),
              IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));
}

TEST_F(ValidDataBufferTest, AcceptsPushesWhilePopping) {
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, CreateBuffer(/*depth=*/2));
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));

  // With one entry occupied.
  EXPECT_THAT(sim.Cycle(1, /*pop_ready=*/false), IsOkAndHolds(Optional(1)));
  EXPECT_THAT(sim.Cycle(2, /*pop_ready=*/true), IsOkAndHolds(Optional(1)));
  // With both entries occupied.
  EXPECT_THAT(sim.Cycle(3, /*pop_ready=*/false), IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.Cycle(4, /*pop_ready=*/true), IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.FailedAssertions(), IsEmpty());
  EXPECT_THAT(sim.Cycle(5, /*pop_ready=*/true), IsOkAndHolds(Optional(3)));
  EXPECT_THAT(sim.FailedAssertions(), IsEmpty());
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(4)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(5)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));
}

TEST_F(ValidDataBufferTest, DropsValueOnOverflowWithoutAssert) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block, CreateBuffer(/*depth=*/2, /*add_overflow_assert=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));

  XLS_ASSERT_OK(sim.Cycle(1, /*pop_ready=*/false).status());
  XLS_ASSERT_OK(sim.Cycle(2, /*pop_ready=*/false).status());
  XLS_ASSERT_OK(sim.Cycle(3, /*pop_ready=*/false).status());
  EXPECT_THAT(sim.FailedAssertions(), IsEmpty());
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(1)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(2)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));
}

TEST_F(ValidDataBufferTest, WithoutReset) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * block,
      CreateValidDataBuffer(package_.get(), "buffer",
                            package_->GetBitsType(kDataWidth),
                            /*depth=*/2, {.clock_name = "my_clk"}));
  ASSERT_TRUE(block->GetClockPort().has_value());
  EXPECT_EQ(block->GetClockPort()->name, "my_clk");
  EXPECT_FALSE(block->GetResetPort().has_value());
  for (Register* reg : block->GetRegisters()) {
    EXPECT_FALSE(reg->reset_value().has_value()) << reg->name();
  }

  // The interpreter starts with all registers zeroed, i.e., with the buffer
  // empty.
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));
  EXPECT_THAT(sim.Cycle(1, /*pop_ready=*/false), IsOkAndHolds(Optional(1)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(Optional(1)));
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));
}

class ValidDataBufferDepthTest : public ValidDataBufferTest,
                                 public ::testing::WithParamInterface<int64_t> {
};

TEST_P(ValidDataBufferDepthTest, FillsToDepthThenAssertsOnOverflow) {
  const int64_t depth = GetParam();
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, CreateBuffer(depth));
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));

  // The buffer holds `depth` values, presenting the oldest one throughout.
  for (int64_t i = 0; i < depth; ++i) {
    EXPECT_THAT(sim.Cycle(i, /*pop_ready=*/false), IsOkAndHolds(Optional(0)))
        << "push " << i;
    EXPECT_THAT(sim.FailedAssertions(), IsEmpty()) << "push " << i;
  }
  // One more has nowhere to go.
  EXPECT_THAT(sim.Cycle(depth, /*pop_ready=*/false), IsOkAndHolds(Optional(0)));
  EXPECT_THAT(sim.FailedAssertions(),
              ElementsAre(HasSubstr("while it was full")));
  // The stored values come out in order.
  for (int64_t i = 0; i < depth; ++i) {
    EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
                IsOkAndHolds(Optional(i)))
        << "pop " << i;
  }
  EXPECT_THAT(sim.Cycle(std::nullopt, /*pop_ready=*/true),
              IsOkAndHolds(std::nullopt));
}

TEST_P(ValidDataBufferDepthTest, MatchesQueueModel) {
  const int64_t depth = GetParam();
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, CreateBuffer(depth));
  XLS_ASSERT_OK_AND_ASSIGN(BufferSim sim, BufferSim::Create(block));

  absl::BitGen bitgen;
  std::deque<int64_t> model;
  int64_t next_value = 0;
  for (int64_t cycle = 0; cycle < 10000; ++cycle) {
    const bool pop_ready = absl::Bernoulli(bitgen, 0.5);
    // A value can only arrive if there's room for it (i.e., unless the buffer
    // is full and nothing is popped).
    const bool popping = pop_ready && !model.empty();
    std::optional<int64_t> push;
    if ((static_cast<int64_t>(model.size()) < depth || popping) &&
        absl::Bernoulli(bitgen, 0.7)) {
      push = next_value++;
    }

    std::optional<int64_t> expected =
        model.empty() ? push : std::make_optional(model.front());
    ASSERT_THAT(sim.Cycle(push, pop_ready), IsOkAndHolds(expected))
        << "cycle " << cycle;
    ASSERT_THAT(sim.FailedAssertions(), IsEmpty()) << "cycle " << cycle;

    if (push.has_value()) {
      model.push_back(*push);
    }
    if (pop_ready && !model.empty()) {
      model.pop_front();
    }
    ASSERT_LE(static_cast<int64_t>(model.size()), depth);
  }
}

INSTANTIATE_TEST_SUITE_P(ValidDataBufferDepthTestInstantiation,
                         ValidDataBufferDepthTest,
                         ::testing::Values(1, 2, 3, 5),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace xls::verilog
