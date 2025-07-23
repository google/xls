// Copyright 2025 The XLS Authors
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

#include "xls/passes/proc_state_bits_shattering_pass.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class ProcStateBitsShatteringPassTest : public IrTestBase {
 protected:
  ProcStateBitsShatteringPassTest() = default;

  absl::StatusOr<bool> Run(
      Package* p,
      std::optional<int64_t> split_next_value_selects = std::nullopt) {
    PassResults results;
    OptimizationPassOptions options;
    options.split_next_value_selects = split_next_value_selects;
    OptimizationContext context;
    return ProcStateBitsShatteringPass().Run(p, options, &results, context);
  }
};

TEST_F(ProcStateBitsShatteringPassTest, SimpleSplitStateElement) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_out, p->CreateStreamingChannel("x_out", ChannelOps::kSendOnly,
                                                 p->GetBitsType(16)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_out, p->CreateStreamingChannel("y_out", ChannelOps::kSendOnly,
                                                 p->GetBitsType(16)));
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 16)));
  BValue y = pb.StateElement("y", Value(UBits(0, 16)));
  BValue send_x =
      pb.Send(x_out, pb.Literal(Value::Token()), x, /*loc=*/{}, "send_x");
  BValue send_y =
      pb.Send(y_out, pb.Literal(Value::Token()), y, /*loc=*/{}, "send_y");
  BValue x_lo = pb.BitSlice(x, /*start=*/0, /*width=*/6);
  BValue x_mid = pb.BitSlice(x, /*start=*/6, /*width=*/1);
  BValue x_hi = pb.BitSlice(x, /*start=*/7, /*width=*/9);
  BValue new_x_mid = pb.Select(pb.OrReduce(x_lo), {x_mid, pb.Not(x_mid)});
  pb.Next(x, pb.Concat({x_hi, new_x_mid, x_lo}));
  pb.Next(y, x);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(Run(p.get(), /*split_next_value_selects=*/2), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * new_x, proc->GetStateElement("x"));
  EXPECT_THAT(new_x->type(), m::Type("(bits[6], bits[1], bits[9])"));
  EXPECT_THAT(send_x.node(),
              m::Send(m::Literal(Value::Token()),
                      m::Concat(m::TupleIndex(m::StateRead("x"), 2),
                                m::TupleIndex(m::StateRead("x"), 1),
                                m::TupleIndex(m::StateRead("x"), 0))));
  EXPECT_THAT(send_y.node(),
              m::Send(m::Literal(Value::Token()), m::StateRead("y")))
      << proc->DumpIr();
}

void IrFuzzProcStateBitsShattering(FuzzPackageWithArgs fuzz_package_with_args) {
  ProcStateBitsShatteringPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzProcStateBitsShattering)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
