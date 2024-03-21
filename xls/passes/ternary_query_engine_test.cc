// Copyright 2020 The XLS Authors
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

#include "xls/passes/ternary_query_engine.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/ternary_evaluator.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class TernaryQueryEngineTest : public IrTestBase {
 protected:
  // Create a BValue with known bits equal to the given ternary vector. Created
  // using a param and AND/OR masks.
  BValue MakeValueWithKnownBits(std::string_view name,
                                TernaryVector known_bits, FunctionBuilder* fb) {
    absl::InlinedVector<bool, 1> known_zeros;
    absl::InlinedVector<bool, 1> known_ones;
    for (TernaryValue value : known_bits) {
      known_zeros.push_back(value == TernaryValue::kKnownZero);
      known_ones.push_back(value == TernaryValue::kKnownOne);
    }
    BValue and_mask = fb->Literal(bits_ops::Not(Bits(known_zeros)));
    BValue or_mask = fb->Literal(Bits(known_ones));
    return fb->Or(or_mask,
                  fb->And(and_mask, fb->Param(name, fb->package()->GetBitsType(
                                                        known_bits.size()))));
  }

  // Runs QueryEngine on the op created with the passed in function. The
  // inputs to the op is crafted to have known bits equal to the given
  // TernaryVectors.
  absl::StatusOr<std::string> RunOnBinaryOp(
      std::string_view lhs_known_bits, std::string_view rhs_known_bits,
      std::function<void(BValue, BValue, FunctionBuilder*)> make_op) {
    Package p("test_package");
    FunctionBuilder fb("f", &p);
    BValue lhs = MakeValueWithKnownBits(
        "lhs", StringToTernaryVector(lhs_known_bits).value(), &fb);
    BValue rhs = MakeValueWithKnownBits(
        "rhs", StringToTernaryVector(rhs_known_bits).value(), &fb);
    make_op(lhs, rhs, &fb);
    XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
    VLOG(3) << f->DumpIr();
    TernaryQueryEngine query_engine;
    XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());
    return query_engine.ToString(f->return_value());
  }
};

TEST_F(TernaryQueryEngineTest, Uge) {
  auto make_uge = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->UGe(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b11X", make_uge), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX1X", make_uge), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_uge), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_uge), IsOkAndHolds("0b0"));
}

TEST_F(TernaryQueryEngineTest, Ugt) {
  auto make_ugt = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->UGt(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ugt), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX1X", make_ugt), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ugt), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ugt), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ugt), IsOkAndHolds("0b0"));
}

TEST_F(TernaryQueryEngineTest, Ule) {
  auto make_ule = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->ULe(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ule), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b000", "0bX1X", make_ule), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ule), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ule), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ule), IsOkAndHolds("0b1"));
}

TEST_F(TernaryQueryEngineTest, Ult) {
  auto make_ult = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->ULt(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXXX", "0b111", make_ult), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b000", "0bX1X", make_ult), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b111", "0bX10", make_ult), IsOkAndHolds("0b0"));
  EXPECT_THAT(RunOnBinaryOp("0b1XX", "0bX11", make_ult), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b0XX", "0b1XX", make_ult), IsOkAndHolds("0b1"));
}

TEST_F(TernaryQueryEngineTest, Ne) {
  auto make_ne = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Ne(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b110", make_ne), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0bXX1", "0b111", make_ne), IsOkAndHolds("0bX"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b111", make_ne), IsOkAndHolds("0b1"));
  EXPECT_THAT(RunOnBinaryOp("0b011", "0b011", make_ne), IsOkAndHolds("0b0"));
}

TEST_F(TernaryQueryEngineTest, Gate) {
  auto make_gate = [](BValue lhs, BValue rhs, FunctionBuilder* fb) {
    fb->Gate(lhs, rhs);
  };
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b110", make_gate), IsOkAndHolds("0bXX0"));
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b111", make_gate), IsOkAndHolds("0bXXX"));
  EXPECT_THAT(RunOnBinaryOp("0bX", "0b0X1", make_gate), IsOkAndHolds("0b0XX"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b110", make_gate), IsOkAndHolds("0b110"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b111", make_gate), IsOkAndHolds("0b111"));
  EXPECT_THAT(RunOnBinaryOp("0b1", "0b0X1", make_gate), IsOkAndHolds("0b0X1"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b110", make_gate), IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b111", make_gate), IsOkAndHolds("0b000"));
  EXPECT_THAT(RunOnBinaryOp("0b0", "0b0X1", make_gate), IsOkAndHolds("0b000"));
}


}  // namespace
}  // namespace xls
