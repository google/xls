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

#include "xls/codegen/register_chaining_analysis.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"

using testing::AnyOf;
using testing::ElementsAre;
using testing::UnorderedElementsAre;

namespace xls::verilog {
namespace {

class RegisterChainingAnalysisTest : public IrTestBase {
 public:
  template <int64_t kCount, typename Checker>
  void CheckAllPermutations(std::array<RegisterData, kCount> arr,
                            Checker check_order) {
    auto cmp_register_data = [](const RegisterData& l, const RegisterData& r) {
      return l.reg->name() < r.reg->name();
    };
    absl::c_sort(arr, cmp_register_data);
    int i = 0;
    do {
      ++i;
      check_order(arr);
    } while (!HasFailure() && absl::c_next_permutation(arr, cmp_register_data));
    RecordProperty("permutations", i);
  }

  void MarkAllPairs(ConcurrentStageGroups& csg, int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      for (int64_t j = i + 1; j <= end; ++j) {
        csg.MarkMutuallyExclusive(i, j);
      }
    }
  }

  // Create a straight-shot pipeline like in the LongChain test.
  //
  // (write/32 'A (input/32  ))) ; stage 0
  // (write/32 'B (read/32 'A))) ; stage 1
  // ; other stages
  // (output/32   (read/32 'last))) ; stage kCount
  template <int64_t kCount>
  absl::StatusOr<std::array<RegisterData, kCount>> CreateStraightShot(
      Package* p) {
    BlockBuilder bb(TestName(), p);
    XLS_RETURN_IF_ERROR(bb.AddClockPort("clk"));
    XLS_ASSIGN_OR_RETURN(auto result, CreateStraightShot<kCount>(bb, 0));
    XLS_RETURN_IF_ERROR(bb.Build().status());
    return result;
  }

  // Create a straight-shot pipeline like in the LongChain test starting at a
  // particular stage.
  //
  // (write/32 'A (input/32  ))) ; stage start
  // (write/32 'B (read/32 'A))) ; stage start + 1
  // ; other stages
  // (output/32   (read/32 'last))) ; stage start + kCount
  template <int64_t kCount>
  absl::StatusOr<std::array<RegisterData, kCount>> CreateStraightShot(
      BlockBuilder& bb, int64_t start_stage, std::string_view prefix = "") {
    Package* p = bb.package();
    auto input =
        bb.InputPort(absl::StrFormat("%sIN", prefix), p->GetBitsType(32));
    std::array<RegisterRead*, kCount> reads{};
    std::array<Register*, kCount> regs{};
    std::array<RegisterWrite*, kCount> writes{};
    for (int64_t i = 0; i < kCount; ++i) {
      auto reg_name = absl::StrFormat("%sreg_%c", prefix, 'A' + i);
      reads[i] = bb.InsertRegister(reg_name,
                                   i == 0 ? input : BValue(reads[i - 1], &bb))
                     .node()
                     ->As<RegisterRead>();
      XLS_ASSIGN_OR_RETURN(regs[i], bb.block()->GetRegister(reg_name));
      XLS_ASSIGN_OR_RETURN(writes[i], bb.block()->GetRegisterWrite(regs[i]));
    }
    bb.OutputPort(absl::StrFormat("%sOUT", prefix), BValue(reads.back(), &bb));
    std::array<RegisterData, kCount> result{};
    for (int64_t i = 0; i < kCount; ++i) {
      result[i] = RegisterData{.reg = regs[i],
                               .read = reads[i],
                               .read_stage = start_stage + i + 1,
                               .write = writes[i],
                               .write_stage = start_stage + i};
    }
    return result;
  }

  // Create a straight-shot pipeline with a loopback like LoopbackRegister test
  //
  // (write/32 'A (read/32 'end)))                  ; stage 0
  // (write/32 'B (read/32 'A)))                    ; stage 1
  // ; other stages
  // (write/32 'end (+ 1 (read/32 'end_less_one)))) ; stage end
  template <int64_t kCount>
  absl::StatusOr<std::array<RegisterData, kCount>> CreateLoopback(Package* p) {
    BlockBuilder bb(TestName(), p);
    XLS_RETURN_IF_ERROR(bb.AddClockPort("clk"));
    std::array<BValue, kCount> reads{};
    std::array<Register*, kCount> regs{};
    std::array<BValue, kCount> writes{};
    for (int64_t i = 0; i < kCount; ++i) {
      auto reg_name = absl::StrFormat("reg_%c", 'A' + i);
      XLS_ASSIGN_OR_RETURN(
          regs[i], bb.block()->AddRegister(reg_name, p->GetBitsType(32)));
      EXPECT_EQ(regs[i]->name(), reg_name);
    }
    for (int64_t i = 0; i < kCount; ++i) {
      reads[i] = bb.RegisterRead(regs[i]);
    }
    for (int64_t i = 0; i < kCount; ++i) {
      if (i != kCount - 1) {
        writes[i] =
            bb.RegisterWrite(regs[i], i == 0 ? reads.back() : reads[i - 1]);
      } else {
        writes[i] = bb.RegisterWrite(
            regs[i], bb.Add(bb.Literal(UBits(1, 32)), reads[i - 1]));
      }
    }
    XLS_RETURN_IF_ERROR(bb.Build().status());
    std::array<RegisterData, kCount> result{};
    for (int64_t i = 0; i < kCount; ++i) {
      result[i] =
          RegisterData{.reg = regs[i],
                       .read = reads[i].node()->template As<RegisterRead>(),
                       .read_stage = (i + 1) % kCount,
                       .write = writes[i].node()->template As<RegisterWrite>(),
                       .write_stage = i};
    }
    return result;
  }
};

MATCHER_P(Reg, name, "Register name mismatch.") {
  const RegisterData& rd = arg;
  return testing::ExplainMatchResult(name, rd.reg->name(), result_listener);
}

TEST_F(RegisterChainingAnalysisTest, LongChain) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  // pipeline:
  // (
  //     ((write/32 'A (input/32  )))
  //     ((write/32 'B (read/32 'A)))
  //     ((write/32 'C (read/32 'B)))
  //     ((write/32 'D (read/32 'C)))
  //     ((write/32 'E (read/32 'D)))
  //     ((write/32 'F (read/32 'E)))
  //     ((output/32   (read/32 'F)))
  // )
  auto input_32 = bb.InputPort("IN_32", p->GetBitsType(32));
  auto read_a = bb.InsertRegister("reg_A", input_32);
  auto read_b = bb.InsertRegister("reg_B", read_a);
  auto read_c = bb.InsertRegister("reg_C", read_b);
  auto read_d = bb.InsertRegister("reg_D", read_c);
  auto read_e = bb.InsertRegister("reg_E", read_d);
  auto read_f = bb.InsertRegister("reg_F", read_e);
  bb.OutputPort("OUT_32", read_f);

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * a, blk->GetRegister("reg_A"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_a, blk->GetRegisterWrite(a));
  XLS_ASSERT_OK_AND_ASSIGN(Register * b, blk->GetRegister("reg_B"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_b, blk->GetRegisterWrite(b));
  XLS_ASSERT_OK_AND_ASSIGN(Register * c, blk->GetRegister("reg_C"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_c, blk->GetRegisterWrite(c));
  XLS_ASSERT_OK_AND_ASSIGN(Register * d, blk->GetRegister("reg_D"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_d, blk->GetRegisterWrite(d));
  XLS_ASSERT_OK_AND_ASSIGN(Register * e, blk->GetRegister("reg_E"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_e, blk->GetRegisterWrite(e));
  XLS_ASSERT_OK_AND_ASSIGN(Register * f, blk->GetRegister("reg_F"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_f, blk->GetRegisterWrite(f));

  RegisterData a_data{
      .reg = a,
      .read = read_a.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_a,
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = b,
      .read = read_b.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = write_b,
      .write_stage = 1,
  };
  RegisterData c_data{
      .reg = c,
      .read = read_c.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_c,
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = d,
      .read = read_d.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = write_d,
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = e,
      .read = read_e.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = write_e,
      .write_stage = 4,
  };
  RegisterData f_data{
      .reg = f,
      .read = read_f.node()->As<RegisterRead>(),
      .read_stage = 6,
      .write = write_f,
      .write_stage = 5,
  };
  auto check_order = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                UnorderedElementsAre(ElementsAre(a_data, b_data, c_data, d_data,
                                                 e_data, f_data)));
  };
  CheckAllPermutations<6>({a_data, b_data, c_data, d_data, e_data, f_data},
                          check_order);
}

TEST_F(RegisterChainingAnalysisTest, BrokenChain) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  // pipeline:
  // (
  //     ((write/32 'A (input/32  )))
  //     ((write/32 'B (read/32 'A)))
  //     ((write/32 'C (read/32 'B)))
  //     ((write/32 'D (+ 1 (read/32 'C))))
  //     ((write/32 'E (read/32 'D)))
  //     ((write/32 'F (read/32 'E)))
  //     ((output/32   (read/32 'F)))
  // )
  auto input_32 = bb.InputPort("IN_32", p->GetBitsType(32));
  auto read_a = bb.InsertRegister("reg_A", input_32);
  auto read_b = bb.InsertRegister("reg_B", read_a);
  auto read_c = bb.InsertRegister("reg_C", read_b);
  auto read_d =
      bb.InsertRegister("reg_D", bb.Add(read_c, bb.Literal(UBits(1, 32))));
  auto read_e = bb.InsertRegister("reg_E", read_d);
  auto read_f = bb.InsertRegister("reg_F", read_e);
  bb.OutputPort("OUT_32", read_f);

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * a, blk->GetRegister("reg_A"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_a, blk->GetRegisterWrite(a));
  XLS_ASSERT_OK_AND_ASSIGN(Register * b, blk->GetRegister("reg_B"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_b, blk->GetRegisterWrite(b));
  XLS_ASSERT_OK_AND_ASSIGN(Register * c, blk->GetRegister("reg_C"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_c, blk->GetRegisterWrite(c));
  XLS_ASSERT_OK_AND_ASSIGN(Register * d, blk->GetRegister("reg_D"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_d, blk->GetRegisterWrite(d));
  XLS_ASSERT_OK_AND_ASSIGN(Register * e, blk->GetRegister("reg_E"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_e, blk->GetRegisterWrite(e));
  XLS_ASSERT_OK_AND_ASSIGN(Register * f, blk->GetRegister("reg_F"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_f, blk->GetRegisterWrite(f));

  RegisterData a_data{
      .reg = a,
      .read = read_a.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_a,
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = b,
      .read = read_b.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = write_b,
      .write_stage = 1,
  };
  RegisterData c_data{
      .reg = c,
      .read = read_c.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_c,
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = d,
      .read = read_d.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = write_d,
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = e,
      .read = read_e.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = write_e,
      .write_stage = 4,
  };
  RegisterData f_data{
      .reg = f,
      .read = read_f.node()->As<RegisterRead>(),
      .read_stage = 6,
      .write = write_f,
      .write_stage = 5,
  };
  auto check_order = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                UnorderedElementsAre(ElementsAre(a_data, b_data, c_data),
                                     ElementsAre(d_data, e_data, f_data)));
  };
  CheckAllPermutations<6>({a_data, b_data, c_data, d_data, e_data, f_data},
                          check_order);
}

TEST_F(RegisterChainingAnalysisTest, LoopbackRegister) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  // pipeline:
  // (
  //     ((write/32 'A (read/32 'F)))
  //     ((write/32 'B (read/32 'A)))
  //     ((write/32 'C (read/32 'B)))
  //     ((write/32 'D (read/32 'C)))
  //     ((write/32 'E (read/32 'D)))
  //     ((write/32 'F (read/32 'E)))
  // )
  // A/F is a loopback register.
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * f, bb.block()->AddRegister("reg_F", p->GetBitsType(32)));
  auto read_f = bb.RegisterRead(f);
  auto read_a = bb.InsertRegister("reg_A", read_f);
  auto read_b = bb.InsertRegister("reg_B", read_a);
  auto read_c = bb.InsertRegister("reg_C", read_b);
  auto read_d = bb.InsertRegister("reg_D", read_c);
  auto read_e = bb.InsertRegister("reg_E", read_d);
  auto write_f = bb.RegisterWrite(f, read_e).node()->As<RegisterWrite>();

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * a, blk->GetRegister("reg_A"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_a, blk->GetRegisterWrite(a));
  XLS_ASSERT_OK_AND_ASSIGN(Register * b, blk->GetRegister("reg_B"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_b, blk->GetRegisterWrite(b));
  XLS_ASSERT_OK_AND_ASSIGN(Register * c, blk->GetRegister("reg_C"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_c, blk->GetRegisterWrite(c));
  XLS_ASSERT_OK_AND_ASSIGN(Register * d, blk->GetRegister("reg_D"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_d, blk->GetRegisterWrite(d));
  XLS_ASSERT_OK_AND_ASSIGN(Register * e, blk->GetRegister("reg_E"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_e, blk->GetRegisterWrite(e));

  RegisterData a_data{
      .reg = a,
      .read = read_a.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_a,
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = b,
      .read = read_b.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = write_b,
      .write_stage = 1,
  };
  RegisterData c_data{
      .reg = c,
      .read = read_c.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_c,
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = d,
      .read = read_d.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = write_d,
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = e,
      .read = read_e.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = write_e,
      .write_stage = 4,
  };
  RegisterData f_data{
      .reg = f,
      .read = read_f.node()->As<RegisterRead>(),
      // loopback
      .read_stage = 0,
      .write = write_f,
      .write_stage = 5,
  };
  auto check_order = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                // Loopback needs to be first.
                UnorderedElementsAre(ElementsAre(f_data, a_data, b_data, c_data,
                                                 d_data, e_data)));
  };
  CheckAllPermutations<6>({a_data, b_data, c_data, d_data, e_data, f_data},
                          check_order);
}
TEST_F(RegisterChainingAnalysisTest, LoopbackRegisterChainSplits) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  // pipeline:
  // (let ((ReadB (read/32 'B))))
  // (
  //     ((write/32 'A ReadB))
  //     ((write/32 'B (read/32 'A)))
  //     ((write/32 'C ReadB))
  //     ((write/32 'D (read/32 'C)))
  //     ((write/32 'E (read/32 'D)))
  //     ((output/32   (read/32 'E)))
  // )
  // A is a loopback register.
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * b, bb.block()->AddRegister("reg_B", p->GetBitsType(32)));
  auto read_b = bb.RegisterRead(b);
  auto read_a = bb.InsertRegister("reg_A", read_b);
  auto write_b = bb.RegisterWrite(b, read_a).node()->As<RegisterWrite>();
  auto read_c = bb.InsertRegister("reg_C", read_b);
  auto read_d = bb.InsertRegister("reg_D", read_c);
  auto read_e = bb.InsertRegister("reg_E", read_d);
  bb.OutputPort("OUT", read_e);

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * a, blk->GetRegister("reg_A"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_a, blk->GetRegisterWrite(a));
  XLS_ASSERT_OK_AND_ASSIGN(Register * c, blk->GetRegister("reg_C"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_c, blk->GetRegisterWrite(c));
  XLS_ASSERT_OK_AND_ASSIGN(Register * d, blk->GetRegister("reg_D"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_d, blk->GetRegisterWrite(d));
  XLS_ASSERT_OK_AND_ASSIGN(Register * e, blk->GetRegister("reg_E"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_e, blk->GetRegisterWrite(e));

  RegisterData a_data{
      .reg = a,
      .read = read_a.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_a,
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = b,
      .read = read_b.node()->As<RegisterRead>(),
      .read_stage = 0,
      .write = write_b,
      .write_stage = 1,
  };
  RegisterData c_data{
      .reg = c,
      .read = read_c.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_c,
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = d,
      .read = read_d.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = write_d,
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = e,
      .read = read_e.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = write_e,
      .write_stage = 4,
  };
  auto check_order = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                // Loopback needs to be first.
                UnorderedElementsAre(ElementsAre(b_data, a_data),
                                     ElementsAre(c_data, d_data, e_data)));
  };
  CheckAllPermutations<5>({a_data, b_data, c_data, d_data, e_data},
                          check_order);
}

TEST_F(RegisterChainingAnalysisTest, MultipleChains) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  // pipeline:
  // (
  //     ((write/32 'A (input/32  ))
  //      (write/64 'X (input/64  )))
  //     ((write/32 'B (read/32 'A))
  //      (write/64 'Y (read/64 'X)))
  //     ((write/32 'C (read/32 'B))
  //      (write/64 'Z (read/64 'Y)))
  //     ((output/32   (read/32 'C))
  //      (output/64   (read/64 'Z)))
  // )
  // 2 chains. a 32-bit and 64-bit one.
  auto input_32 = bb.InputPort("IN_32", p->GetBitsType(32));
  auto input_64 = bb.InputPort("IN_64", p->GetBitsType(64));
  auto read_a = bb.InsertRegister("reg_A", input_32);
  auto read_x = bb.InsertRegister("reg_X", input_64);
  auto read_b = bb.InsertRegister("reg_B", read_a);
  auto read_y = bb.InsertRegister("reg_Y", read_x);
  auto read_c = bb.InsertRegister("reg_C", read_b);
  auto read_z = bb.InsertRegister("reg_Z", read_y);
  bb.OutputPort("OUT_32", read_c);
  bb.OutputPort("OUT_64", read_z);

  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * a, blk->GetRegister("reg_A"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_a, blk->GetRegisterWrite(a));
  XLS_ASSERT_OK_AND_ASSIGN(Register * b, blk->GetRegister("reg_B"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_b, blk->GetRegisterWrite(b));
  XLS_ASSERT_OK_AND_ASSIGN(Register * c, blk->GetRegister("reg_C"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_c, blk->GetRegisterWrite(c));
  XLS_ASSERT_OK_AND_ASSIGN(Register * x, blk->GetRegister("reg_X"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_x, blk->GetRegisterWrite(x));
  XLS_ASSERT_OK_AND_ASSIGN(Register * y, blk->GetRegister("reg_Y"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_y, blk->GetRegisterWrite(y));
  XLS_ASSERT_OK_AND_ASSIGN(Register * z, blk->GetRegister("reg_Z"));
  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * write_z, blk->GetRegisterWrite(z));

  RegisterData a_data{
      .reg = a,
      .read = read_a.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_a,
      .write_stage = 0,
  };
  RegisterData x_data{
      .reg = x,
      .read = read_x.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = write_x,
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = b,
      .read = read_b.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = write_b,
      .write_stage = 1,
  };
  RegisterData y_data{
      .reg = y,
      .read = read_y.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = write_y,
      .write_stage = 1,
  };
  RegisterData c_data{
      .reg = c,
      .read = read_c.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_c,
      .write_stage = 2,
  };
  RegisterData z_data{
      .reg = z,
      .read = read_z.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = write_z,
      .write_stage = 2,
  };
  auto check_order = [&](std::array<RegisterData, 6> data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                UnorderedElementsAre(ElementsAre(a_data, b_data, c_data),
                                     ElementsAre(x_data, y_data, z_data)));
  };
  CheckAllPermutations<6>({a_data, b_data, c_data, x_data, y_data, z_data},
                          check_order);
}

TEST_F(RegisterChainingAnalysisTest, LoopbackContinues) {
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((write/32 'A (read/32 'C_back))) ; stage 0
  //     ((write/32 'B (read/32 'A)))      ; stage 1
  //     ((let ((b_val (read/32 'B)))      ; stage 2
  //           (write/32 'C_back b_val)
  //           (write/32 'C_cont b_val))
  //     ((write/32 'D (read/32 'C_cont))) ; stage 3
  //     ((write/32 'E (read/32 'D)))      ; stage 4
  //     ((write/32 'F (read/32 'E)))      ; stage 5
  //     ((output/32   (read/32 'F)))      ; stage 6
  // )
  // Needs to be split into 2 sections C_back, A, B & C_cont
  BlockBuilder bb(TestName(), p.get());
  auto* t32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_a, bb.block()->AddRegister("A", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_b, bb.block()->AddRegister("B", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_c_cont,
                           bb.block()->AddRegister("C_cont", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_c_back,
                           bb.block()->AddRegister("C_back", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_d, bb.block()->AddRegister("D", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_e, bb.block()->AddRegister("E", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_f, bb.block()->AddRegister("F", t32));

  BValue a_read = bb.RegisterRead(reg_a, SourceInfo(), reg_a->name());
  BValue b_read = bb.RegisterRead(reg_b, SourceInfo(), reg_b->name());
  BValue c_cont_read =
      bb.RegisterRead(reg_c_cont, SourceInfo(), reg_c_cont->name());
  BValue c_back_read =
      bb.RegisterRead(reg_c_back, SourceInfo(), reg_c_back->name());
  BValue d_read = bb.RegisterRead(reg_d, SourceInfo(), reg_d->name());
  BValue e_read = bb.RegisterRead(reg_e, SourceInfo(), reg_e->name());
  BValue f_read = bb.RegisterRead(reg_f, SourceInfo(), reg_f->name());

  BValue a_write = bb.RegisterWrite(
      reg_a, c_back_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_a->name() + "_write");
  BValue b_write = bb.RegisterWrite(reg_b, a_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_b->name() + "_write");
  BValue c_cont_write = bb.RegisterWrite(
      reg_c_cont, b_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_c_cont->name() + "_write");
  BValue c_back_write = bb.RegisterWrite(
      reg_c_back, b_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_c_back->name() + "_write");
  BValue d_write = bb.RegisterWrite(
      reg_d, c_cont_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_d->name() + "_write");
  BValue e_write = bb.RegisterWrite(reg_e, d_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_e->name() + "_write");
  BValue f_write = bb.RegisterWrite(reg_f, e_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_f->name() + "_write");
  bb.OutputPort("OUT", f_read);

  RegisterData a_data{
      .reg = reg_a,
      .read = a_read.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = a_write.node()->As<RegisterWrite>(),
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = reg_b,
      .read = b_read.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = b_write.node()->As<RegisterWrite>(),
      .write_stage = 1,
  };
  RegisterData c_cont_data{
      .reg = reg_c_cont,
      .read = c_cont_read.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = c_cont_write.node()->As<RegisterWrite>(),
      .write_stage = 2,
  };
  RegisterData c_back_data{
      .reg = reg_c_back,
      .read = c_back_read.node()->As<RegisterRead>(),
      .read_stage = 0,
      .write = c_back_write.node()->As<RegisterWrite>(),
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = reg_d,
      .read = d_read.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = d_write.node()->As<RegisterWrite>(),
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = reg_e,
      .read = e_read.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = e_write.node()->As<RegisterWrite>(),
      .write_stage = 4,
  };
  RegisterData f_data{
      .reg = reg_f,
      .read = f_read.node()->As<RegisterRead>(),
      .read_stage = 6,
      .write = f_write.node()->As<RegisterWrite>(),
      .write_stage = 5,
  };
  auto check_order = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(
        rc.chains(),
        // Loopback needs to be first.
        // Cases where a value extends beyond the loopback can't be on the same
        // chain as the loopback.
        AnyOf(UnorderedElementsAre(
                  ElementsAre(c_back_data, a_data, b_data),
                  ElementsAre(c_cont_data, d_data, e_data, f_data)),
              UnorderedElementsAre(
                  ElementsAre(c_back_data, a_data),
                  ElementsAre(b_data, c_cont_data, d_data, e_data, f_data)),
              UnorderedElementsAre(ElementsAre(c_back_data),
                                   ElementsAre(a_data, b_data, c_cont_data,
                                               d_data, e_data, f_data))));
  };
  CheckAllPermutations<7>(
      {a_data, b_data, c_cont_data, c_back_data, d_data, e_data, f_data},
      check_order);
}

TEST_F(RegisterChainingAnalysisTest, BasicSelectMutexList) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((no_mutex)        (write/32 'A (input/32  )))
  //     ((mutex_region 'X) (write/32 'B (read/32 'A)))
  //     ((mutex_region 'X) (write/32 'C (read/32 'B)))
  //     ((mutex_region 'X) (write/32 'D (read/32 'C)))
  //     ((no_mutex)        (write/32 'E (read/32 'D)))
  //     ((no_mutex)        (write/32 'F (read/32 'E)))
  //     ((no_mutex)        (output/32   (read/32 'F)))
  // )
  // NB can combine registers 'B and 'C. Still need to write into 'A and 'D for
  // real.
  XLS_ASSERT_OK_AND_ASSIGN((const std::array<RegisterData, kRegCnt> datas),
                           CreateStraightShot<kRegCnt>(p.get()));

  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 1);
  csg.MarkMutuallyExclusive(1, 2);
  csg.MarkMutuallyExclusive(1, 3);
  csg.MarkMutuallyExclusive(2, 3);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(), UnorderedElementsAre(ElementsAre(
                                 Reg("reg_A"), Reg("reg_B"), Reg("reg_C"),
                                 Reg("reg_D"), Reg("reg_E"), Reg("reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split,
                UnorderedElementsAre(ElementsAre(Reg("reg_B"), Reg("reg_C"))));
  };
  CheckAllPermutations<kRegCnt>(datas, check_mutex);
}

TEST_F(RegisterChainingAnalysisTest, SplitMutexRegion) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((mutex_region 'X) (write/32 'A (input/32  )))  ; stage 0
  //     ((mutex_region 'X) (write/32 'B (read/32 'A)))  ; stage 1
  //     ((mutex_region 'X) (write/32 'C (read/32 'B)))  ; stage 2
  //     ((mutex_region 'Y) (write/32 'D (read/32 'C)))  ; stage 3
  //     ((mutex_region 'Y) (write/32 'E (read/32 'D)))  ; stage 4
  //     ((mutex_region 'Y) (write/32 'F (read/32 'E)))  ; stage 5
  //     ((no_mutex)        (output/32   (read/32 'F)))  ; stage 6
  // )
  // NB can combine registers 'A & 'B and 'D & 'E
  XLS_ASSERT_OK_AND_ASSIGN((const std::array<RegisterData, kRegCnt> datas),
                           CreateStraightShot<kRegCnt>(p.get()));
  EXPECT_THAT(datas, ElementsAre(Reg("reg_A"), Reg("reg_B"), Reg("reg_C"),
                                 Reg("reg_D"), Reg("reg_E"), Reg("reg_F")));

  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 1);
  csg.MarkMutuallyExclusive(0, 1);
  csg.MarkMutuallyExclusive(0, 2);
  csg.MarkMutuallyExclusive(1, 2);
  csg.MarkMutuallyExclusive(3, 4);
  csg.MarkMutuallyExclusive(3, 5);
  csg.MarkMutuallyExclusive(4, 5);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(), UnorderedElementsAre(ElementsAre(
                                 Reg("reg_A"), Reg("reg_B"), Reg("reg_C"),
                                 Reg("reg_D"), Reg("reg_E"), Reg("reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split,
                UnorderedElementsAre(ElementsAre(Reg("reg_A"), Reg("reg_B")),
                                     ElementsAre(Reg("reg_D"), Reg("reg_E"))));
  };
  CheckAllPermutations<kRegCnt>(datas, check_mutex);
}

TEST_F(RegisterChainingAnalysisTest, MutexTouchesEnd) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((no_mutex)        (write/32 'A (input/32  ))) ; stage 0
  //     ((no_mutex)        (write/32 'B (read/32 'A))) ; stage 1
  //     ((no_mutex)        (write/32 'C (read/32 'B))) ; stage 2
  //     ((mutex_region 'X) (write/32 'D (read/32 'C))) ; stage 3
  //     ((mutex_region 'X) (write/32 'E (read/32 'D))) ; stage 4
  //     ((mutex_region 'X) (write/32 'F (read/32 'E))) ; stage 5
  //     ((mutex_region 'X) (output/32   (read/32 'F))) ; stage 6
  // )
  // NB can combine registers 'D, 'E', & 'F.
  XLS_ASSERT_OK_AND_ASSIGN((const std::array<RegisterData, kRegCnt> registers),
                           CreateStraightShot<kRegCnt>(p.get()));
  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 1);
  csg.MarkMutuallyExclusive(3, 4);
  csg.MarkMutuallyExclusive(3, 5);
  csg.MarkMutuallyExclusive(3, 6);
  csg.MarkMutuallyExclusive(4, 5);
  csg.MarkMutuallyExclusive(4, 6);
  csg.MarkMutuallyExclusive(5, 6);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(), UnorderedElementsAre(ElementsAre(
                                 Reg("reg_A"), Reg("reg_B"), Reg("reg_C"),
                                 Reg("reg_D"), Reg("reg_E"), Reg("reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split, UnorderedElementsAre(ElementsAre(
                           Reg("reg_D"), Reg("reg_E"), Reg("reg_F"))));
  };
  CheckAllPermutations<kRegCnt>(registers, check_mutex);
}

TEST_F(RegisterChainingAnalysisTest, LoopbackMutex) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((mutex_region 'X) (write/32 'A (read/32 'F)))       ; stage 0
  //     ((mutex_region 'X) (write/32 'B (read/32 'A)))       ; stage 1
  //     ((mutex_region 'X) (write/32 'C (read/32 'B)))       ; stage 2
  //     ((mutex_region 'X) (write/32 'D (read/32 'C)))       ; stage 3
  //     ((no_mutex)        (write/32 'E (read/32 'D)))       ; stage 4
  //     ((no_mutex)        (write/32 'F (+ 1 (read/32 'E)))) ; stage 5
  // )
  // NB can combine registers 'A, 'B' & 'C
  XLS_ASSERT_OK_AND_ASSIGN((const std::array<RegisterData, kRegCnt> registers),
                           CreateLoopback<kRegCnt>(p.get()));
  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 1);
  csg.MarkMutuallyExclusive(0, 1);
  csg.MarkMutuallyExclusive(0, 2);
  csg.MarkMutuallyExclusive(0, 3);
  csg.MarkMutuallyExclusive(1, 2);
  csg.MarkMutuallyExclusive(1, 3);
  csg.MarkMutuallyExclusive(2, 3);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(), UnorderedElementsAre(ElementsAre(
                                 Reg("reg_F"), Reg("reg_A"), Reg("reg_B"),
                                 Reg("reg_C"), Reg("reg_D"), Reg("reg_E"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split, UnorderedElementsAre(ElementsAre(
                           Reg("reg_A"), Reg("reg_B"), Reg("reg_C"))));
  };
  CheckAllPermutations<kRegCnt>(registers, check_mutex);
}

TEST_F(RegisterChainingAnalysisTest, MutexTouchesStart) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((mutex_region 'X) (write/32 'A (input/32  ))) ; stage 0
  //     ((mutex_region 'X) (write/32 'B (read/32 'A))) ; stage 1
  //     ((mutex_region 'X) (write/32 'C (read/32 'B))) ; stage 2
  //     ((mutex_region 'X) (write/32 'D (read/32 'C))) ; stage 3
  //     ((no_mutex)        (write/32 'E (read/32 'D))) ; stage 4
  //     ((no_mutex)        (write/32 'F (read/32 'E))) ; stage 5
  //     ((no_mutex)        (output/32   (read/32 'F))) ; stage 6
  // )
  // NB can registers 'A, 'B' & 'C
  XLS_ASSERT_OK_AND_ASSIGN((const std::array<RegisterData, kRegCnt> datas),
                           CreateStraightShot<kRegCnt>(p.get()));
  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 1);
  csg.MarkMutuallyExclusive(0, 1);
  csg.MarkMutuallyExclusive(0, 2);
  csg.MarkMutuallyExclusive(0, 3);
  csg.MarkMutuallyExclusive(1, 2);
  csg.MarkMutuallyExclusive(1, 3);
  csg.MarkMutuallyExclusive(2, 3);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(), UnorderedElementsAre(ElementsAre(
                                 Reg("reg_A"), Reg("reg_B"), Reg("reg_C"),
                                 Reg("reg_D"), Reg("reg_E"), Reg("reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split, UnorderedElementsAre(ElementsAre(
                           Reg("reg_A"), Reg("reg_B"), Reg("reg_C"))));
  };
  CheckAllPermutations<kRegCnt>(datas, check_mutex);
}

TEST_F(RegisterChainingAnalysisTest, OverlappingFullyUsedMutexRegions) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ; stage 0
  //     ((mutex_range 0 6) (write/32 'early_A (input/32  )))
  //     ; stage 1
  //     ((mutex_range 0 6) (write/32 'early_B (read/32 'early_A)))
  //     ; stage 2
  //     ((mutex_range 0 6) (write/32 'early_C (read/32 'early_B)))
  //     ; stage 3
  //     ((mutex_range 0 9) (write/32 'early_D (read/32 'early_C))
  //                        (write/32 'late_A  (input/32)))
  //     ; stage 4
  //     ((mutex_range 0 9) (write/32 'early_E (read/32 'early_D))
  //                        (write/32 'late_B  (read/32 'late_A )))
  //     ; stage 5
  //     ((mutex_range 0 9) (write/32 'early_F (read/32 'early_E))
  //                        (write/32 'late_C  (read/32 'late_B )))
  //     ; stage 6
  //     ((mutex_range 0 9) (output/32         (read/32 'early_F))
  //                        (write/32 'late_D  (read/32 'late_C )))
  //     ; stage 7
  //     ((mutex_range 3 9) (write/32 'late_E  (read/32 'late_D)))
  //     ; stage 8
  //     ((mutex_range 3 9) (write/32 'late_F  (read/32 'late_E)))
  //     ; stage 9
  //     ((mutex_range 3 9) (output/32         (read/32 'late_F)))
  // )
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      (const std::array<RegisterData, kRegCnt> datas_early),
      CreateStraightShot<kRegCnt>(bb, /*start_stage=*/0, /*prefix=*/"early_"));
  XLS_ASSERT_OK_AND_ASSIGN(
      (const std::array<RegisterData, kRegCnt> datas_late),
      CreateStraightShot<kRegCnt>(bb, /*start_stage=*/3, /*prefix=*/"late_"));
  XLS_ASSERT_OK(bb.Build().status());
  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 3 + 1);
  MarkAllPairs(csg, 0, 6);
  MarkAllPairs(csg, 3, 9);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt * 2>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                UnorderedElementsAre(
                    ElementsAre(Reg("early_reg_A"), Reg("early_reg_B"),
                                Reg("early_reg_C"), Reg("early_reg_D"),
                                Reg("early_reg_E"), Reg("early_reg_F")),
                    ElementsAre(Reg("late_reg_A"), Reg("late_reg_B"),
                                Reg("late_reg_C"), Reg("late_reg_D"),
                                Reg("late_reg_E"), Reg("late_reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split, UnorderedElementsAre(
                           ElementsAre(Reg("early_reg_A"), Reg("early_reg_B"),
                                       Reg("early_reg_C"), Reg("early_reg_D"),
                                       Reg("early_reg_E"), Reg("early_reg_F")),
                           ElementsAre(Reg("late_reg_A"), Reg("late_reg_B"),
                                       Reg("late_reg_C"), Reg("late_reg_D"),
                                       Reg("late_reg_E"), Reg("late_reg_F"))));
  };
  std::array<RegisterData, kRegCnt * 2> datas;
  absl::c_copy(datas_late, absl::c_copy(datas_early, datas.begin()));
  check_mutex(datas);
}

TEST_F(RegisterChainingAnalysisTest, OverlappingMutexRegions) {
  static constexpr int64_t kRegCnt = 6;
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ; stage 0
  //     ((mutex_range 0 4) (write/32 'early_A (input/32  )))
  //     ; stage 1
  //     ((mutex_range 0 4) (write/32 'early_B (read/32 'early_A)))
  //     ; stage 2
  //     ((mutex_range 0 4) (write/32 'early_C (read/32 'early_B)))
  //     ; stage 3
  //     ((mutex_range 0 6) (write/32 'early_D (read/32 'early_C))
  //                        (write/32 'late_A  (input/32)))
  //     ; stage 4
  //     ((mutex_range 3 6) (write/32 'early_E (read/32 'early_D))
  //                        (write/32 'late_B  (read/32 'late_A )))
  //     ; stage 5
  //     ((mutex_range 3 6) (write/32 'early_F (read/32 'early_E))
  //                        (write/32 'late_C  (read/32 'late_B )))
  //     ; stage 6
  //     ((mutex_range 3 6) (output/32         (read/32 'early_F))
  //                        (write/32 'late_D  (read/32 'late_C )))
  //     ; stage 7
  //     ((no_mutex)        (write/32 'late_E  (read/32 'late_D)))
  //     ; stage 8
  //     ((no_mutex)        (write/32 'late_F  (read/32 'late_E)))
  //     ; stage 9
  //     ((no_mutex)        (output/32         (read/32 'late_F)))
  // )
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      (const std::array<RegisterData, kRegCnt> datas_early),
      CreateStraightShot<kRegCnt>(bb, /*start_stage=*/0, /*prefix=*/"early_"));
  XLS_ASSERT_OK_AND_ASSIGN(
      (const std::array<RegisterData, kRegCnt> datas_late),
      CreateStraightShot<kRegCnt>(bb, /*start_stage=*/3, /*prefix=*/"late_"));
  XLS_ASSERT_OK(bb.Build().status());
  CodegenPassOptions opt;
  ConcurrentStageGroups csg(kRegCnt + 3 + 1);
  MarkAllPairs(csg, 0, 4);
  MarkAllPairs(csg, 3, 6);

  auto check_mutex = [&](const std::array<RegisterData, kRegCnt * 2>& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(rc.chains(),
                UnorderedElementsAre(
                    ElementsAre(Reg("early_reg_A"), Reg("early_reg_B"),
                                Reg("early_reg_C"), Reg("early_reg_D"),
                                Reg("early_reg_E"), Reg("early_reg_F")),
                    ElementsAre(Reg("late_reg_A"), Reg("late_reg_B"),
                                Reg("late_reg_C"), Reg("late_reg_D"),
                                Reg("late_reg_E"), Reg("late_reg_F"))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split,
                UnorderedElementsAre(
                    ElementsAre(Reg("late_reg_A"), Reg("late_reg_B"),
                                Reg("late_reg_C")),
                    ElementsAre(Reg("early_reg_A"), Reg("early_reg_B"),
                                Reg("early_reg_C"), Reg("early_reg_D")),
                    ElementsAre(Reg("early_reg_E"), Reg("early_reg_F"))));
  };
  std::array<RegisterData, kRegCnt * 2> datas;
  absl::c_copy(datas_late, absl::c_copy(datas_early, datas.begin()));
  check_mutex(datas);
}

TEST_F(RegisterChainingAnalysisTest, MutexLoopbackContinues) {
  auto p = CreatePackage();
  // pipeline:
  // (
  //     ((mutex_region 'X) (write/32 'A (read/32 'C_back))) ; stage 0
  //     ((mutex_region 'X) (write/32 'B (read/32 'A)))      ; stage 1
  //     ((mutex_region 'X) (let ((b_val (read/32 'B)))      ; stage 2
  //                             (write/32 'C_back b_val)
  //                             (write/32 'C_cont b_val))
  //     ((mutex_region 'X) (write/32 'D (read/32 'C_cont))) ; stage 3
  //     ((mutex_region 'X) (write/32 'E (read/32 'D)))      ; stage 4
  //     ((no_mutex)        (write/32 'F (read/32 'E)))      ; stage 5
  //     ((no_mutex)        (output/32   (read/32 'F)))      ; stage 6
  // )
  // Needs to be split into 2 sections C_back, A, B & C_cont
  BlockBuilder bb(TestName(), p.get());
  auto* t32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_a, bb.block()->AddRegister("A", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_b, bb.block()->AddRegister("B", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_c_cont,
                           bb.block()->AddRegister("C_cont", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_c_back,
                           bb.block()->AddRegister("C_back", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_d, bb.block()->AddRegister("D", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_e, bb.block()->AddRegister("E", t32));
  XLS_ASSERT_OK_AND_ASSIGN(auto reg_f, bb.block()->AddRegister("F", t32));

  BValue a_read = bb.RegisterRead(reg_a, SourceInfo(), reg_a->name());
  BValue b_read = bb.RegisterRead(reg_b, SourceInfo(), reg_b->name());
  BValue c_cont_read =
      bb.RegisterRead(reg_c_cont, SourceInfo(), reg_c_cont->name());
  BValue c_back_read =
      bb.RegisterRead(reg_c_back, SourceInfo(), reg_c_back->name());
  BValue d_read = bb.RegisterRead(reg_d, SourceInfo(), reg_d->name());
  BValue e_read = bb.RegisterRead(reg_e, SourceInfo(), reg_e->name());
  BValue f_read = bb.RegisterRead(reg_f, SourceInfo(), reg_f->name());

  BValue a_write = bb.RegisterWrite(
      reg_a, c_back_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_a->name() + "_write");
  BValue b_write = bb.RegisterWrite(reg_b, a_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_b->name() + "_write");
  BValue c_cont_write = bb.RegisterWrite(
      reg_c_cont, b_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_c_cont->name() + "_write");
  BValue c_back_write = bb.RegisterWrite(
      reg_c_back, b_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_c_back->name() + "_write");
  BValue d_write = bb.RegisterWrite(
      reg_d, c_cont_read, /*load_enable=*/std::nullopt, /*reset=*/std::nullopt,
      SourceInfo(), reg_d->name() + "_write");
  BValue e_write = bb.RegisterWrite(reg_e, d_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_e->name() + "_write");
  BValue f_write = bb.RegisterWrite(reg_f, e_read, /*load_enable=*/std::nullopt,
                                    /*reset=*/std::nullopt, SourceInfo(),
                                    reg_f->name() + "_write");
  bb.OutputPort("OUT", f_read);

  RegisterData a_data{
      .reg = reg_a,
      .read = a_read.node()->As<RegisterRead>(),
      .read_stage = 1,
      .write = a_write.node()->As<RegisterWrite>(),
      .write_stage = 0,
  };
  RegisterData b_data{
      .reg = reg_b,
      .read = b_read.node()->As<RegisterRead>(),
      .read_stage = 2,
      .write = b_write.node()->As<RegisterWrite>(),
      .write_stage = 1,
  };
  RegisterData c_cont_data{
      .reg = reg_c_cont,
      .read = c_cont_read.node()->As<RegisterRead>(),
      .read_stage = 3,
      .write = c_cont_write.node()->As<RegisterWrite>(),
      .write_stage = 2,
  };
  RegisterData c_back_data{
      .reg = reg_c_back,
      .read = c_back_read.node()->As<RegisterRead>(),
      .read_stage = 0,
      .write = c_back_write.node()->As<RegisterWrite>(),
      .write_stage = 2,
  };
  RegisterData d_data{
      .reg = reg_d,
      .read = d_read.node()->As<RegisterRead>(),
      .read_stage = 4,
      .write = d_write.node()->As<RegisterWrite>(),
      .write_stage = 3,
  };
  RegisterData e_data{
      .reg = reg_e,
      .read = e_read.node()->As<RegisterRead>(),
      .read_stage = 5,
      .write = e_write.node()->As<RegisterWrite>(),
      .write_stage = 4,
  };
  RegisterData f_data{
      .reg = reg_f,
      .read = f_read.node()->As<RegisterRead>(),
      .read_stage = 6,
      .write = f_write.node()->As<RegisterWrite>(),
      .write_stage = 5,
  };

  CodegenPassOptions opt;
  ConcurrentStageGroups csg(7);
  MarkAllPairs(csg, 0, 4);

  auto check_mutex = [&](const auto& data) {
    RegisterChains rc;
    for (const auto& v : data) {
      rc.InsertAndReduce(v);
    }
    EXPECT_THAT(
        rc.chains(),
        // Loopback needs to be first.
        // Cases where a value extends beyond the loopback can't be on the same
        // chain as the loopback.
        AnyOf(UnorderedElementsAre(
                  ElementsAre(c_back_data, a_data, b_data),
                  ElementsAre(c_cont_data, d_data, e_data, f_data)),
              UnorderedElementsAre(
                  ElementsAre(c_back_data, a_data),
                  ElementsAre(b_data, c_cont_data, d_data, e_data, f_data)),
              UnorderedElementsAre(ElementsAre(c_back_data),
                                   ElementsAre(a_data, b_data, c_cont_data,
                                               d_data, e_data, f_data))));
    XLS_ASSERT_OK_AND_ASSIGN(auto split, rc.SplitBetweenMutexRegions(csg, opt));
    EXPECT_THAT(split,
                AnyOf(UnorderedElementsAre(
                          ElementsAre(Reg("C_back"), Reg("A"), Reg("B")),
                          ElementsAre(Reg("C_cont"), Reg("D"))),
                      UnorderedElementsAre(
                          ElementsAre(Reg("C_back"), Reg("A")),
                          ElementsAre(Reg("B"), Reg("C_cont"), Reg("D"))),
                      // NB There is sort of an implicit [Reg("C_back")] here
                      // but since this list doesn't include singletons it
                      // doesn't appear.
                      UnorderedElementsAre(ElementsAre(
                          Reg("A"), Reg("B"), Reg("C_cont"), Reg("D")))));
  };
  CheckAllPermutations<7>(
      {a_data, b_data, c_cont_data, c_back_data, d_data, e_data, f_data},
      check_mutex);
}

}  // namespace
}  // namespace xls::verilog
