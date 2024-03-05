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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
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
    } while ((!HasFailure() || i < 10) &&
             absl::c_next_permutation(arr, cmp_register_data));
    RecordProperty("permutations", i);
  }
};

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

}  // namespace
}  // namespace xls::verilog
