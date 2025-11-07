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

#include "xls/ir/block_testutils.h"

#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::z3::IsProvenTrue;
using ::xls::solvers::z3::TryProveEquivalence;

class UnrollBlockTest : public IrTestBase {};

TEST_F(UnrollBlockTest, BasicBlockEquivalence) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  BlockBuilder bb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      bb.block()->AddRegister("foo", p->GetBitsType(4), Value(UBits(2, 4))));
  auto read = bb.RegisterRead(reg);
  auto reset = bb.ResetPort("reset", ResetBehavior{.active_low = false});
  bb.RegisterWrite(reg, bb.Add(read, bb.InputPort("foo_ch", p->GetBitsType(4))),
                   /*load_enable=*/bb.InputPort("foo_le", p->GetBitsType(1)),
                   /*reset=*/reset);
  bb.OutputPort("foobar", read);
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  auto ch_1 = fb.Param("foo_ch_act0_input", p->GetBitsType(4));
  auto le_1 = fb.Param("foo_le_act0_input", p->GetBitsType(1));
  auto reset_1 = fb.Param("reset_act0_input", p->GetBitsType(1));
  auto ch_2 = fb.Param("foo_ch_act1_input", p->GetBitsType(4));
  auto le_2 = fb.Param("foo_le_act1_input", p->GetBitsType(1));
  auto reset_2 = fb.Param("reset_act1_input", p->GetBitsType(1));
  fb.Param("foo_ch_act2_input", p->GetBitsType(4));
  fb.Param("foo_le_act2_input", p->GetBitsType(1));
  fb.Param("reset_act2_input", p->GetBitsType(1));
  auto reset_val = fb.Literal(UBits(2, 4));
  auto read_1 = reset_val;
  auto read_2 = fb.Select(
      reset_1, /*on_true=*/reset_val,
      /*on_false=*/
      fb.Select(le_1, /*on_true=*/fb.Add(read_1, ch_1), /*on_false=*/read_1));
  auto read_3 = fb.Select(
      reset_2, /*on_true=*/reset_val,
      /*on_false=*/
      fb.Select(le_2, /*on_true=*/fb.Add(read_2, ch_2), /*on_false=*/read_2));

  fb.Tuple({fb.Tuple({read_1}), fb.Tuple({read_2}), fb.Tuple({read_3})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollBlockToFunction(blk, 3, /*include_state=*/false,
                            /*zero_invalid_outputs=*/false));

  RecordProperty("func", f->DumpIr());
  RecordProperty("block", blk->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  auto equiv = TryProveEquivalence(f, converted);
  EXPECT_THAT(equiv, IsOkAndHolds(IsProvenTrue()));
  if (equiv.ok() && std::holds_alternative<solvers::z3::ProvenFalse>(*equiv)) {
    RecordProperty("counterexample",
                   DumpWithNodeValues(
                       converted, std::get<solvers::z3::ProvenFalse>(*equiv))
                       .value_or(converted->DumpIr()));
    RecordProperty(
        "counterexample_func",
        DumpWithNodeValues(f, std::get<solvers::z3::ProvenFalse>(*equiv))
            .value_or(f->DumpIr()));
  }
}

TEST_F(UnrollBlockTest, BasicBlockEquivalenceWithState) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  BlockBuilder bb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg,
      bb.block()->AddRegister("foo", p->GetBitsType(6), Value(UBits(2, 6))));
  auto read = bb.BitSlice(bb.RegisterRead(reg), 0, 4);
  auto reset = bb.ResetPort("reset", ResetBehavior{.active_low = false});
  bb.RegisterWrite(
      reg,
      bb.ZeroExtend(bb.Add(read, bb.InputPort("foo_ch", p->GetBitsType(4))), 6),
      /*load_enable=*/bb.InputPort("foo_le", p->GetBitsType(1)),
      /*reset=*/reset);
  bb.OutputPort("foobar", read);
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());

  auto ch_1 = fb.Param("foo_ch_act0_input", p->GetBitsType(4));
  auto le_1 = fb.Param("foo_le_act0_input", p->GetBitsType(1));
  auto reset_1 = fb.Param("reset_act0_input", p->GetBitsType(1));
  auto ch_2 = fb.Param("foo_ch_act1_input", p->GetBitsType(4));
  auto le_2 = fb.Param("foo_le_act1_input", p->GetBitsType(1));
  auto reset_2 = fb.Param("reset_act1_input", p->GetBitsType(1));
  auto ch_3 = fb.Param("foo_ch_act2_input", p->GetBitsType(4));
  auto le_3 = fb.Param("foo_le_act2_input", p->GetBitsType(1));
  auto reset_3 = fb.Param("reset_act2_input", p->GetBitsType(1));
  auto reset_val = fb.Literal(UBits(2, 4));
  auto read_1 = reset_val;
  auto read_2 = fb.Select(
      reset_1, /*on_true=*/reset_val,
      /*on_false=*/
      fb.Select(le_1, /*on_true=*/fb.Add(read_1, ch_1), /*on_false=*/read_1));
  auto read_3 = fb.Select(
      reset_2, /*on_true=*/reset_val,
      /*on_false=*/
      fb.Select(le_2, /*on_true=*/fb.Add(read_2, ch_2), /*on_false=*/read_2));
  auto read_4 = fb.Select(
      reset_3, /*on_true=*/reset_val,
      /*on_false=*/
      fb.Select(le_3, /*on_true=*/fb.Add(read_3, ch_3), /*on_false=*/read_3));

  auto act_result = [&](BValue port, BValue reg) {
    return fb.Tuple({fb.Tuple({port}), fb.Tuple({fb.ZeroExtend(reg, 6)})});
  };

  fb.Tuple({act_result(read_1, read_2), act_result(read_2, read_3),
            act_result(read_3, read_4)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollBlockToFunction(blk, 3, /*include_state=*/true,
                            /*zero_invalid_outputs=*/false));

  RecordProperty("func", f->DumpIr());
  RecordProperty("block", blk->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  auto equiv = TryProveEquivalence(f, converted);
  EXPECT_THAT(equiv, IsOkAndHolds(IsProvenTrue()));
  if (equiv.ok() && std::holds_alternative<solvers::z3::ProvenFalse>(*equiv)) {
    RecordProperty("counterexample",
                   DumpWithNodeValues(
                       converted, std::get<solvers::z3::ProvenFalse>(*equiv))
                       .value_or(converted->DumpIr()));
    RecordProperty(
        "counterexample_func",
        DumpWithNodeValues(f, std::get<solvers::z3::ProvenFalse>(*equiv))
            .value_or(f->DumpIr()));
  }
}

}  // namespace
}  // namespace xls
