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

#include "xls/passes/bit_count_query_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value_builder.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls {
namespace {

using internal::LeadingBits;
using LeadingBitsTree = LeafTypeTree<LeadingBits>;

class BitCountQueryEngineTest : public IrTestBase {};

TEST_F(BitCountQueryEngineTest, DataFlow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue val = fb.Tuple(
      {fb.Literal(ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0b1100, 4)),
                                       ValueBuilder::Bits(UBits(0b0011, 4))})),
       fb.Literal(
           ValueBuilder::Tuple({ValueBuilder::Bits(UBits(0b1000, 4)),
                                ValueBuilder::Bits(UBits(0b0111, 4))}))});
  BValue ab = fb.TupleIndex(val, 0);
  BValue cd = fb.TupleIndex(val, 1);
  BValue a = fb.TupleIndex(ab, 0);
  BValue b = fb.TupleIndex(ab, 1);
  BValue c = fb.TupleIndex(cd, 0);
  BValue d = fb.TupleIndex(cd, 1);
  BValue sel = fb.Select(fb.Param("foo", p->GetBitsType(1)), {d, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  BitCountQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f));
  EXPECT_EQ(qe.KnownLeadingZeros(a.node()), 0);
  EXPECT_EQ(qe.KnownLeadingZeros(b.node()), 2);
  EXPECT_EQ(qe.KnownLeadingZeros(c.node()), 0);
  EXPECT_EQ(qe.KnownLeadingZeros(d.node()), 1);
  EXPECT_EQ(qe.KnownLeadingZeros(sel.node()), 1);

  EXPECT_EQ(qe.KnownLeadingOnes(a.node()), 2);
  EXPECT_EQ(qe.KnownLeadingOnes(b.node()), 0);
  EXPECT_EQ(qe.KnownLeadingOnes(c.node()), 1);
  EXPECT_EQ(qe.KnownLeadingOnes(d.node()), 0);
  EXPECT_EQ(qe.KnownLeadingOnes(sel.node()), 0);

  EXPECT_EQ(qe.KnownLeadingSignBits(a.node()), 2);
  EXPECT_EQ(qe.KnownLeadingSignBits(b.node()), 2);
  EXPECT_EQ(qe.KnownLeadingSignBits(c.node()), 1);
  EXPECT_EQ(qe.KnownLeadingSignBits(d.node()), 1);
  EXPECT_EQ(qe.KnownLeadingSignBits(sel.node()), 1);
}

TEST_F(BitCountQueryEngineTest, SignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  fb.SignExtend(input, 16);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(input.node()), 1);
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 9);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 11);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 11);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 11);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 11);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 11);
  }
}

TEST_F(BitCountQueryEngineTest, ZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  fb.ZeroExtend(input, 16);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(input.node()), 1);
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 8);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 8);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 8);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(f->return_value()), 11);
    EXPECT_EQ(qe.KnownLeadingOnes(f->return_value()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(f->return_value()), 11);
  }
}

TEST_F(BitCountQueryEngineTest, Concat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  BValue zeros = fb.Concat({fb.Literal(UBits(0, 8)), input});
  BValue ones = fb.Concat({fb.Literal(SBits(-1, 8)), input});
  BValue half_zero = fb.Concat({fb.Literal(UBits(0b00001111, 8)), input});
  BValue half_ones = fb.Concat({fb.Literal(UBits(0b11110000, 8)), input});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingZeros(ones.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingZeros(half_ones.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(zeros.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(half_zero.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(half_ones.node()), 4);

    EXPECT_EQ(qe.KnownLeadingSignBits(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_ones.node()), 4);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingZeros(ones.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingZeros(half_ones.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(zeros.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(half_zero.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(half_ones.node()), 4);

    EXPECT_EQ(qe.KnownLeadingSignBits(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_ones.node()), 4);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingZeros(ones.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingZeros(half_ones.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(zeros.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(ones.node()), 11);
    EXPECT_EQ(qe.KnownLeadingOnes(half_zero.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(half_ones.node()), 4);

    EXPECT_EQ(qe.KnownLeadingSignBits(zeros.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(ones.node()), 11);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_ones.node()), 4);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(zeros.node()), 11);
    EXPECT_EQ(qe.KnownLeadingZeros(ones.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingZeros(half_ones.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(zeros.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(half_zero.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(half_ones.node()), 4);

    EXPECT_EQ(qe.KnownLeadingSignBits(zeros.node()), 11);
    EXPECT_EQ(qe.KnownLeadingSignBits(ones.node()), 8);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_zero.node()), 4);
    EXPECT_EQ(qe.KnownLeadingSignBits(half_ones.node()), 4);
  }
}

TEST_F(BitCountQueryEngineTest, ConcatZeroLen) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  BValue res = fb.Concat({fb.Literal(UBits(0, 0)), input});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 3);
  }
}

TEST_F(BitCountQueryEngineTest, ConcatAllZeroLen) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue res = fb.Concat({fb.Literal(UBits(0, 0)), fb.Literal(UBits(0, 0)),
                          fb.Literal(UBits(0, 0))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BitCountQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(f));
  EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 0);
  EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
  EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 0);
}

TEST_F(BitCountQueryEngineTest, ConcatDups) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lead = fb.Param("lead", p->GetBitsType(3));
  BValue input = fb.Param("foo", p->GetBitsType(8));
  BValue res = fb.Concat({lead, lead, lead, input, lead});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lead.node(), LeadingBitsTree::CreateSingleElementTree(
                         p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 9);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    XLS_ASSERT_OK(qe.AddGiven(
        lead.node(), LeadingBitsTree::CreateSingleElementTree(
                         p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(res.node()), 12);
    EXPECT_EQ(qe.KnownLeadingOnes(res.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(res.node()), 12);
  }
}

TEST_F(BitCountQueryEngineTest, Neg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  BValue neg = fb.Negate(input);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(input.node()), 1);
    EXPECT_EQ(qe.KnownLeadingZeros(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(neg.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(neg.node()), 2);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(neg.node()), 0);
    // Since we know this is a positive number (leading zeros) the negative has
    // the same number of sign bits at minimum (it might have 1 more but thats
    // only at specific values).
    EXPECT_EQ(qe.KnownLeadingSignBits(neg.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(neg.node()), 2);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(neg.node()), 8);
    EXPECT_EQ(qe.KnownLeadingOnes(neg.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(neg.node()), 8);
  }
}

void NegFuzzZ3(int8_t sign_cnt) {
  auto p = std::make_unique<Package>("neg_fuzz");
  FunctionBuilder model_fb("model", p.get());
  FunctionBuilder fb("test", p.get());
  auto value_with_leading_sign_bits = [&](FunctionBuilder& fb,
                                          std::string_view name_base,
                                          int8_t leading_sign_bits) -> BValue {
    if (leading_sign_bits == 0) {
      return fb.Param(name_base, p->GetBitsType(16));
    } else {
      BValue sign = fb.SignExtend(
          fb.Param(absl::StrCat(name_base, "_sign"), p->GetBitsType(1)),
          leading_sign_bits);
      if (leading_sign_bits != 16) {
        return fb.Concat(
            {sign, fb.Param(absl::StrCat(name_base, "_base"),
                            p->GetBitsType(16 - leading_sign_bits))});
      } else {
        return sign;
      }
    }
  };
  auto mk_fuzz = [&](FunctionBuilder& fb) -> BValue {
    BValue lhs = value_with_leading_sign_bits(fb, "lhs", sign_cnt);
    BValue neg = fb.Negate(lhs);
    return neg;
  };
  BValue model_neg = mk_fuzz(model_fb);
  XLS_ASSERT_OK_AND_ASSIGN(Function * model, model_fb.Build());

  BitCountQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(model));
  int64_t known_signs = qe.KnownLeadingSignBits(model_neg.node()).value_or(0);

  // Make the test fn.
  BValue neg = mk_fuzz(fb);
  // Extract the leading bits.

  // Extract the top sign bits.
  BValue leading_bits =
      fb.BitSlice(neg, neg.BitCountOrDie() - known_signs, known_signs);
  // Validate that all the bits are the same value.
  BValue all_same =
      fb.Or(fb.AndReduce(leading_bits), fb.AndReduce(fb.Not(leading_bits)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * check, fb.Build());

  testing::Test::RecordProperty("ir", check->DumpIr());
  testing::Test::RecordProperty("known_sign", known_signs);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto res, solvers::z3::TryProve(check, all_same.node(),
                                      solvers::z3::Predicate::NotEqualToZero(),
                                      /*rlimit=*/0));
  EXPECT_THAT(res, solvers::z3::IsProvenTrue());
}

FUZZ_TEST(BitCountQueryEngineFuzzTest, NegFuzzZ3)
    .WithDomains(fuzztest::InRange(0, 16));

TEST_F(BitCountQueryEngineTest, Not) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(8));
  BValue not_inst = fb.Not(input);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(input.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(input.node()), 1);
    EXPECT_EQ(qe.KnownLeadingZeros(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(not_inst.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(not_inst.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(not_inst.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(not_inst.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(8), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(not_inst.node()), 3);
    EXPECT_EQ(qe.KnownLeadingOnes(not_inst.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(not_inst.node()), 3);
  }
}

TEST_F(BitCountQueryEngineTest, BitSlice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(16));
  BValue start = fb.BitSlice(input, 0, 10);
  BValue mid = fb.BitSlice(input, 3, 10);
  BValue end = fb.BitSlice(input, 6, 10);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(start.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(mid.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(end.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(start.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(mid.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(end.node()), 6);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownZeros(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(mid.node()), 3);
    EXPECT_EQ(qe.KnownLeadingZeros(end.node()), 6);

    EXPECT_EQ(qe.KnownLeadingOnes(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(start.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(mid.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(end.node()), 6);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownOnes(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(mid.node()), 3);
    EXPECT_EQ(qe.KnownLeadingOnes(end.node()), 6);

    EXPECT_EQ(qe.KnownLeadingSignBits(start.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(mid.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(end.node()), 6);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(12))));
    EXPECT_EQ(qe.KnownLeadingZeros(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(end.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(start.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(mid.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(end.node()), 0);

    // SSSS_SSSS_SSSS_XXXX
    //        ^^_^^^^_^^^^
    EXPECT_EQ(qe.KnownLeadingSignBits(start.node()), 6);
    // SSSS_SSSS_SSSS_XXXX
    //    ^_^^^^_^^^^_^
    EXPECT_EQ(qe.KnownLeadingSignBits(mid.node()), 9);
    // SSSS_SSSS_SSSS_XXXX
    // ^^^^_^^^^_^^
    EXPECT_EQ(qe.KnownLeadingSignBits(end.node()), 10);
  }
}

TEST_F(BitCountQueryEngineTest, Shll) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(16));
  BValue lit = fb.Shll(input, fb.Literal(UBits(3, 5)));
  BValue clr = fb.Shll(input, fb.Literal(UBits(16, 5)));
  BValue dyn = fb.Shll(input, fb.Param("bar", p->GetBitsType(5)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(clr.node()), 16);
    EXPECT_EQ(qe.KnownLeadingZeros(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(clr.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 1);
    EXPECT_EQ(qe.KnownLeadingSignBits(clr.node()), 16);
    EXPECT_EQ(qe.KnownLeadingSignBits(dyn.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownZeros(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownOnes(3))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownZeros(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownOnes(6))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
  }
}

TEST_F(BitCountQueryEngineTest, Shrl) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(16));
  BValue lit = fb.Shrl(input, fb.Literal(UBits(3, 5)));
  BValue clr = fb.Shrl(input, fb.Literal(UBits(16, 5)));
  BValue dyn = fb.Shrl(input, fb.Param("dyn", p->GetBitsType(5)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingZeros(clr.node()), 16);
    EXPECT_EQ(qe.KnownLeadingZeros(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(clr.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingSignBits(clr.node()), 16);
    EXPECT_EQ(qe.KnownLeadingSignBits(dyn.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 3);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 3);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 11);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 11);
  }
}

TEST_F(BitCountQueryEngineTest, Shra) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue input = fb.Param("foo", p->GetBitsType(16));
  BValue lit = fb.Shra(input, fb.Literal(UBits(3, 5)));
  BValue clr = fb.Shra(input, fb.Literal(UBits(16, 5)));
  BValue dyn = fb.Shra(input, fb.Param("dyn", p->GetBitsType(5)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(clr.node()), 0);
    EXPECT_EQ(qe.KnownLeadingZeros(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(clr.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(dyn.node()), 0);

    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 4);
    EXPECT_EQ(qe.KnownLeadingSignBits(clr.node()), 16);
    EXPECT_EQ(qe.KnownLeadingSignBits(dyn.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::SignValues(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 11);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 11);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 11);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        input.node(), LeadingBitsTree::CreateSingleElementTree(
                          p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(lit.node()), 11);
    EXPECT_EQ(qe.KnownLeadingOnes(lit.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(lit.node()), 11);
  }
}

TEST_F(BitCountQueryEngineTest, Add) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("lhs", p->GetBitsType(16));
  BValue rhs = fb.Param("rhs", p->GetBitsType(16));
  BValue add = fb.Add(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(add.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::SignValues(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::SignValues(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(add.node()), 7);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(add.node()), 7);
    EXPECT_EQ(qe.KnownLeadingOnes(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(add.node()), 7);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(add.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(add.node()), 7);
    EXPECT_EQ(qe.KnownLeadingSignBits(add.node()), 7);
  }
}

template <typename Fn>
  requires(std::is_invocable_r_v<BValue, Fn, FunctionBuilder&, BValue, BValue>)
void Z3BinOpFuzzTest(Fn make_combine, int8_t sign_lhs, int8_t sign_rhs) {
  auto p = std::make_unique<Package>("bin_fuzz");
  FunctionBuilder model_fb("model", p.get());
  FunctionBuilder fb("test", p.get());
  auto value_with_leading_sign_bits = [&](FunctionBuilder& fb,
                                          std::string_view name_base,
                                          int8_t leading_sign_bits) -> BValue {
    if (leading_sign_bits == 0) {
      return fb.Param(name_base, p->GetBitsType(16));
    } else {
      BValue sign = fb.SignExtend(
          fb.Param(absl::StrCat(name_base, "_sign"), p->GetBitsType(1)),
          leading_sign_bits);
      if (leading_sign_bits != 16) {
        return fb.Concat(
            {sign, fb.Param(absl::StrCat(name_base, "_base"),
                            p->GetBitsType(16 - leading_sign_bits))});
      } else {
        return sign;
      }
    }
  };
  auto mk_fuzz = [&](FunctionBuilder& fb) -> BValue {
    BValue lhs = value_with_leading_sign_bits(fb, "lhs", sign_lhs);
    BValue rhs = value_with_leading_sign_bits(fb, "rhs", sign_rhs);
    BValue binop = make_combine(fb, lhs, rhs);
    return binop;
  };
  BValue model_op = mk_fuzz(model_fb);
  XLS_ASSERT_OK_AND_ASSIGN(Function * model, model_fb.Build());

  BitCountQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(model));
  int64_t known_signs = qe.KnownLeadingSignBits(model_op.node()).value_or(0);

  // Make the test fn.
  BValue binop = mk_fuzz(fb);
  // Extract the leading bits.

  // Extract the top sign bits.
  BValue leading_bits =
      fb.BitSlice(binop, binop.BitCountOrDie() - known_signs, known_signs);
  // Validate that all the bits are the same value.
  BValue all_same =
      fb.Or(fb.AndReduce(leading_bits), fb.AndReduce(fb.Not(leading_bits)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * check, fb.Build());

  testing::Test::RecordProperty("ir", check->DumpIr());
  testing::Test::RecordProperty("known_sign", known_signs);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto res, solvers::z3::TryProve(check, all_same.node(),
                                      solvers::z3::Predicate::NotEqualToZero(),
                                      /*rlimit=*/0));
  EXPECT_THAT(res, solvers::z3::IsProvenTrue());
}

void AddFuzzZ3(int8_t lhs, int8_t rhs) {
  Z3BinOpFuzzTest([](FunctionBuilder& fb, BValue lhs,
                     BValue rhs) { return fb.Add(lhs, rhs); },
                  lhs, rhs);
}
FUZZ_TEST(BitCountQueryEngineFuzzTest, AddFuzzZ3)
    .WithDomains(fuzztest::InRange(0, 16), fuzztest::InRange(0, 16));

TEST_F(BitCountQueryEngineTest, Sub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("lhs", p->GetBitsType(16));
  BValue rhs = fb.Param("rhs", p->GetBitsType(16));
  BValue sub = fb.Subtract(lhs, rhs);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    EXPECT_EQ(qe.KnownLeadingZeros(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(sub.node()), 1);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::SignValues(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::SignValues(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(sub.node()), 7);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownZeros(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(sub.node()), 7);
  }
  {
    BitCountQueryEngine qe;
    XLS_ASSERT_OK(qe.Populate(f));
    XLS_ASSERT_OK(qe.AddGiven(
        lhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    XLS_ASSERT_OK(qe.AddGiven(
        rhs.node(), LeadingBitsTree::CreateSingleElementTree(
                        p->GetBitsType(16), LeadingBits::KnownOnes(8))));
    EXPECT_EQ(qe.KnownLeadingZeros(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingOnes(sub.node()), 0);
    EXPECT_EQ(qe.KnownLeadingSignBits(sub.node()), 7);
  }
}

void SubFuzzZ3(int8_t lhs, int8_t rhs) {
  Z3BinOpFuzzTest([](FunctionBuilder& fb, BValue lhs,
                     BValue rhs) { return fb.Subtract(lhs, rhs); },
                  lhs, rhs);
}
FUZZ_TEST(BitCountQueryEngineFuzzTest, SubFuzzZ3)
    .WithDomains(fuzztest::InRange(0, 16), fuzztest::InRange(0, 16));

}  // namespace
}  // namespace xls
