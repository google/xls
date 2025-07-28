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

#include "xls/passes/reassociation_pass.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/passes/arith_simplification_pass.h"
#include "xls/passes/basic_simplification_pass.h"
#include "xls/passes/bit_slice_simplification_pass.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/narrowing_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_test_helpers.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

constexpr absl::Duration kProverTimeout = absl::Seconds(20);

using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;

using ::xls::solvers::z3::ScopedVerifyEquivalence;

class ReassociationPassTest : public IrTestBase {
 protected:
  ReassociationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    OptimizationCompoundPass pass("TestPass", TestName());
    bool run_result = false;
    pass.Add<RecordIfPassChanged<ReassociationPass>>(&run_result);
    pass.Add<DeadCodeEliminationPass>();
    XLS_ASSIGN_OR_RETURN(
        bool compound_result,
        pass.Run(p, OptimizationPassOptions(), &results, context));
    return compound_result && run_result;
  }
  absl::StatusOr<bool> RunWithNarrowing(Package* p) {
    PassResults results;
    OptimizationContext context;
    OptimizationCompoundPass pass("TestPass", TestName());
    bool run_result = false;
    // NB Several tests were written for a version of reassociation that did
    // significantly more narrowing in the pass then the current one does. Run
    // narrowing and a bunch of other passes to clean up.
    pass.Add<RecordIfPassChanged<ReassociationPass>>(&run_result);
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<ConstantFoldingPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<NarrowingPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<BitSliceSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<ArithSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<BasicSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<BitSliceSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    XLS_ASSIGN_OR_RETURN(
        bool compound_result,
        pass.Run(p, OptimizationPassOptions(), &results, context));
    return compound_result && run_result;
  }
  absl::StatusOr<bool> RunWithConstProp(Package* p) {
    PassResults results;
    OptimizationContext context;
    OptimizationCompoundPass pass("TestPass", TestName());
    bool run_result = false;
    pass.Add<RecordIfPassChanged<ReassociationPass>>(&run_result);
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<ConstantFoldingPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<ArithSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    pass.Add<BasicSimplificationPass>();
    pass.Add<DeadCodeEliminationPass>();
    XLS_ASSIGN_OR_RETURN(
        bool compound_result,
        pass.Run(p, OptimizationPassOptions(), &results, context));
    return compound_result && run_result;
  }

  int64_t MaxAddDepth(Function* f) { return MaxOpDepth({Op::kAdd}, f); }
  int64_t MaxOpDepth(absl::Span<Op const> ops, Function* f) {
    absl::flat_hash_map<Node*, int64_t> cnts;
    cnts.reserve(f->node_count());
    for (Node* n : TopoSort(f)) {
      if (n->operand_count() == 0) {
        cnts[n] = 0;
        continue;
      }
      absl::Span<Node* const> operands = n->operands();
      int64_t max_pred = cnts[*absl::c_max_element(
          operands, [&](Node* l, Node* r) { return cnts[l] < cnts[r]; })];
      if (n->OpIn(ops)) {
        cnts[n] = 1 + max_pred;
      } else {
        cnts[n] = max_pred;
      }
    }
    return cnts[f->return_value()];
  }
};

TEST_F(ReassociationPassTest, SingleAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32), fb.Param("b", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, TwoAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32), fb.Add(fb.Param("b", u32), fb.Param("c", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ZeroResult) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  // ((x + y) + z) - (x + (y + z))
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue z = fb.Param("z", u32);
  fb.Subtract(fb.Add(fb.Add(x, y), z), fb.Add(x, fb.Add(y, z)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 32)));
}

TEST_F(ReassociationPassTest, ZeroResultDeeper) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  // (((x + y) + z) + w) - (w + (x + (y + z)))
  BValue w = fb.Param("w", u32);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue z = fb.Param("z", u32);
  fb.Subtract(fb.Add(fb.Add(fb.Add(x, y), z), w),
              fb.Add(w, fb.Add(x, fb.Add(y, z))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 32)));
}

TEST_F(ReassociationPassTest, ZeroPlusConstant) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  // (((x + y) + z) + 12) - (4 + (x + (y + z)))
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue z = fb.Param("z", u32);
  fb.Subtract(fb.Add(fb.Add(fb.Add(x, y), z), fb.Literal(UBits(12, 32))),
              fb.Add(fb.Literal(UBits(4, 32)), fb.Add(x, fb.Add(y, z))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Sub(m::Literal(), m::Literal()));
}

TEST_F(ReassociationPassTest, ChainOfConstants) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  std::vector<BValue> res;
  res.reserve(20);
  res.push_back(fb.Param("f", u32));
  for (int64_t i = 1; i < 21; ++i) {
    res.push_back(fb.Add(res.back(), fb.Literal(UBits(i, 32))));
  }
  fb.Tuple(absl::MakeConstSpan(res).subspan(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value()->operands(),
      testing::AllOf(testing::SizeIs(20),
                     testing::Each(m::Add(m::Param("f"), m::Literal()))));
}

// TODO(allight): This is a slightly suboptimal reassociation. Ideally we would
// use the same node for both the res[19] element and in the full addition tree.
// We do not do so however.
TEST_F(ReassociationPassTest, ChainOfConstantsAndOthers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  std::vector<BValue> res;
  res.reserve(20);
  res.push_back(fb.Param("a", u32));
  for (int64_t i = 1; i < 21; ++i) {
    res.push_back(fb.Add(res.back(), fb.Literal(UBits(i, 32))));
  }
  BValue last = res.back();
  for (int64_t i = 0; i < 5; ++i) {
    last =
        fb.Add(last, fb.Param(absl::StrFormat("lst%d", i), p->GetBitsType(32)));
  }
  res.push_back(last);
  fb.Tuple(absl::MakeConstSpan(res).subspan(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  // Check the first 20 tuple elements.
  EXPECT_THAT(f->return_value()->operands().subspan(0, 20),
              testing::Each(m::Add(m::Param("a"), m::Literal())));
  // TODO(allight): Recognizing this would be good but is hard to do without
  // breaking other pieces.
  // EXPECT_THAT(f->return_value()->operands()[19]->users(),
  //             testing::SizeIs(2));
  EXPECT_EQ(MaxAddDepth(f), 3);
}

TEST_F(ReassociationPassTest, ChainOfUnsignedConstantsExt) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  std::vector<BValue> res;
  res.reserve(20);
  res.push_back(fb.Param("f", p->GetBitsType(8)));
  for (int64_t i = 1; i < 21; ++i) {
    res.push_back(
        fb.Add(fb.ZeroExtend(res.back(), 8 + i), fb.Literal(UBits(i, 8 + i))));
  }
  fb.Tuple(absl::MakeConstSpan(res).subspan(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value()->operands(),
              testing::AllOf(testing::SizeIs(20),
                             testing::Each(m::Add(m::ZeroExt(m::Param("f")),
                                                  m::Literal()))));
}

TEST_F(ReassociationPassTest, ChainOfSubs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  std::vector<BValue> res;
  res.reserve(20);
  res.push_back(fb.Param("f", u32));
  for (int64_t i = 1; i < 21; ++i) {
    res.push_back(i % 2 == 1
                      ? fb.Add(res.back(), fb.Literal(UBits(i, 32)))
                      : fb.Subtract(res.back(), fb.Literal(UBits(i, 32))));
  }
  fb.Tuple(absl::MakeConstSpan(res).subspan(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value()->operands(),
      testing::AllOf(testing::SizeIs(20),
                     testing::Each(m::Add(m::Param("f"), m::Literal()))));
}

TEST_F(ReassociationPassTest, ChainOfThreeAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32),
         fb.Add(fb.Param("b", u32),
                fb.Add(fb.Param("c", u32), fb.Param("d", u32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Add(m::Param("a"), m::Param("b")),
                                        m::Add(m::Param("c"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, PairSameAheadAndIsolateLoner) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  fb.Add(a, fb.Add(a, fb.Add(a, fb.Add(b, b))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                            m::Add(m::Param("b"), m::Param("b"))),
                     m::Param("a")));
}

TEST_F(ReassociationPassTest, PairSameAheadAndPairDifferentWithInterval) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  fb.Add(a, fb.Add(a, fb.Add(a, fb.Add(b, fb.Add(b, c)))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                            m::Add(m::Param("b"), m::Param("b"))),
                     m::Add(m::Param("a"), m::Param("c"))));
}

TEST_F(ReassociationPassTest, PairSameAheadTwice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  fb.Add(a, fb.Add(a, fb.Add(a, fb.Add(b, fb.Add(b, fb.Add(c, c))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                    m::Add(m::Param("b"), m::Param("b"))),
             m::Add(m::Add(m::Param("c"), m::Param("c")), m::Param("a"))));
}

TEST_F(ReassociationPassTest, PairSameAheadTwiceAndPairDifferentWithInterval) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue d = fb.Param("d", u32);
  fb.Add(a,
         fb.Add(a, fb.Add(a, fb.Add(b, fb.Add(b, fb.Add(c, fb.Add(c, d)))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                            m::Add(m::Param("b"), m::Param("b"))),
                     m::Add(m::Add(m::Param("c"), m::Param("c")),
                            m::Add(m::Param("a"), m::Param("d")))));
}

TEST_F(ReassociationPassTest, PairSameTwiceAndPairAhead) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  fb.Add(a, fb.Add(a, fb.Add(a, fb.Add(a, fb.Add(b, fb.Add(c, c))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                    m::Add(m::Param("a"), m::Param("a"))),
             m::Add(m::Add(m::Param("c"), m::Param("c")), m::Param("b"))));
}

TEST_F(ReassociationPassTest, PairScrambledDifferentWithIntervalTwice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue d = fb.Param("d", u32);
  fb.Add(
      d,
      fb.Add(
          a,
          fb.Add(
              d,
              fb.Add(
                  d,
                  fb.Add(
                      b,
                      fb.Add(
                          c,
                          fb.Add(
                              a,
                              fb.Add(c,
                                     fb.Add(b, fb.Add(a, fb.Add(b, c)))))))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Add(m::Add(m::Param("a"), m::Param("a")),
                                   m::Add(m::Param("b"), m::Param("b"))),
                            m::Add(m::Add(m::Param("a"), m::Param("b")),
                                   m::Add(m::Param("c"), m::Param("c")))),
                     m::Add(m::Add(m::Param("d"), m::Param("d")),
                            m::Add(m::Param("c"), m::Param("d")))));
}

TEST_F(ReassociationPassTest, PairSameWithZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.ZeroExtend(fb.Param("a", p->GetBitsType(14)), 32);
  BValue b = fb.ZeroExtend(fb.Param("b", p->GetBitsType(13)), 32);
  BValue c = fb.ZeroExtend(fb.Param("c", p->GetBitsType(12)), 32);
  BValue d = fb.ZeroExtend(fb.Param("d", p->GetBitsType(11)), 32);
  BValue e = fb.ZeroExtend(fb.Param("e", p->GetBitsType(10)), 32);
  fb.Add(c,
         fb.Add(b, fb.Add(a, fb.Add(d, fb.Add(d, fb.Add(e, fb.Add(e, d)))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ZeroExt(m::Add(
          m::ZeroExt(m::Add(m::ZeroExt(m::Add(m::ZeroExt(m::Param("e")),
                                              m::ZeroExt(m::Param("e")))),
                            m::ZeroExt(m::Add(m::ZeroExt(m::Param("d")),
                                              m::ZeroExt(m::Param("d")))))),
          m::ZeroExt(m::Add(m::ZeroExt(m::Add(m::ZeroExt(m::Param("d")),
                                              m::ZeroExt(m::Param("c")))),
                            m::ZeroExt(m::Add(m::ZeroExt(m::Param("b")),
                                              m::ZeroExt(m::Param("a")))))))));
}

TEST_F(ReassociationPassTest, NearToFinalBitWidth) {
  auto p = CreatePackage();
  // We need to make the addition unbalanced so reassociation actually touches
  // it instead of just waiting for narrowing to get ahold of it.
  FunctionBuilder fb(TestName(), p.get());
  BValue v3 = fb.Param("v3", p->GetBitsType(3));
  BValue v3_ext7 = fb.SignExtend(v3, 7);
  BValue v6 = fb.Param("v6", p->GetBitsType(6));
  BValue v6_ext7 = fb.SignExtend(v6, 7);
  BValue add_7 = fb.Add(v6_ext7, v3_ext7);
  BValue v16 = fb.Param("v16", p->GetBitsType(16));
  BValue ext_9 = fb.SignExtend(add_7, 9);
  BValue v8 = fb.Param("v8", p->GetBitsType(8));
  BValue v8_ext9 = fb.SignExtend(v8, 9);
  BValue add_9 = fb.Add(ext_9, v8_ext9);
  BValue ext_17_add_9 = fb.SignExtend(add_9, 17);
  BValue ext_17 = fb.SignExtend(v16, 17);
  fb.Add(ext_17, ext_17_add_9);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  // NB Previous versions of reassociation would perform some narrowing here.
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Add(m::SignExt(m::Add(m::SignExt(m::Param("v3")),
                               m::SignExt(m::Param("v6")))),
             m::Add(m::SignExt(m::Param("v8")), m::SignExt(m::Param("v16")))));
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthUnsignedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.ZeroExtend(fb.Param("a", u32), 35),
         fb.ZeroExtend(
             fb.Add(fb.ZeroExtend(fb.Param("b", u32), 34),
                    fb.ZeroExtend(fb.Add(fb.ZeroExtend(fb.Param("c", u32), 33),
                                         fb.ZeroExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(RunWithNarrowing(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(AllOf(m::Add(_, _), m::Type("bits[34]"))));
  EXPECT_THAT(f->nodes(), testing::Contains(m::Add(_, _)).Times(3));
  EXPECT_THAT(f->nodes(),
              Contains(AllOf(m::Type("bits[33]"), m::Add(_, _))).Times(2));
  EXPECT_THAT(f->nodes(),
              Contains(AllOf(m::Type("bits[34]"), m::Add(_, _))).Times(1));
  EXPECT_EQ(MaxAddDepth(f), 2);
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthSignedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.SignExtend(fb.Param("a", u32), 35),
         fb.SignExtend(
             fb.Add(fb.SignExtend(fb.Param("b", u32), 34),
                    fb.SignExtend(fb.Add(fb.SignExtend(fb.Param("c", u32), 33),
                                         fb.SignExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(RunWithNarrowing(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // NB Order is sensitive to the exact node-id assignment.
      m::SignExt(AllOf(m::Add(m::SignExt(m::Add(m::SignExt(m::Param("a")),
                                                m::SignExt(m::Param("b")))),
                              m::SignExt(m::Add(m::SignExt(m::Param("c")),
                                                m::SignExt(m::Param("d"))))),
                       m::Type("bits[34]"))));
  EXPECT_EQ(MaxAddDepth(f), 2);
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthMixedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.SignExtend(fb.Param("a", u32), 35),
         fb.SignExtend(
             fb.Add(fb.ZeroExtend(fb.Param("b", u32), 34),
                    fb.ZeroExtend(fb.Add(fb.SignExtend(fb.Param("c", u32), 33),
                                         fb.SignExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ChainOfThreeUMulRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.UMul(fb.Param("c", u32), fb.Param("d", u32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::UMul(m::UMul(m::Param("a"), m::Param("b")),
                      m::UMul(m::Param("c"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, ChainOfFourUMulRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.UMul(fb.Param("c", u32),
                          fb.UMul(fb.Param("d", u32), fb.Param("e", u32)))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::UMul(m::UMul(m::UMul(m::Param("a"), m::Param("b")),
                              m::UMul(m::Param("c"), m::Param("d"))),
                      m::Param("e")));
}

TEST_F(ReassociationPassTest, ChainOfMixedOperations) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.Add(fb.Param("c", u32),
                         fb.Add(fb.Param("d", u32), fb.Param("e", u32)))));
  XLS_ASSERT_OK(fb.Build().status());
}

TEST_F(ReassociationPassTest, ChainOfThreeAddsLeft) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
                fb.Param("c", u32)),
         fb.Param("d", u32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Add(m::Param("a"), m::Param("b")),
                                        m::Add(m::Param("c"), m::Param("d"))));
  EXPECT_EQ(MaxAddDepth(f), 2);
}

TEST_F(ReassociationPassTest, DeepChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue lhs = fb.Param("p0", u32);
  for (int64_t i = 1; i < 41; ++i) {
    lhs = fb.Add(lhs, fb.Param(absl::StrFormat("p%d", i), u32));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxAddDepth(f), 6);
}

TEST_F(ReassociationPassTest, DeepChainOfFullWidthUnsignedAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue lhs = fb.Param("p0", u8);
  for (int64_t i = 1; i < 10; ++i) {
    lhs = fb.Add(fb.ZeroExtend(lhs, 8 + i),
                 fb.ZeroExtend(fb.Param(absl::StrFormat("p%d", i), u8), 8 + i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[17]")));

  ScopedVerifyEquivalence stays_equivalent(f);
  // NB We could perform significant narrowing ourselves (by noticing that
  // arguments are both ExtendOp) but things are simpler if we just let the
  // normal narrowing pass do that for us.
  ASSERT_THAT(RunWithNarrowing(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(AllOf(m::Add(), m::Type("bits[12]"))));
  EXPECT_EQ(MaxAddDepth(f), 4);
}

TEST_F(ReassociationPassTest, DeepChainOfFullWidthSignedAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue lhs = fb.Param("p0", u8);
  for (int64_t i = 1; i < 10; ++i) {
    lhs = fb.Add(fb.SignExtend(lhs, 8 + i),
                 fb.SignExtend(fb.Param(absl::StrFormat("p%d", i), u8), 8 + i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[17]")));

  ScopedVerifyEquivalence stays_equivalent(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(RunWithNarrowing(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Add(), m::Type("bits[12]"))));
}

TEST_F(ReassociationPassTest, BalancedTreeOfThreeAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)), fb.Param("c", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, BalancedTreeOfFourAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Add(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, BalancedTreeOfFiveAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Add(fb.Param("c", u32),
                fb.Add(fb.Param("d", u32), fb.Param("e", u32))));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ReassociateMultipleLiterals) {
  // Multiple Literals should be reassociated to the right even if the tree is
  // balanced.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Literal(UBits(42, 32))),
         fb.Add(fb.Add(fb.Literal(UBits(123, 32)), fb.Param("b", u32)),
                fb.Add(fb.Param("c", p->GetBitsType(32)),
                       fb.Literal(UBits(10, 32)))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Param("a"), m::Param("b")),
                     m::Add(m::Param("c"),
                            m::Add(m::Add(m::Literal(42), m::Literal(123)),
                                   m::Literal(10)))));
}

TEST_F(ReassociationPassTest, SingleLiteralNoReassociate) {
  // If there is a single literal in the expression and the tree is balanced
  // then no reassociation should happen.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Literal(UBits(42, 32))),
         fb.Add(fb.Param("b", u32), fb.Param("c", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SingleSubtract) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Param("a", u32), fb.Param("b", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SingleSubtractOfLiteral) {
  // Allow basic-simp etc to handle this case.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Param("a", u32), fb.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, TreeOfSubtracts) {
  // Add and sub are equal cost so don't bother rewriting this even though we
  // could turn it into `(a + d) - (b + c)`.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Subtract(fb.Param("a", u32), fb.Param("b", u32)),
              fb.Subtract(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, AddOfSubtracts) {
  // Actual delay models see no difference between add and sub so no need to
  // make any changes since this is already balanced.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Subtract(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Subtract(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, AddOfManySubtracts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  // ((a - b) + (c - (d + e + f + g)))
  fb.Add(fb.Subtract(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Subtract(
             fb.Param("c", u32),
             fb.Add(fb.Param("d", u32),
                    fb.Add(fb.Param("e", u32),
                           fb.Add(fb.Param("f", u32), fb.Param("g", u32))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Sub(m::Add(m::Sub(m::Param("a"), m::Param("b")),
                    m::Sub(m::Param("c"), m::Param("d"))),
             m::Add(m::Add(m::Param("e"), m::Param("f")), m::Param("g"))));
}

TEST_F(ReassociationPassTest, SubtractOfAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
              fb.Add(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubtractOfAddsWithLiterals) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Add(fb.Param("a", u32), fb.Literal(Value(UBits(100, 32)))),
              fb.Add(fb.Literal(Value(UBits(42, 32))), fb.Param("b", u32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(m::Param("a"), m::Param("b")),
                                        m::Literal(UBits(58, 32))));
}

TEST_F(ReassociationPassTest, SubOfSub) {
  // Actual delay models see no difference between add and sub so no need to
  // make any changes since this is already balanced.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Subtract(fb.Param("x", p->GetBitsType(32)),
              fb.Subtract(fb.Param("y", p->GetBitsType(32)),
                          fb.Param("z", p->GetBitsType(32))));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, BalanceEarlyUse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Param("lhs0", p->GetBitsType(4));
  for (int64_t i = 1; i < 5; ++i) {
    lhs = fb.Add(fb.Param(absl::StrFormat("lhs%d", i), p->GetBitsType(4)), lhs);
  }
  BValue rhs = fb.Param("rhs0", p->GetBitsType(4));
  for (int64_t i = 1; i < 5; ++i) {
    rhs = fb.Add(fb.Param(absl::StrFormat("rhs%d", i), p->GetBitsType(4)), rhs);
  }
  fb.Tuple({lhs, fb.Add(lhs, rhs)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  auto lhs_match = m::Add(m::Add(m::Add(m::Param("lhs0"), m::Param("lhs1")),
                                 m::Add(m::Param("lhs2"), m::Param("lhs3"))),
                          m::Param("lhs4"));
  auto rhs_match = m::Add(m::Add(m::Add(m::Param("rhs0"), m::Param("rhs1")),
                                 m::Add(m::Param("rhs2"), m::Param("rhs3"))),
                          m::Add(m::Param("rhs4"), lhs_match));
  EXPECT_THAT(f->return_value(), m::Tuple(lhs_match, rhs_match));
  EXPECT_THAT(f->nodes(), Contains(m::Add()).Times(9));
}

TEST_F(ReassociationPassTest, BalanceEarlyUseIsNotDuplicated) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Add(fb.Add(fb.Param("w", p->GetBitsType(4)),
                             fb.Param("x", p->GetBitsType(4))),
                      fb.Add(fb.Param("y", p->GetBitsType(4)),
                             fb.Param("z", p->GetBitsType(4))));
  BValue rhs = fb.Param("a", p->GetBitsType(4));
  for (int64_t i = 0; i < 4; ++i) {
    rhs = fb.Add(fb.Param(absl::StrFormat("rhs%d", i), p->GetBitsType(4)), rhs);
  }
  fb.Tuple({lhs, fb.Add(lhs, rhs)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Tuple(lhs.node(),
               m::Add(m::Add(m::Add(lhs.node(), m::Param("a")),
                             m::Add(m::Param("rhs0"), m::Param("rhs1"))),
                      m::Add(m::Param("rhs2"), m::Param("rhs3")))));
}

TEST_F(ReassociationPassTest, DoubleUseBalanceDoesntChange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue lhs = fb.Add(fb.Add(fb.Param("w", p->GetBitsType(4)),
                             fb.Param("x", p->GetBitsType(4))),
                      fb.Add(fb.Param("y", p->GetBitsType(4)),
                             fb.Param("z", p->GetBitsType(4))));
  BValue rhs = fb.Param("a", p->GetBitsType(4));
  for (int64_t i = 0; i < 4; ++i) {
    rhs = fb.Add(fb.Param(absl::StrFormat("rhs%d", i), p->GetBitsType(4)), rhs);
  }
  fb.Tuple({lhs, fb.Add(lhs, rhs)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  // Use the normal pass to get it into its reassociated state.
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Make sure running the pass again doesn't cause more changes.
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubUnderflowZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Literal(UBits(1023, 10)),
         fb.ZeroExtend(fb.Subtract(fb.Param("x", p->GetBitsType(5)),
                                   fb.Literal(UBits(1, 5))),
                       10));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubUnderflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue small = fb.Param("small", p->GetBitsType(2));
  BValue ext = fb.ZeroExtend(small, 16);
  BValue sub = fb.Subtract(fb.Literal(UBits(1, 16)), ext);
  BValue sub2 = fb.Subtract(fb.Param("big", p->GetBitsType(16)), sub);
  fb.Subtract(sub2, fb.Param("second", p->GetBitsType(16)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxOpDepth({Op::kAdd, Op::kSub}, f), 2);
  // NB This ends up as
  //
  // (sum small (neg 1) big (neg second))
  //
  // There are several equivalent ways to represent this depending on the exact
  // way we pair up leaf values. The choice is consistent but based on the exact
  // order node ids are assigned.
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(), m::Sub()));
}

TEST_F(ReassociationPassTest, SubUnderflowZeroExtend2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue small = fb.ZeroExtend(fb.Param("small", p->GetBitsType(2)), 16);
  BValue mid = fb.ZeroExtend(fb.Param("mid", p->GetBitsType(4)), 16);
  BValue sub = fb.Subtract(fb.Literal(UBits(0, 16)), small);
  BValue sub2 = fb.Subtract(fb.Param("big", p->GetBitsType(16)), sub);
  fb.Subtract(mid, sub2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxOpDepth({Op::kAdd, Op::kSub}, f), 2);
  EXPECT_THAT(f->return_value(), m::Add(m::Sub(), m::Sub()));
}

TEST_F(ReassociationPassTest, SubUnderflowZeroExtend3) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue small = fb.Param("small", p->GetBitsType(2));
  BValue zero_ext_17 = fb.ZeroExtend(small, 17);
  BValue slice_8 = fb.BitSlice(zero_ext_17, 0, 8);
  BValue slice_16 = fb.BitSlice(zero_ext_17, 0, 16);
  BValue top_bits = fb.Literal(UBits(0, 8));
  BValue ext_16 = fb.Concat({top_bits, slice_8});
  BValue add_v = fb.Add(slice_16, fb.Literal(UBits(0, 16)));
  fb.Subtract(ext_16, add_v);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubUnderflowZeroExtend4) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue small = fb.Param("inp", p->GetBitsType(16));
  BValue lit_sub =
      fb.Subtract(fb.Literal(UBits(16, 10)), fb.Literal(UBits(0, 10)));
  fb.Subtract(small, fb.ZeroExtend(lit_sub, 16));
  XLS_ASSERT_OK(fb.Build().status());
  // Already balanced
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubUnderflowZeroExtend5) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue small = fb.Param("inp", p->GetBitsType(16));
  BValue lit_sub =
      fb.Subtract(fb.Literal(UBits(16, 10)), fb.Literal(UBits(0, 10)));
  fb.Subtract(
      fb.Add(fb.Param("param", p->GetBitsType(16)), fb.Subtract(small, small)),
      fb.ZeroExtend(lit_sub, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Sub(m::Param("param"), m::ZeroExt(lit_sub.node())));
}

TEST_F(ReassociationPassTest, ConcatMultipleValues) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto zero_extend = [&](BValue v, int64_t width) {
    if (!v.valid()) {
      return BValue();
    }
    EXPECT_LT(v.BitCountOrDie() + 5, width) << "bad test setup!";
    width -= v.BitCountOrDie();
    std::vector<BValue> vals;
    // 5 literals implement the extend.
    for (int64_t i = 0; i < 4; ++i) {
      vals.push_back(fb.Literal(UBits(0, 1)));
      --width;
    }
    vals.push_back(fb.Literal(UBits(0, width)));
    vals.push_back(v);
    return fb.Concat(vals);
  };
  Type* u8 = p->GetBitsType(8);
  fb.Add(
      zero_extend(fb.Param("a", u8), 50),
      zero_extend(fb.Add(zero_extend(fb.Param("b", u8), 40),
                         zero_extend(fb.Add(zero_extend(fb.Param("c", u8), 30),
                                            zero_extend(fb.Param("d", u8), 30)),
                                     40)),
                  50));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxAddDepth(f), 2);
}

TEST_F(ReassociationPassTest, RecognizeNonZeroExtConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto zero_extend = [&](BValue v, int64_t width) {
    if (!v.valid()) {
      return BValue();
    }
    EXPECT_LT(v.BitCountOrDie() + 5, width) << "bad test setup!";
    width -= v.BitCountOrDie();
    std::vector<BValue> vals;
    // 5 literals implement the extend.
    for (int64_t i = 0; i < 4; ++i) {
      vals.push_back(fb.Literal(UBits(0, 1)));
      --width;
    }
    vals.push_back(fb.Literal(UBits(0, width)));
    vals.push_back(v);
    return fb.Concat(vals);
  };
  Type* u8 = p->GetBitsType(8);
  fb.Add(zero_extend(fb.Param("a", u8), 50),
         fb.Concat({
             fb.Literal(UBits(0, 4)),
             fb.Param("foo", p->GetBitsType(1)),
             fb.Literal(UBits(0, 5)),
             fb.Add(zero_extend(fb.Param("b", u8), 40),
                    zero_extend(fb.Add(zero_extend(fb.Param("c", u8), 30),
                                       zero_extend(fb.Param("d", u8), 30)),
                                40)),
         }));
  XLS_ASSERT_OK(fb.Build().status());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, NegativeZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Literal(SBits(-2, 32)),
         fb.Add(fb.Literal(UBits(1, 32)),
                fb.ZeroExtend(fb.Negate(fb.Param("foo", p->GetBitsType(16))),
                              32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::ZeroExt(m::Neg(m::Param("foo"))),
                                        m::Literal(SBits(-1, 32))));
}

TEST_F(ReassociationPassTest, NegativeExtendNoCancel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue foo = fb.Param("foo", p->GetBitsType(8));
  fb.Add(fb.ZeroExtend(foo, 16), fb.ZeroExtend(fb.Negate(foo), 16));
  XLS_ASSERT_OK(fb.Build().status());

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  // TODO(allight): We could reduce this to 0x100 constant. We'd need to add
  // support for synthetic constants into reassociation.
}

TEST_F(ReassociationPassTest, NegativeOneBitLiteralZeroExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Add(fb.Param("foo", p->GetBitsType(16)), fb.Literal(SBits(-1, 16))),
         fb.ZeroExtend(fb.Negate(fb.Literal(UBits(1, 1))), 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("foo"));
}

TEST_F(ReassociationPassTest, NegativeZeroExtend2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Add(fb.Add(fb.Param("foo", p->GetBitsType(16)),
                       fb.Literal(SBits(-1, 16))),
                fb.Literal(SBits(-1, 16))),
         fb.ZeroExtend(fb.Negate(fb.Param("bar", p->GetBitsType(4))), 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxAddDepth(f), 2);
  EXPECT_THAT(
      f->return_value(),
      m::Add(m::Add(m::Param("foo"), m::ZeroExt(m::Neg(m::Param("bar")))),
             m::Literal(UBits(0xfffe, 16))));
}

TEST_F(ReassociationPassTest, SubZeroExtLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue v1 = fb.Literal(UBits(0, 2));
  BValue cc_1 = fb.Concat({v1, fb.Param("foo", p->GetBitsType(13))});
  BValue v2 = fb.Literal(UBits(1, 1));
  BValue v3 = fb.Literal(UBits(0, 15));
  BValue cc_2 = fb.Concat({cc_1, v2});
  BValue cc_3 = fb.Concat({v3, v2});
  BValue sub = fb.Subtract(cc_2, cc_3);
  fb.Add(sub, cc_3);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
}

TEST_F(ReassociationPassTest, SMulWithNeg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("a", p->GetBitsType(8));
  BValue neg_sq = fb.Negate(fb.SMul(param, param));
  BValue mul2 = fb.SMul(neg_sq, param);
  fb.SMul(mul2, param);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Neg(m::SMul(m::SMul(m::Param(), m::Param()),
                             m::SMul(m::Param(), m::Param()))));
}

TEST_F(ReassociationPassTest, UMulWithNeg) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("a", p->GetBitsType(8));
  BValue neg_sq = fb.Negate(fb.UMul(param, param));
  BValue mul2 = fb.UMul(neg_sq, param);
  fb.UMul(mul2, param);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Neg(m::UMul(m::UMul(m::Param(), m::Param()),
                             m::UMul(m::Param(), m::Param()))));
}

TEST_F(ReassociationPassTest, UMulWithNeg2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("a", p->GetBitsType(8));
  BValue mul1 = fb.UMul(param, param);
  BValue mul2 = fb.Negate(fb.UMul(mul1, param));
  fb.UMul(mul2, param);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithConstProp(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Neg(m::UMul(m::UMul(m::Param(), m::Param()),
                             m::UMul(m::Param(), m::Param()))));
}

// This is a deceptively tricky test.
//
// Basically what it is doing is making an ir that is:
// (add-chain-no-overflow) + (add-chain-overflow)
//
// Since the last add in the (add-chain-no-overflow) has the same bit width as
// the overflow one though the 2 operands of that add can be placed in the
// reassociation set of the overflow nodes.
//
// When doing reassociation the second-to-last node in the no-overflow chain is
// examined and its seen that its user does not use it as an identity node and
// so no reassociation happens. Later the final bottom add is examined and it
// sees all the overflow nodes plus the last no-overflow node and one of the
// adds in its reassociation leaf set and reassociates. If we left it there
// however we'd end up with a suboptimal result since the tree rooted at the
// second-to-last no-overflow add node can also be flattened. We need to check
// that this flattening is actually performed.
TEST_F(ReassociationPassTest, MultipleOverflowTypes) {
  static constexpr int64_t kBaseBitCount = 2;
  static constexpr int64_t kChainDepth = 4;
  static constexpr int64_t kOverflowBitCount = kBaseBitCount + kChainDepth - 1;
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue non_overflow =
      fb.Param("no_overflow", p->GetBitsType(kBaseBitCount - 1));
  for (int64_t i = 0; i < kChainDepth; ++i) {
    non_overflow =
        fb.Add(fb.ZeroExtend(non_overflow, kBaseBitCount + i),
               fb.ZeroExtend(fb.Param(absl::StrCat("non_overflow_", i),
                                      p->GetBitsType(kBaseBitCount - 1 + i)),
                             kBaseBitCount + i));
  }
  BValue overflow = fb.Param("overflow", p->GetBitsType(kOverflowBitCount));
  for (int64_t i = 0; i < kChainDepth; ++i) {
    overflow = fb.Add(overflow, fb.Param(absl::StrCat("overflow_", i),
                                         p->GetBitsType(kOverflowBitCount)));
  }
  fb.Add(non_overflow, overflow);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // NB Z3 seems to have a surprisingly hard time proving this. Possibly because
  // a value switches sides on the add tree breaking pattern matching proofs?
  ScopedVerifyEquivalence stays_equivalent(f);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(RunWithNarrowing(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxAddDepth(f), 4);
}

TEST_F(ReassociationPassTest, MultipleUsersReassociatesEarly) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // With a total of 8 leafs could do it in 3 depth but since we don't want to
  // duplicate the add tree its in 4.
  BValue shared = fb.Param("shared_0", p->GetBitsType(4));
  for (int64_t i = 1; i <= 6; ++i) {
    shared =
        fb.Add(shared, fb.Param(absl::StrCat("shared_", i), p->GetBitsType(4)));
  }
  fb.Tuple({
      fb.Add(shared, fb.Param("unshared_1", p->GetBitsType(4))),
      fb.Add(shared, fb.Param("unshared_2", p->GetBitsType(4))),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(MaxAddDepth(f), 4);
  // The tuple's adds shouldn't have changed.
  EXPECT_EQ(f->return_value()->As<Tuple>()->operand(0)->operand(0),
            f->return_value()->As<Tuple>()->operand(1)->operand(0));
  EXPECT_THAT(f->return_value()->As<Tuple>()->operand(0)->operand(0),
              m::Add(m::Add(m::Add(), m::Add()), m::Add(m::Add(), m::Param())));
}

TEST_F(ReassociationPassTest, MulOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue inp = fb.Param("inp", p->GetBitsType(4));
  BValue inp_sq = fb.UMul(inp, inp);
  BValue one_or_zero = fb.ZeroExtend(fb.Param("bit", p->GetBitsType(1)), 4);
  BValue mul_3 = fb.UMul(inp_sq, one_or_zero);
  fb.UMul(mul_3, fb.Literal(UBits(0b1001, 4)), 8);

  XLS_ASSERT_OK(fb.Build().status());
  // Since the mul_3 can't be reassociated through this is already balanced.
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, MulOverflow2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue inp = fb.Param("inp", p->GetBitsType(4));
  BValue inp_sq = fb.UMul(inp, inp);
  BValue inp_cube = fb.UMul(inp_sq, inp);
  BValue inp_quad = fb.UMul(inp_cube, inp);
  BValue inp_quint = fb.UMul(inp_quad, inp);
  BValue one_or_zero = fb.ZeroExtend(fb.Param("bit", p->GetBitsType(1)), 4);
  BValue mul_3 = fb.UMul(inp_quint, one_or_zero);
  fb.UMul(mul_3, fb.Literal(UBits(0b1001, 4)), 8);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // No reassociation through the extending mul.
  EXPECT_THAT(
      f->return_value(),
      AllOf(m::Type("bits[8]"),
            m::UMul(AllOf(m::UMul(), m::Type("bits[4]")), m::Literal())));
  EXPECT_EQ(MaxOpDepth({Op::kUMul}, f), 4);
}

TEST_F(ReassociationPassTest, SignExtendOfNegationOverflows) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x1 = fb.Param("x1", p->GetBitsType(4));
  BValue x2 = fb.Param("x2", p->GetBitsType(8));
  BValue neg_x1 = fb.Negate(x1);
  BValue neg_x2 = fb.Negate(x2);
  BValue extended_neg_x1 = fb.SignExtend(neg_x1, 8);
  BValue should_be_zero = fb.Add(neg_x2, x2);
  fb.Add(extended_neg_x1, should_be_zero);

  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

// Discovered by the IR fuzzer 2025-07-17 (id:
// 94240786ff168aa236eb9b9f5d5215223c9d5ce9).
TEST_F(ReassociationPassTest, AddZeroToOverflowValueKeepsOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(1));
  BValue one = fb.Literal(UBits(1, 1));
  // This add can overflow
  BValue add_1 = fb.Add(one, param);
  BValue zero = fb.Literal(UBits(0, 1));
  // But this one looks like it can't. Really though the full value is still
  // overflowing.
  BValue add_2 = fb.Add(add_1, zero);
  BValue zero_ext = fb.ZeroExtend(add_2, 64);
  BValue big_zero = fb.Literal(UBits(0, 64));
  fb.Add(zero_ext, big_zero);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ScopedRecordIr sri(p.get(), "_ir");

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
}

TEST_F(ReassociationPassTest, BadReassociation) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, Parser::ParsePackage(R"ir(
package FuzzTest

top fn FuzzTest(p3: bits[1] id=1) -> bits[64] {
  zero_literal: bits[1] = literal(value=0, id=2)
  one_literal_1: bits[1] = literal(value=1, id=3)
  one_literal_2: bits[1] = literal(value=1, id=4)
  add.5: bits[1] = add(p3, zero_literal, id=5)
  sub.6: bits[1] = sub(one_literal_1, one_literal_2, id=6)
  add.7: bits[1] = add(add.5, sub.6, id=7)
  zero_literal_7: bits[7] = literal(value=0, id=8)
  zero_ext.9: bits[64] = zero_ext(add.7, new_bit_count=64, id=9)
  zero_ext.10: bits[64] = zero_ext(zero_literal_7, new_bit_count=64, id=10)
  ret add.11: bits[64] = add(zero_ext.9, zero_ext.10, id=11)
}
  )ir"));
  ScopedVerifyEquivalence stays_equivalent(
      p->GetTop().value()->AsFunctionOrDie(), kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
}

void IrFuzzReassociation(FuzzPackageWithArgs fuzz_package_with_args) {
  ReassociationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzReassociation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
