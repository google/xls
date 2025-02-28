// Copyright 2022 The XLS Authors
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

#include "xls/passes/proc_state_tuple_flattening_pass.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::StartsWith;
using ::testing::UnorderedElementsAre;

enum class NextValueType : std::uint8_t {
  kNextStateVector,
  kNextValueNodes,
};

template <typename Sink>
void AbslStringify(Sink& sink, NextValueType e) {
  absl::Format(&sink, "%s",
               e == NextValueType::kNextStateVector ? "NextStateVector"
                                                    : "NextValueNodes");
}

class ProcStateFlatteningPassTest
    : public IrTestBase,
      public testing::WithParamInterface<NextValueType> {
 protected:
  ProcStateFlatteningPassTest() = default;

  absl::StatusOr<Proc*> BuildProc(ProcBuilder& pb,
                                  absl::Span<const BValue> next_state) {
    switch (GetParam()) {
      case NextValueType::kNextStateVector:
        return pb.Build(next_state);
      case NextValueType::kNextValueNodes: {
        for (int64_t index = 0; index < next_state.size(); ++index) {
          BValue param = pb.GetStateParam(index);
          BValue next_value = next_state[index];
          pb.Next(param, next_value);
        }
        return pb.Build();
      }
    }
    ABSL_UNREACHABLE();
  }
  absl::StatusOr<Proc*> BuildProc(TokenlessProcBuilder& pb,
                                  absl::Span<const BValue> next_state) {
    switch (GetParam()) {
      case NextValueType::kNextStateVector:
        return pb.Build(next_state);
      case NextValueType::kNextValueNodes: {
        for (int64_t index = 0; index < next_state.size(); ++index) {
          BValue param = pb.GetStateParam(index);
          BValue next_value = next_state[index];
          pb.Next(param, next_value);
        }
        return pb.Build();
      }
    }
    ABSL_UNREACHABLE();
  }

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         ProcStateTupleFlatteningPass().Run(
                             p, OptimizationPassOptions(), &results, context));
    // Run dataflow_simplification and dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DataflowSimplificationPass()
            .Run(p, OptimizationPassOptions(), &results, context)
            .status());
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .Run(p, OptimizationPassOptions(), &results, context)
            .status());
    return changed;
  }
};

TEST_P(ProcStateFlatteningPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK(BuildProc(pb, {}).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(ProcStateFlatteningPassTest, NontupleStateProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 64)));
  BValue a = pb.StateElement("a", Value::UBitsArray({1, 2, 3}, 16).value());
  XLS_ASSERT_OK(BuildProc(pb, {x, y, a}));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(ProcStateFlatteningPassTest, EmptyTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {x}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  // An empty tuple decomposes into zero elements, so the proc should be
  // stateless now.
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_P(ProcStateFlatteningPassTest, MultipleEmptyTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  BValue y = pb.StateElement("y", Value::Tuple({}));
  BValue z =
      pb.StateElement("z", Value::Tuple({Value::Tuple({}), Value::Tuple({})}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {x, y, z}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  // All the empty tuples decompose into zero elements, so the proc should be
  // stateless now.
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_P(ProcStateFlatteningPassTest, EmptyTupleAndBitsState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  BValue z = pb.StateElement("z", Value::Tuple({}));
  BValue q = pb.StateElement("q", Value(UBits(0, 64)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {x, y, z, pb.Add(q, q)}));

  EXPECT_EQ(proc->GetStateElementCount(), 4);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 2);

  // The name uniquer thinks the names "y" and "q" are already taken (as they
  // were the names of previously deleted nodes). So the new state params get
  // suffixes.
  // TODO(meheff): 2022/4/7 Figure out how to preserve the names.
  EXPECT_EQ(proc->GetStateRead(int64_t{0})->GetName(), "y__1");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value(UBits(0, 32)));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(m::StateRead("y"), m::StateRead("y"))));

  EXPECT_EQ(proc->GetStateRead(1)->GetName(), "q__1");
  EXPECT_EQ(proc->GetStateElement(1)->initial_value(), Value(UBits(0, 64)));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(1)),
      ElementsAre(m::Next(m::StateRead("q"),
                          m::Add(m::StateRead("q"), m::StateRead("q")))));
}

TEST_P(ProcStateFlatteningPassTest, TrivialTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {x}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateRead(int64_t{0})->GetName(), "x__1");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value(UBits(42, 32)));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      UnorderedElementsAre(m::Next(m::StateRead("x"), m::StateRead("x"))));
}

TEST_P(ProcStateFlatteningPassTest, TrivialTupleStateWithNextExpression) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, BuildProc(pb, {pb.Tuple({pb.Not(pb.TupleIndex(x, 0))})}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateRead(int64_t{0})->GetName(), "x__1");
  EXPECT_EQ(proc->GetStateElement(0)->initial_value(), Value(UBits(42, 32)));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              UnorderedElementsAre(
                  m::Next(m::StateRead("x"), m::Not(m::StateRead("x")))));
}

TEST_P(ProcStateFlatteningPassTest, ComplicatedState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue a = pb.StateElement(
      "a",
      Value::Tuple({Value(UBits(1, 32)),
                    Value::Tuple({Value(UBits(2, 32)), Value(UBits(3, 32))})}));
  BValue b = pb.StateElement("b", Value(UBits(4, 32)));
  BValue c = pb.StateElement(
      "c", Value::Tuple({Value(UBits(5, 32)), Value(UBits(6, 32))}));

  BValue next_a = pb.Tuple({b, c});
  BValue next_b = pb.TupleIndex(a, 0);
  BValue next_c = pb.TupleIndex(a, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           BuildProc(pb, {next_a, next_b, next_c}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 6);

  EXPECT_EQ(proc->GetStateRead(int64_t{0})->GetName(), "a_0");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      UnorderedElementsAre(m::Next(m::StateRead("a_0"), m::StateRead("b"))));

  EXPECT_EQ(proc->GetStateRead(1)->GetName(), "a_1");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(1)),
      UnorderedElementsAre(m::Next(m::StateRead("a_1"), m::StateRead("c_0"))));

  EXPECT_EQ(proc->GetStateRead(2)->GetName(), "a_2");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(2)),
      UnorderedElementsAre(m::Next(m::StateRead("a_2"), m::StateRead("c_1"))));

  EXPECT_EQ(proc->GetStateRead(3)->GetName(), "b__1");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(3)),
      UnorderedElementsAre(m::Next(m::StateRead("b"), m::StateRead("a_0"))));

  EXPECT_EQ(proc->GetStateRead(4)->GetName(), "c_0");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(4)),
      UnorderedElementsAre(m::Next(m::StateRead("c_0"), m::StateRead("a_1"))));

  EXPECT_EQ(proc->GetStateRead(5)->GetName(), "c_1");
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(5)),
      UnorderedElementsAre(m::Next(m::StateRead("c_1"), m::StateRead("a_2"))));
}

TEST_P(ProcStateFlatteningPassTest, NextPredicateIsState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue a = pb.StateElement(
      "a",
      Value::Tuple({Value(UBits(1, 1)),
                    Value::Tuple({Value(UBits(2, 32)), Value(UBits(3, 32))})}));
  BValue b = pb.StateElement("b", Value(UBits(1, 1)));
  BValue not_b = pb.Not(b);

  BValue next_a_if_b = a;
  BValue next_a_if_not_b =
      pb.Tuple({pb.Not(pb.TupleIndex(a, 0)), pb.TupleIndex(a, 1)});
  BValue next_b = not_b;
  pb.Next(a, next_a_if_b, /*pred=*/b);
  pb.Next(a, next_a_if_not_b, /*pred=*/not_b);
  pb.Next(b, next_b);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, BuildProc(pb, {}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true)) << p->DumpIr();

  EXPECT_EQ(proc->GetStateElementCount(), 4);
  EXPECT_THAT(proc->nodes(),
              AllOf(Contains(m::Next(m::StateRead("a_0"), _,
                                     m::StateRead(StartsWith("b")))),
                    Contains(m::Next(m::StateRead("a_0"), _,
                                     m::Not(m::StateRead(StartsWith("b"))))),
                    Contains(m::Next(m::StateRead("a_1"), _,
                                     m::StateRead(StartsWith("b")))),
                    Contains(m::Next(m::StateRead("a_1"), _,
                                     m::Not(m::StateRead(StartsWith("b"))))),
                    Contains(m::Next(m::StateRead("a_2"), _,
                                     m::StateRead(StartsWith("b")))),
                    Contains(m::Next(m::StateRead("a_2"), _,
                                     m::Not(m::StateRead(StartsWith("b"))))),
                    Contains(m::Next(m::StateRead(StartsWith("b")),
                                     m::Not(m::StateRead(StartsWith("b")))))));
}

INSTANTIATE_TEST_SUITE_P(NextValueTypes, ProcStateFlatteningPassTest,
                         testing::Values(NextValueType::kNextStateVector,
                                         NextValueType::kNextValueNodes),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace xls
