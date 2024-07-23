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

#include "xls/passes/table_switch_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::AnyOf;

class TableSwitchPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    TableSwitchPass pass;
    return pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results);
  }

  // Returns a vector holding the results of the given function when run with
  // the values from 0 to "max_index" before the table switch pass is applied.
  absl::StatusOr<std::vector<Value>> GetBeforeData(Function* f, int max_index,
                                                   int width = 32) {
    // Run a bunch of data past the intended bounds, just for extra safety.
    constexpr int kOverflow = 128;

    std::vector<Value> data;
    for (int i = 0; i < max_index + kOverflow; i++) {
      XLS_ASSIGN_OR_RETURN(Value value, DropInterpreterEvents(InterpretFunction(
                                            f, {Value(UBits(i, width))})));
      data.push_back(value);
    }
    return data;
  }

  // Compares values from before the table switch pass is applied to those
  // afterwards, returning an error if there's a mismatch.
  absl::Status CompareBeforeAfter(Function* f,
                                  const std::vector<Value>& before_data) {
    for (int i = 0; i < before_data.size(); i++) {
      XLS_ASSIGN_OR_RETURN(
          Value value,
          DropInterpreterEvents(InterpretFunction(
              f, {Value(UBits(i, before_data[0].GetFlatBitCount()))})));
      if (value != before_data[i]) {
        return absl::InternalError(
            absl::StrFormat("Args don't match - expected: %s, actual %s",
                            before_data[i].ToString(), value.ToString()));
      }
    }

    return absl::OkStatus();
  }
};

// Verifies that an N-deep tree is converted into a table lookup; smoke test.
TEST_F(TableSwitchPassTest, SwitchesBinaryTree) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(eq.10, cases=[literal.0, literal.1])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.2])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.3])
  sel.23: bits[32] = sel(eq.13, cases=[sel.22, literal.4])
  sel.24: bits[32] = sel(eq.14, cases=[sel.23, literal.5])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.24, literal.6])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Capture the behavior before the transformation.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({1, 2, 3, 4, 5, 6, 0}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));

  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// Verifies that an N-deep tree is converted into a table lookup; smoke test.
TEST_F(TableSwitchPassTest, SimplePrioritySelectLookup) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  concat.16: bits[6] = concat(eq.15, eq.14, eq.13, eq.12, eq.11, eq.10)
  ret priority_sel.17: bits[32] = priority_sel(concat.16, cases=[literal.1, literal.2, literal.3, literal.4, literal.5, literal.6], default=literal.0)
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Capture the behavior before the transformation.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({1, 2, 3, 4, 5, 6, 0}, 32));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Literal(array),
                            /*indices=*/{m::Param("index")}));

  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// Verifies that chained is converted into a table lookup; smoke test.
TEST_F(TableSwitchPassTest, ChainedPrioritySelectLookup) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  literal.7: bits[32] = literal(value=7)
  literal.8: bits[32] = literal(value=8)
  literal.9: bits[32] = literal(value=9)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  concat.16: bits[6] = concat(eq.15, eq.14, eq.13, eq.12, eq.11, eq.10)
  priority_sel.17: bits[32] = priority_sel(concat.16, cases=[literal.1, literal.2, literal.3, literal.4, literal.5, literal.6], default=literal.0)
  eq.18: bits[1] = eq(index, literal.6)
  priority_sel.19: bits[32] = priority_sel(eq.18, cases=[literal.7], default=priority_sel.17)
  eq.20: bits[1] = eq(index, literal.7)
  eq.21: bits[1] = eq(index, literal.8)
  concat.22: bits[2] = concat(eq.21, eq.20)
  ret priority_sel.24: bits[32] = priority_sel(concat.22, cases=[literal.8, literal.9], default=priority_sel.19)
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Capture the behavior before the transformation.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array, Value::UBitsArray({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, 32));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Literal(array),
                            /*indices=*/{m::Param("index")}));

  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// This test verifies that table switching works if the selects are in
// lowest-to-highest selection order (in terms of dependencies).
TEST_F(TableSwitchPassTest, HandlesLowHighOrder) {
  constexpr int kNumLiterals = 7;

  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(eq.15, cases=[literal.5, literal.6])
  sel.21: bits[32] = sel(eq.14, cases=[sel.20, literal.4])
  sel.22: bits[32] = sel(eq.13, cases=[sel.21, literal.3])
  sel.23: bits[32] = sel(eq.12, cases=[sel.22, literal.2])
  sel.24: bits[32] = sel(eq.11, cases=[sel.23, literal.1])
  ret sel.25: bits[32] = sel(eq.10, cases=[sel.24, literal.0])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({0, 1, 2, 3, 4, 6, 5}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// This test verifies that table switching works if the selects are in
// lowest-to-highest selection order (in terms of dependencies).
TEST_F(TableSwitchPassTest, SimplePrioritySelectHandlesLowHighOrder) {
  constexpr int kNumLiterals = 7;

  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  concat.16: bits[6] = concat(eq.15, eq.14, eq.13, eq.12, eq.11, eq.10)
  ret priority_sel.17: bits[32] = priority_sel(concat.16, cases=[literal.0, literal.1, literal.2, literal.3, literal.4, literal.6], default=literal.5)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({0, 1, 2, 3, 4, 6, 5}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// This test verifies that the values of the literals themselves don't matter.
TEST_F(TableSwitchPassTest, IgnoresLiterals) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  literal.50: bits[32] = literal(value=434)
  literal.51: bits[32] = literal(value=3335)
  literal.52: bits[32] = literal(value=2)
  literal.53: bits[32] = literal(value=889798)
  literal.54: bits[32] = literal(value=436)
  literal.55: bits[32] = literal(value=1235)
  literal.56: bits[32] = literal(value=555)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(eq.10, cases=[literal.50, literal.51])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.53])
  sel.23: bits[32] = sel(eq.13, cases=[sel.22, literal.54])
  sel.24: bits[32] = sel(eq.14, cases=[sel.23, literal.55])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.24, literal.56])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  PassResults results;
  TableSwitchPass pass;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array,
      Value::UBitsArray({3335, 2, 889798, 436, 1235, 555, 434}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
}

// This test verifies that the values of the literals themselves don't matter.
TEST_F(TableSwitchPassTest, SimplePrioritySelectIgnoresLiterals) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  literal.50: bits[32] = literal(value=434)
  literal.51: bits[32] = literal(value=3335)
  literal.52: bits[32] = literal(value=2)
  literal.53: bits[32] = literal(value=889798)
  literal.54: bits[32] = literal(value=436)
  literal.55: bits[32] = literal(value=1235)
  literal.56: bits[32] = literal(value=555)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  concat.16: bits[6] = concat(eq.10, eq.11, eq.12, eq.13, eq.14, eq.15)
  ret priority_sel.17: bits[32] = priority_sel(concat.16, cases=[literal.56, literal.55, literal.54, literal.53, literal.52, literal.51], default=literal.50)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  PassResults results;
  TableSwitchPass pass;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array,
      Value::UBitsArray({3335, 2, 889798, 436, 1235, 555, 434}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
}

// Verifies that a single literal switch is _not_ converted into a table
// lookup.
TEST_F(TableSwitchPassTest, SkipsTrivial) {
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1 : bits[32]= literal(value=1)
  eq.4: bits[1] = eq(index, literal.0)
  ret result: bits[32] = sel(eq.4, cases=[literal.0, literal.1])
}
)";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  PassResults results;
  TableSwitchPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(false));
}

// Verifies that a single literal switch is _not_ converted into a table
// lookup.
TEST_F(TableSwitchPassTest, SkipsTrivialPrioritySelect) {
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1 : bits[32]= literal(value=1)
  eq.4: bits[1] = eq(index, literal.0)
  ret result: bits[32] = priority_sel(eq.4, cases=[literal.1], default=literal.0)
}
)";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  PassResults results;
  TableSwitchPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(false));
}

// Verifies that TableSwitch only allows binary selects.
TEST_F(TableSwitchPassTest, BinaryOnly) {
  const std::string program = R"(
fn main(index: bits[32], bad_selector: bits[3]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.50: bits[32] = literal(value=0)
  literal.51: bits[32] = literal(value=111)
  literal.52: bits[32] = literal(value=222)
  literal.53: bits[32] = literal(value=333)
  literal.55: bits[32] = literal(value=555)
  literal.56: bits[32] = literal(value=666)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(bad_selector, cases=[literal.50, literal.51, literal.56], default=literal.55)
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.53])
  sel.24: bits[32] = sel(eq.14, cases=[sel.22, literal.55])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.24, literal.56])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  PassResults results;
  TableSwitchPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(false));
}

// Verifies that non-zero index sets, even with holes (e.g., [1, 2, 3, 5]), can
// be switched to tables.
TEST_F(TableSwitchPassTest, NonzeroStart) {
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.5: bits[32] = literal(value=5)
  literal.50: bits[32] = literal(value=0)
  literal.51: bits[32] = literal(value=111)
  literal.52: bits[32] = literal(value=222)
  literal.53: bits[32] = literal(value=333)
  literal.55: bits[32] = literal(value=555)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.15: bits[1] = eq(index, literal.5)
  sel.21: bits[32] = sel(eq.11, cases=[literal.50, literal.51])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.52])
  sel.23: bits[32] = sel(eq.13, cases=[sel.22, literal.53])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.23, literal.55])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array, Value::UBitsArray({0, 111, 222, 333, 0, 555, 0}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
}

// Verifies that non-dense index sets (e.g., [0, 1, 3, 7, 9]) aren't switched,
// but subsets >= the minimum size are.
TEST_F(TableSwitchPassTest, DenseOnly) {
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.3: bits[32] = literal(value=3)
  literal.7: bits[32] = literal(value=7)
  literal.9: bits[32] = literal(value=9)
  literal.50: bits[32] = literal(value=0)
  literal.51: bits[32] = literal(value=111)
  literal.52: bits[32] = literal(value=222)
  literal.53: bits[32] = literal(value=333)
  literal.57: bits[32] = literal(value=777)
  literal.59: bits[32] = literal(value=999)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.13: bits[1] = eq(index, literal.3)
  eq.17: bits[1] = eq(index, literal.7)
  eq.19: bits[1] = eq(index, literal.9)
  sel.20: bits[32] = sel(eq.10, cases=[literal.50, literal.51])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[32] = sel(eq.13, cases=[sel.21, literal.53])
  sel.24: bits[32] = sel(eq.17, cases=[sel.22, literal.57])
  ret sel.25: bits[32] = sel(eq.19, cases=[sel.24, literal.59])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({111, 222, 0, 333, 0}, 32));
  bool has_array_index = false;
  for (const Node* node : f->nodes()) {
    if (node->op() == Op::kArrayIndex) {
      has_array_index = true;
      EXPECT_THAT(node, m::ArrayIndex(m::Literal(array),
                                      /*indices=*/{m::Param()}));
    }
  }
  EXPECT_TRUE(has_array_index);
}

// This test verifies that two separate table-switches can still be combined in
// a larger computation.
TEST_F(TableSwitchPassTest, CanCombine) {
  constexpr int kNumLiterals = 8;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  literal.50: bits[32] = literal(value=0)
  literal.51: bits[32] = literal(value=111)
  literal.52: bits[32] = literal(value=222)
  literal.53: bits[32] = literal(value=333)
  literal.54: bits[32] = literal(value=444)
  literal.55: bits[32] = literal(value=555)
  literal.56: bits[32] = literal(value=666)
  literal.57: bits[32] = literal(value=777)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  ugt.16: bits[1] = ugt(index, literal.3)
  sel.20: bits[32] = sel(eq.10, cases=[literal.50, literal.51])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.53])

  sel.23: bits[32] = sel(eq.13, cases=[literal.54, literal.55])
  sel.24: bits[32] = sel(eq.14, cases=[sel.23, literal.56])
  sel.25: bits[32] = sel(eq.14, cases=[sel.24, literal.57])
  ret result: bits[32] = sel(eq.10, cases=[sel.22, sel.25])
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  bool has_array_index = false;
  XLS_ASSERT_OK_AND_ASSIGN(Value array_0,
                           Value::UBitsArray({111, 222, 333, 0}, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Value array_1,
                           Value::UBitsArray({555, 666, 777, 444}, 32));
  for (const Node* node : f->nodes()) {
    if (node->op() == Op::kArrayIndex) {
      has_array_index = true;
      EXPECT_THAT(node, AnyOf(m::ArrayIndex(m::Literal(array_0),
                                            /*indices=*/{m::Param()}),
                              m::ArrayIndex(m::Literal(array_1),
                                            /*indices=*/{m::Param()})));
    }
  }
  EXPECT_TRUE(has_array_index);
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// This test verifies that OOB accesses to this structure return the correct
// value. In the original select chains, the "terminal" case is sort-of an
// final "else" case: if we match that top select, then we return the right-hand
// case. If not, then the index is unknown - it's out of the range of the
// existing specific-integral matches.
// To make our array rewrite match this behavior, then we need to make the last
// element that "else" - in XLS, OOB array_index accesses return the last
// element in the array.
TEST_F(TableSwitchPassTest, HandlesOOBAccesses) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(eq.10, cases=[literal.0, literal.1])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.2])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.3])
  sel.23: bits[32] = sel(eq.13, cases=[sel.22, literal.4])
  sel.24: bits[32] = sel(eq.14, cases=[sel.23, literal.5])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.24, literal.6])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Note the extra literals here.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({1, 2, 3, 4, 5, 6, 0}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// Verifies that TableSwitchPass works for literals > 64b.
TEST_F(TableSwitchPassTest, EnormousLiterals) {
  constexpr int kNumLiterals = 6;
  std::string program = R"(
fn main(index: bits[128]) -> bits[128] {
  literal.0: bits[128] = literal(value=0x0)
  literal.1: bits[128] = literal(value=0x1)
  literal.2: bits[128] = literal(value=0x2)
  literal.3: bits[128] = literal(value=0x3)
  literal.4: bits[128] = literal(value=0x4)
  literal.5: bits[128] = literal(value=0x5)
  literal.50: bits[128] = literal(value=0x$0)
  literal.51: bits[128] = literal(value=0x$1)
  literal.52: bits[128] = literal(value=0x$2)
  literal.53: bits[128] = literal(value=0x$3)
  literal.54: bits[128] = literal(value=0x$4)
  literal.55: bits[128] = literal(value=0x$5)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[128] = sel(eq.10, cases=[literal.50, literal.51])
  sel.21: bits[128] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[128] = sel(eq.12, cases=[sel.21, literal.53])
  sel.23: bits[128] = sel(eq.13, cases=[sel.22, literal.54])
  ret sel.24: bits[128] = sel(eq.14, cases=[sel.23, literal.55])
})";

  std::vector<std::string> literals;
  int num_characters = 128 / 4;
  // Create the literal strings ("aaaa....aaaa" and so on).
  literals.reserve(kNumLiterals);
  for (int i = 0; i < kNumLiterals; i++) {
    literals.push_back(std::string(num_characters, 'a' + i));
  }
  program = absl::Substitute(program, literals[0], literals[1], literals[2],
                             literals[3], literals[4], literals[5]);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals, /*width=*/128));

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

TEST_F(TableSwitchPassTest, Crasher70d2fb09) {
  // Minimized example extracted from the optimization pipeline right before
  // TableSwitchPass is run.
  std::string program = R"(
fn main(x0: bits[53]) -> bits[53] {
  x1: bits[1] = eq(x0, x0, id=2, pos=[(0,3,20)])
  literal.31: bits[53] = literal(value=0, id=31, pos=[(0,18,28)])
  x12: bits[53] = sel(x1, cases=[x0, literal.31], id=42, pos=[(0,18,28)])
  ret x15: bits[53] = sel(x1, cases=[x12, literal.31], id=43, pos=[(0,21,28)])
}
)";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(TableSwitchPassTest, Crasher741eb996) {
  // Minimized example extracted from the optimization pipeline right before
  // TableSwitchPass is run.
  std::string program = R"(
package sample

fn main(x4: bits[62], x1: bits[25], x2: bits[24]) -> bits[62] {
  bit_slice.10: bits[24] = bit_slice(x4, start=0, width=24, id=10)
  eq.69: bits[1] = eq(x2, bit_slice.10, id=69, pos=[(0,11,28)])
  literal.57: bits[62] = literal(value=0, id=57, pos=[(0,11,28)])
  literal.5: bits[24] = literal(value=0, id=5, pos=[(0,3,29)])
  x13: bits[62] = sel(eq.69, cases=[x4, literal.57], id=58, pos=[(0,11,28)])
  eq.70: bits[1] = eq(x2, literal.5, id=70, pos=[(0,32,28)])
  x16: bits[62] = sel(eq.69, cases=[x13, literal.57], id=62, pos=[(0,14,28)])
  ret x28: bits[62] = sel(eq.70, cases=[x16, literal.57], id=66, pos=[(0,32,28)])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, OptimizationPassOptions(), &results),
              IsOkAndHolds(false));
}

TEST_F(TableSwitchPassTest, FullIndexSpace) {
  std::string program = R"(
fn main(index: bits[2], else: bits[32]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  eq_0: bits[1] = eq(index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_2: bits[1] = eq(index, literal_2)
  eq_3: bits[1] = eq(index, literal_3)

  // The comparisons can appear in any order. Final else does not have to be a
  // literal because it is dead (never selected by chain).
  sel_3: bits[32] = sel(eq_1, cases=[else, _222])
  sel_2: bits[32] = sel(eq_3, cases=[sel_3, _444])
  sel_1: bits[32] = sel(eq_2, cases=[sel_2, _333])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _111])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(
          m::Literal(Value::UBitsArray({111, 222, 333, 444}, 32).value()),
          /*indices=*/{m::Param()}));
}

TEST_F(TableSwitchPassTest, FullIndexSpaceNe) {
  std::string program = R"(
fn main(index: bits[2], else: bits[32]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  ne_0: bits[1] = ne(index, literal_0)
  ne_1: bits[1] = ne(index, literal_1)
  ne_2: bits[1] = ne(index, literal_2)
  ne_3: bits[1] = ne(index, literal_3)

  // The comparisons can appear in any order. Final else does not have to be a
  // literal because it is dead (never selected by chain).
  sel_3: bits[32] = sel(ne_1, cases=[_222, else])
  sel_2: bits[32] = sel(ne_3, cases=[_444, sel_3])
  sel_1: bits[32] = sel(ne_2, cases=[_333, sel_2])
  ret sel_0: bits[32] = sel(ne_0, cases=[_111, sel_1])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(
          m::Literal(Value::UBitsArray({111, 222, 333, 444}, 32).value()),
          /*indices=*/{m::Param()}));
}

TEST_F(TableSwitchPassTest, FullIndexSpaceMixOfNeAndEq) {
  std::string program = R"(
fn main(index: bits[2], else: bits[32]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  ne_0: bits[1] = ne(index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_2: bits[1] = eq(index, literal_2)
  ne_3: bits[1] = ne(index, literal_3)

  // The comparisons can appear in any order. Final else does not have to be a
  // literal because it is dead (never selected by chain).
  sel_3: bits[32] = sel(eq_1, cases=[else, _222])
  sel_2: bits[32] = sel(ne_3, cases=[_444, sel_3])
  sel_1: bits[32] = sel(eq_2, cases=[sel_2, _333])
  ret sel_0: bits[32] = sel(ne_0, cases=[_111, sel_1])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(
          m::Literal(Value::UBitsArray({111, 222, 333, 444}, 32).value()),
          /*indices=*/{m::Param()}));
}

TEST_F(TableSwitchPassTest, DifferentIndexes) {
  std::string program = R"(
fn main(index: bits[2], other_index: bits[2], else: bits[32]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  eq_0: bits[1] = eq(other_index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_2: bits[1] = eq(index, literal_2)
  eq_3: bits[1] = eq(index, literal_3)

  sel_3: bits[32] = sel(eq_1, cases=[else, _222])
  sel_2: bits[32] = sel(eq_3, cases=[sel_3, _444])
  sel_1: bits[32] = sel(eq_2, cases=[sel_2, _333])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _111])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(TableSwitchPassTest, SingleHoleInIndexSpace) {
  std::string program = R"(
fn main(index: bits[2]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  eq_0: bits[1] = eq(index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_3: bits[1] = eq(index, literal_3)

  // There is no comparison of index to 2. This case is
  // handled by the final alternative _222.
  sel_2: bits[32] = sel(eq_1, cases=[_333, _222])
  sel_1: bits[32] = sel(eq_3, cases=[sel_2, _444])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _111])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(
          m::Literal(Value::UBitsArray({111, 222, 333, 444}, 32).value()),
          /*indices=*/{m::Param()}));
}

TEST_F(TableSwitchPassTest, MultipleHolesInIndexSpace) {
  std::string program = R"(
fn main(index: bits[3]) -> bits[32] {
  _0: bits[32] = literal(value=0)
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)
  _666: bits[32] = literal(value=666)
  _888: bits[32] = literal(value=888)

  literal_0: bits[3] = literal(value=0)
  literal_1: bits[3] = literal(value=1)
  literal_2: bits[3] = literal(value=2)
  literal_3: bits[3] = literal(value=3)
  literal_4: bits[3] = literal(value=4)
  literal_6: bits[3] = literal(value=6)
  eq_0: bits[1] = eq(index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_2: bits[1] = eq(index, literal_2)
  eq_3: bits[1] = eq(index, literal_3)
  eq_4: bits[1] = eq(index, literal_4)
  eq_6: bits[1] = eq(index, literal_6)

  sel_6: bits[32] = sel(eq_6, cases=[_888, _666])
  sel_4: bits[32] = sel(eq_4, cases=[sel_6, _444])
  sel_3: bits[32] = sel(eq_3, cases=[sel_4, _333])
  sel_2: bits[32] = sel(eq_2, cases=[sel_3, _222])
  sel_1: bits[32] = sel(eq_1, cases=[sel_2, _111])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _0])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Literal(Value::UBitsArray(
                                   {0, 111, 222, 333, 444, 888, 666, 888}, 32)
                                   .value()),
                    /*indices=*/{m::Param()}));
}

TEST_F(TableSwitchPassTest, MultipleHolesInIndexSpaceButTooSparse) {
  std::string program = R"(
fn main(index: bits[3]) -> bits[32] {
  _0: bits[32] = literal(value=0)
  _111: bits[32] = literal(value=111)
  _444: bits[32] = literal(value=444)
  _888: bits[32] = literal(value=888)

  literal_0: bits[3] = literal(value=0)
  literal_1: bits[3] = literal(value=1)
  literal_4: bits[3] = literal(value=4)
  eq_0: bits[1] = eq(index, literal_0)
  eq_1: bits[1] = eq(index, literal_1)
  eq_4: bits[1] = eq(index, literal_4)

  sel_2: bits[32] = sel(eq_4, cases=[_888, _444])
  sel_1: bits[32] = sel(eq_1, cases=[sel_2, _111])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _0])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Only three entries are filled in an index space of size 8. The unfilled
  // entries are the fallthrough value of 888. This is beneath the heuristic
  // threshold of half the index space size being filled.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Select());
}

TEST_F(TableSwitchPassTest, DuplicateComparisonsValues) {
  // The string of selects has duplicate comparisons against the same literal.
  std::string program = R"(
fn main(index: bits[2]) -> bits[32] {
  _111: bits[32] = literal(value=111)
  _222: bits[32] = literal(value=222)
  _333: bits[32] = literal(value=333)
  _444: bits[32] = literal(value=444)
  _555: bits[32] = literal(value=555)

  literal_0: bits[2] = literal(value=0)
  literal_1: bits[2] = literal(value=1)
  literal_2: bits[2] = literal(value=2)
  literal_3: bits[2] = literal(value=3)
  eq_0: bits[1] = eq(index, literal_0)
  eq_0_but_swapped: bits[1] = eq(literal_0, index)
  eq_1: bits[1] = eq(index, literal_1)
  eq_3: bits[1] = eq(index, literal_3)

  // There is no comparison of index to 2.
  sel_3: bits[32] = sel(eq_1, cases=[_333, _222])
  sel_2: bits[32] = sel(eq_3, cases=[sel_3, _444])
  // _555 cannot be result of function.
  sel_1: bits[32] = sel(eq_0_but_swapped, cases=[sel_2, _555])
  ret sel_0: bits[32] = sel(eq_0, cases=[sel_1, _111])
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(
          m::Literal(Value::UBitsArray({111, 222, 333, 444}, 32).value()),
          /*indices=*/{m::Param()}));
}

// Verifies that if the chain ends mid-priority-select (and therefore without a
// terminating literal), we terminate & recognize that it can't be replaced this
// way.
TEST_F(TableSwitchPassTest, ChainEndsMidPrioritySelect) {
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  add.16: bits[32] = add(index, index)
  uge.17: bits[1] = uge(add.16, literal.6)
  concat.18: bits[7] = concat(uge.17, eq.15, eq.14, eq.13, eq.12, eq.11, eq.10)
  ret priority_sel.19: bits[32] = priority_sel(concat.18, cases=[literal.1, literal.2, literal.3, literal.4, literal.5, literal.6, add.16], default=literal.0)
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

// Verifies that if the chain starts mid-priority-select, we can still switch
// that part to a table.
TEST_F(TableSwitchPassTest, ChainStartsMidPrioritySelect) {
  constexpr int kNumLiterals = 7;
  const std::string program = R"(
fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1: bits[32] = literal(value=1)
  literal.2: bits[32] = literal(value=2)
  literal.3: bits[32] = literal(value=3)
  literal.4: bits[32] = literal(value=4)
  literal.5: bits[32] = literal(value=5)
  literal.6: bits[32] = literal(value=6)
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.13: bits[1] = eq(index, literal.3)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  add.16: bits[32] = add(index, index)
  uge.17: bits[1] = uge(add.16, literal.6)
  concat.18: bits[7] = concat(eq.15, eq.14, eq.13, eq.12, eq.11, eq.10, uge.17)
  ret priority_sel.19: bits[32] = priority_sel(concat.18, cases=[add.16, literal.1, literal.2, literal.3, literal.4, literal.5, literal.6], default=literal.0)
})";

  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));
  // Capture the behavior before the transformation.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({1, 2, 3, 4, 5, 6, 0}, 32));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::BitSlice(),
          /*cases=*/{m::Add()},
          /*default_value=*/
          m::ArrayIndex(m::Literal(array), /*indices=*/{m::Param("index")})));

  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

}  // namespace
}  // namespace xls
