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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class TableSwitchPassTest : public IrTestBase {
 protected:
  // Returns a vector holding the results of the given function when run with
  // the values from 0 to "max_index" before the table swich pass is applied.
  absl::StatusOr<std::vector<Value>> GetBeforeData(Function* f, int max_index,
                                                   int width = 32) {
    // Run a bunch of data past the intended bounds, just for extra safety.
    constexpr int kOverflow = 128;

    std::vector<Value> data;
    for (int i = 0; i < max_index + kOverflow; i++) {
      XLS_ASSIGN_OR_RETURN(Value value,
                           IrInterpreter::Run(f, {Value(UBits(i, width))}));
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
          IrInterpreter::Run(
              f, {Value(UBits(i, before_data[0].GetFlatBitCount()))}));
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
package p

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

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));

  // Capture the behavior before the transformation.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({1, 2, 3, 4, 5, 6, 0}, 32));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Literal(array),
                                               /*indices=*/{m::Param()}));

  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

// This test verifies that table switching works if the selects are in
// lowest-to-highest selection order (in terms of dependencies).
TEST_F(TableSwitchPassTest, HandlesLowHighOrder) {
  constexpr int kNumLiterals = 7;

  const std::string program = R"(
package p

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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
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
package p

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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
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
package p

fn main(index: bits[32]) -> bits[32] {
  literal.0: bits[32] = literal(value=0)
  literal.1 : bits[32]= literal(value=1)
  eq.4: bits[1] = eq(index, literal.0)
  ret result: bits[32] = sel(eq.4, cases=[literal.0, literal.1])
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(false));
}

// Verifies that TableSwitch only allows binary selects.
TEST_F(TableSwitchPassTest, BinaryOnly) {
  const std::string program = R"(
package p

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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  EXPECT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(false));
}

// Verifies that non-dense index sets (e.g., [0, 1, 2, 5]) aren't switched, but
// subsets >= the minimum size are.
TEST_F(TableSwitchPassTest, DenseOnly) {
  const std::string program = R"(
package p

fn main(index: bits[32]) -> bits[32] {
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
  eq.10: bits[1] = eq(index, literal.0)
  eq.11: bits[1] = eq(index, literal.1)
  eq.12: bits[1] = eq(index, literal.2)
  eq.14: bits[1] = eq(index, literal.4)
  eq.15: bits[1] = eq(index, literal.5)
  sel.20: bits[32] = sel(eq.10, cases=[literal.50, literal.51])
  sel.21: bits[32] = sel(eq.11, cases=[sel.20, literal.52])
  sel.22: bits[32] = sel(eq.12, cases=[sel.21, literal.53])
  sel.24: bits[32] = sel(eq.14, cases=[sel.22, literal.55])
  ret sel.25: bits[32] = sel(eq.15, cases=[sel.24, literal.56])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Value array,
                           Value::UBitsArray({111, 222, 333, 0}, 32));
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
package p

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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
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
package p

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

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));

  // Note the extra literals here.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals));

  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
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
package p

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
  for (int i = 0; i < kNumLiterals; i++) {
    literals.push_back(std::string(num_characters, 'a' + i));
  }
  program = absl::Substitute(program, literals[0], literals[1], literals[2],
                             literals[3], literals[4], literals[5]);
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> before_data,
                           GetBeforeData(f, kNumLiterals, /*width=*/128));

  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
  XLS_ASSERT_OK(CompareBeforeAfter(f, before_data));
}

TEST_F(TableSwitchPassTest, Crasher70d2fb09) {
  // Minimized example extracted from the optimization pipeline right before
  // TableSwitchPass is run.
  std::string program = R"(
package sample

fn main(x0: bits[53]) -> bits[53] {
  x1: bits[1] = eq(x0, x0, id=2, pos=0,3,20)
  literal.31: bits[53] = literal(value=0, id=31, pos=0,18,28)
  x12: bits[53] = sel(x1, cases=[x0, literal.31], id=42, pos=0,18,28)
  ret x15: bits[53] = sel(x1, cases=[x12, literal.31], id=43, pos=0,21,28)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  PassResults results;
  TableSwitchPass pass;
  ASSERT_THAT(pass.RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
