// Copyright 2020 Google LLC
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

#include "xls/ir/verifier.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

class VerifierTest : public IrTestBase {
 protected:
  VerifierTest() {}
};

TEST_F(VerifierTest, WellFormedPackage) {
  std::string input = R"(
package WellFormedPackage

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph2(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(Verify(p.get()));
  XLS_ASSERT_OK(Verify(FindFunction("graph", p.get())));
  XLS_ASSERT_OK(Verify(FindFunction("graph2", p.get())));
}

TEST_F(VerifierTest, NonUniqueNodeId) {
  std::string input = R"(
package NonUniqueNodeId

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.2: bits[42] = sub(add.2, add.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  EXPECT_THAT(Verify(f), StatusIs(absl::StatusCode::kInternal,
                                  HasSubstr("ID 2 is not unique")));
}

TEST_F(VerifierTest, NonUniqueFunctionName) {
  std::string input = R"(
package NonUniqueFunctionName

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(Verify(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function with name graph is not unique")));
}

TEST_F(VerifierTest, BinOpOperandTypeMismatch) {
  std::string input = R"(
package BinOpOperandTypeMismatch

fn graph(p: bits[2], q: bits[42], r: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(q, r)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  // Replace lhs of the 'and' with a different bit-width value.
  FindNode("and.1", f)->ReplaceOperand(FindNode("q", f), FindNode("p", f));
  EXPECT_THAT(Verify(f), StatusIs(absl::StatusCode::kInternal,
                                  HasSubstr("Type of operand 0 (bits[2] via p) "
                                            "does not match type of and.1")));
  EXPECT_THAT(Verify(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Type of operand 0 (bits[2] via p) does not "
                                 "match type of and.1")));
}

TEST_F(VerifierTest, SelectWithUselessDefault) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42]) -> bits[42] {
  literal.1: bits[42] = literal(value=42)
  ret sel.2: bits[42] = sel(p, cases=[q, r], default=literal.1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(Verify(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has useless default value")));
}

TEST_F(VerifierTest, SelectWithMissingDefault) {
  std::string input = R"(
package p

fn f(p: bits[2], q: bits[42], r: bits[42], s:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(Verify(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has no default value")));
}

TEST_F(VerifierTest, SelectWithTooNarrowSelector) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42], s:bits[42], t:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s, t])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      Verify(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Selector must have at least 2 bits to select amongst 4 cases")));
}

}  // namespace
}  // namespace xls
