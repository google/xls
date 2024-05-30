// Copyright 2023 The XLS Authors
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

#include "xls/passes/label_recovery_pass.h"

#include <memory>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

absl::StatusOr<bool> InlineAndRecover(std::string_view program,
                                      std::string* result_text) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(program));
  PassResults results;
  XLS_RETURN_IF_ERROR(
      InliningPass()
          .Run(package.get(), OptimizationPassOptions(), &results)
          .status());
  XLS_RETURN_IF_ERROR(
      DeadFunctionEliminationPass()
          .Run(package.get(), OptimizationPassOptions(), &results)
          .status());
  XLS_ASSIGN_OR_RETURN(bool recovery_changed,
                       LabelRecoveryPass().Run(
                           package.get(), OptimizationPassOptions(), &results));
  *result_text = package->DumpIr();
  return recovery_changed;
}

// Simple scenario where a cover gets inlined into a callee and we recover the
// original label.
TEST(LabelRecoveryPassTest, CoverLabelNoCollisionRecovery) {
  const std::string kProgram = R"(
package p

fn callee(p: bits[1]) -> () {
  ret cover.10: () = cover(p, label="cover_label")
}

top fn caller(p: bits[1]) -> () {
  ret invoke.20: () = invoke(p, to_apply=callee)
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_TRUE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(p: bits[1]) -> () {
  ret cover_21: () = cover(p, label="cover_label", id=22)
}
)");
}

// Cover that gets inlined and recovered via two levels of callee.
TEST(LabelRecoveryPassTest, CoverLabelNoCollisionRecoveryTwoLevels) {
  const std::string kProgram = R"(
package p

fn calleest(p: bits[1]) -> () {
  ret cover.10: () = cover(p, label="cover_label")
}

fn callee(p: bits[1]) -> () {
  ret invoke.20: () = invoke(p, to_apply=calleest)
}

top fn caller(p: bits[1]) -> () {
  ret invoke.30: () = invoke(p, to_apply=callee)
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_TRUE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(p: bits[1]) -> () {
  ret cover_31__1: () = cover(p, label="cover_label", id=34)
}
)");
}

// Two calls to the same callee so recovery cannot occur.
TEST(LabelRecoveryPassTest, CoverLabelCollisionSameCallee) {
  const std::string kProgram = R"(
package p

fn callee(p: bits[1]) -> () {
  ret cover.10: () = cover(p, label="cover_label")
}

top fn caller(p: bits[1]) -> () {
  invoke.20: () = invoke(p, to_apply=callee)
  invoke.30: () = invoke(p, to_apply=callee)
  ret tuple.40: () = tuple()
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_FALSE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(p: bits[1]) -> () {
  cover_41: () = cover(p, label="caller_0_callee_cover_label", id=42)
  cover_43: () = cover(p, label="caller_1_callee_cover_label", id=44)
  ret tuple.40: () = tuple(id=40)
}
)");
}

// A partial diamond to the same callee so recovery cannot occur.
TEST(LabelRecoveryPassTest, CoverLabelCollisionCalleePartialDiamond) {
  const std::string kProgram = R"(
package p

fn callee_one(p: bits[1]) -> () {
  ret cover.10: () = cover(p, label="cover_label")
}

fn callee_two(p: bits[1]) -> () {
  ret invoke.11: () = invoke(p, to_apply=callee_one)
}

top fn caller(p: bits[1]) -> () {
  invoke.20: () = invoke(p, to_apply=callee_one)
  invoke.30: () = invoke(p, to_apply=callee_two)
  ret tuple.40: () = tuple()
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_FALSE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(p: bits[1]) -> () {
  cover_43: () = cover(p, label="caller_1_callee_one_cover_label", id=44)
  cover_41__1: () = cover(p, label="caller_2_callee_two_callee_two_0_callee_one_cover_label", id=46)
  ret tuple.40: () = tuple(id=40)
}
)");
}

// Assert in a callee where its label gets recovered after inlining.
TEST(LabelRecoveryPassTest, SimpleAssertLabelRecovery) {
  const std::string kProgram = R"(
package p

fn callee(the_token: token, p: bits[1]) -> token {
  ret assert.10: token = assert(the_token, p, label="my_assert_label", message="assertion fired!")
}

top fn caller(the_token: token, p: bits[1]) -> token {
  ret invoke.20: token = invoke(the_token, p, to_apply=callee)
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_TRUE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(the_token: token, p: bits[1]) -> token {
  ret assert_21: token = assert(the_token, p, message="assertion fired!", label="my_assert_label", id=22)
}
)");
}

TEST(LabelRecoveryPassTest, NestedCallAssertLabelRecovery) {
  const std::string kProgram = R"(
package p

fn calleest(the_token: token, p: bits[1]) -> token {
  ret assert.10: token = assert(the_token, p, label="my_assert_label", message="assertion fired!")
}

fn callee(the_token: token, p: bits[1]) -> token {
  ret invoke.20: token = invoke(the_token, p, to_apply=calleest)
}

top fn caller(the_token: token, p: bits[1]) -> token {
  ret invoke.30: token = invoke(the_token, p, to_apply=callee)
}
)";
  std::string result_text;
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(kProgram, &result_text));
  EXPECT_TRUE(recovery_changed);
  EXPECT_EQ(result_text, R"(package p

top fn caller(the_token: token, p: bits[1]) -> token {
  ret assert_31__1: token = assert(the_token, p, message="assertion fired!", label="my_assert_label", id=34)
}
)");
}

}  // namespace
}  // namespace xls
