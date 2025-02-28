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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

absl::StatusOr<bool> InlineAndRecover(Package* package) {
  PassResults results;
  OptimizationContext context;
  XLS_RETURN_IF_ERROR(
      InliningPass()
          .Run(package, OptimizationPassOptions(), &results, context)
          .status());
  XLS_RETURN_IF_ERROR(
      DeadFunctionEliminationPass()
          .Run(package, OptimizationPassOptions(), &results, context)
          .status());
  XLS_ASSIGN_OR_RETURN(
      bool recovery_changed,
      LabelRecoveryPass().Run(package, OptimizationPassOptions(), &results,
                              context));
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_TRUE(recovery_changed);
  Cover* cover =
      package->GetTopAsFunction().value()->return_value()->As<Cover>();
  EXPECT_EQ(cover->label(), "cover_label");
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_TRUE(recovery_changed);
  Cover* cover =
      package->GetTopAsFunction().value()->return_value()->As<Cover>();
  EXPECT_EQ(cover->label(), "cover_label");
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_FALSE(recovery_changed);
  for (Node* node : package->GetTopAsFunction().value()->nodes()) {
    if (node->Is<Cover>()) {
      EXPECT_THAT(node->As<Cover>()->label(),
                  testing::AnyOf("caller_0_callee_cover_label",
                                 "caller_1_callee_cover_label"));
    }
  }
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_FALSE(recovery_changed);
  for (Node* node : package->GetTopAsFunction().value()->nodes()) {
    if (node->Is<Cover>()) {
      EXPECT_THAT(
          node->As<Cover>()->label(),
          testing::AnyOf(
              "caller_1_callee_one_cover_label",
              "caller_2_callee_two_callee_two_0_callee_one_cover_label"));
    }
  }
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_TRUE(recovery_changed);
  Assert* asrt =
      package->GetTopAsFunction().value()->return_value()->As<Assert>();
  EXPECT_EQ(asrt->label(), "my_assert_label");
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
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kProgram));
  XLS_ASSERT_OK_AND_ASSIGN(bool recovery_changed,
                           InlineAndRecover(package.get()));
  EXPECT_TRUE(recovery_changed);
  Assert* asrt =
      package->GetTopAsFunction().value()->return_value()->As<Assert>();
  EXPECT_EQ(asrt->label(), "my_assert_label");
}

}  // namespace
}  // namespace xls
