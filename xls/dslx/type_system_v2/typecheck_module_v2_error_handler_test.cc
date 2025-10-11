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

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/type_system_v2/type_inference_error_handler.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"

namespace xls::dslx {
namespace {

using ::testing::AllOf;

struct ErrorCase {
  std::string candidates_string;
  std::function<absl::StatusOr<const TypeAnnotation*>(
      Module& module, absl::Span<const CandidateType> candidates)>
      handler;
};

std::string FormatCandidates(absl::Span<const CandidateType> candidates) {
  std::vector<std::string> annotations;
  for (const CandidateType& next : candidates) {
    annotations.push_back(next.annotation->ToString());
  }
  absl::c_sort(annotations);
  return absl::StrJoin(annotations, ",");
}

class FakeErrorHandler {
 public:
  // Adds an error case to handle, with a lambda for what to do about it.
  void AddCase(ErrorCase error_case) {
    cases_.emplace(error_case.candidates_string, error_case);
  }

  // Deals out a handler function that can be passed to type inference.
  TypeInferenceErrorHandler fn() {
    return [this](const AstNode* node,
                  absl::Span<const CandidateType> candidates) {
      std::string candidates_string = FormatCandidates(candidates);
      const auto it = cases_.find(candidates_string);
      EXPECT_NE(it, cases_.end())
          << "No case added for candidates: " << candidates_string;
      const ErrorCase& error_case = it->second;
      auto result =
          error_case.handler(*candidates[0].annotation->owner(), candidates);
      handled_node_strings_.insert(node->ToString());
      return result;
    };
  }

  // Functions to check what happened after use.
  bool HandledAnyNode() { return !handled_node_strings_.empty(); }
  bool HandledNode(std::string_view node_string) {
    return handled_node_strings_.contains(node_string);
  }

 private:
  absl::flat_hash_map<std::string, ErrorCase> cases_;
  absl::flat_hash_set<std::string> handled_node_strings_;
};

TEST(TypecheckModuleV2ErrorHandlerTest, SignednessMismatch) {
  // The code has a mismatch between X and Y, so it will invoke the error
  // handler for both X and Y. For this example, we use s64 to resolve the
  // conflict in the error handler. Note that the handler could do something
  // nonsensical like return 2 new conflicting types. The correct real usage
  // would be to check for that with a second whole typechecking run.
  FakeErrorHandler handler;
  handler.AddCase(ErrorCase{
      .candidates_string = "s32,u32",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return CreateUnOrSnAnnotation(module, Span::Fake(), true, 64);
      }});

  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(
                                                       R"(
const X = u32:1;
const Y = s32:2;
const Z = X == Y;
)",
                                                       handler.fn()));

  EXPECT_TRUE(handler.HandledNode("X"));
  EXPECT_TRUE(handler.HandledNode("Y"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, AllOf(HasNodeWithType("X", "sN[64]"),
                                      HasNodeWithType("Y", "sN[64]"),
                                      HasNodeWithType("Z", "uN[1]")));
}

TEST(TypecheckModuleV2ErrorHandlerTest, EnumMismatch) {
  // Use a handler that deals with the enum mismatch comparison
  // `EnumA::FOO == EnumB::BAR` by unifying both terms to u32.
  FakeErrorHandler handler;
  handler.AddCase(ErrorCase{
      .candidates_string = "EnumA,EnumB",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return CreateUnOrSnAnnotation(module, Span::Fake(), false, 32);
      }});

  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(
                                                       R"(
enum EnumA : u32 {
  FOO = 1
}

enum EnumB : u32 {
  BAR = 2
}

const Z = EnumA::FOO == EnumB::BAR;
)",
                                                       handler.fn()));

  EXPECT_TRUE(handler.HandledNode("EnumA::FOO"));
  EXPECT_TRUE(handler.HandledNode("EnumB::BAR"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, AllOf(HasNodeWithType("EnumA::FOO", "uN[32]"),
                                      HasNodeWithType("EnumB::BAR", "uN[32]"),
                                      HasNodeWithType("Z", "uN[1]")));
}

TEST(TypecheckModuleV2ErrorHandlerTest, TypeMismatch) {
  FakeErrorHandler handler;
  handler.AddCase(ErrorCase{
      .candidates_string = "u32,u32[5],u64",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return CreateUnOrSnAnnotation(module, Span::Fake(), true, 64);
      }});

  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(
                                                       R"(
fn a(x: u32[5], y: u32) -> u64 { x + y }
)",
                                                       handler.fn()));
  EXPECT_TRUE(handler.HandledNode("x"));
  EXPECT_TRUE(handler.HandledNode("y"));
  EXPECT_TRUE(handler.HandledNode("x + y"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, AllOf(HasNodeWithType("x", "sN[64]"),
                                      HasNodeWithType("y", "sN[64]")));
}

TEST(TypecheckModuleV2ErrorHandlerTest, MultipleErrors) {
  FakeErrorHandler handler;
  handler.AddCase(ErrorCase{
      .candidates_string = "u32,u32[5],u64",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return CreateUnOrSnAnnotation(module, Span::Fake(), true, 64);
      }});
  handler.AddCase(ErrorCase{
      .candidates_string = "u16,u32",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return CreateUnOrSnAnnotation(module, Span::Fake(), false, 32);
      }});

  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(
                                                       R"(
fn a(x: u32[5], y: u32) -> u64 { x + y }
fn b(c: u16, d: u32) -> bool { c == d }
)",
                                                       handler.fn()));
  EXPECT_TRUE(handler.HandledNode("c"));
  EXPECT_TRUE(handler.HandledNode("d"));
  EXPECT_TRUE(handler.HandledNode("x"));
  EXPECT_TRUE(handler.HandledNode("y"));
  EXPECT_TRUE(handler.HandledNode("x + y"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(
      type_info_string,
      AllOf(HasNodeWithType("x", "sN[64]"), HasNodeWithType("y", "sN[64]"),
            HasNodeWithType("c", "uN[32]"), HasNodeWithType("d", "uN[32]")));
}

TEST(TypecheckModuleV2ErrorHandlerTest, PropagatesHandlerError) {
  FakeErrorHandler handler;
  handler.AddCase(ErrorCase{
      .candidates_string = "u32,u32[5],u64",
      .handler = [](Module& module, absl::Span<const CandidateType> candidates)
          -> absl::StatusOr<const TypeAnnotation*> {
        return absl::InvalidArgumentError("Can't fix");
      }});

  EXPECT_THAT(
      TypecheckV2(
          R"(
fn a(x: u32[5], y: u32) -> u64 { x + y }
)",
          handler.fn()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Can't fix")));
}

TEST(TypecheckModuleV2ErrorHandlerTest, NotInvokedWithNoErrors) {
  FakeErrorHandler handler;

  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(
                                                       R"(
fn a(x: u32, y: u32) -> u32 { x + y }
)",
                                                       handler.fn()));
  EXPECT_FALSE(handler.HandledAnyNode());
  XLS_ASSERT_OK_AND_ASSIGN(std::string type_info_string,
                           TypeInfoToString(result.tm));
  EXPECT_THAT(type_info_string, AllOf(HasNodeWithType("x", "uN[32]"),
                                      HasNodeWithType("y", "uN[32]")));
}

}  // namespace
}  // namespace xls::dslx
