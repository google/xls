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

#include "xls/ir/code_template.h"

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

using testing::ElementsAre;
using testing::HasSubstr;
using xls::status_testing::StatusIs;

namespace xls {

namespace {
TEST(CodeTemplateTest, ParsingAndExpressionExtraction) {
  CodeTemplate code_template = *CodeTemplate::Create("");

  XLS_ASSERT_OK_AND_ASSIGN(code_template, CodeTemplate::Create("just text"));
  EXPECT_TRUE(code_template.Expressions().empty());

  XLS_ASSERT_OK_AND_ASSIGN(code_template,
                           CodeTemplate::Create("Time is {time}"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("time"));

  XLS_ASSERT_OK_AND_ASSIGN(code_template, CodeTemplate::Create("Time is {}"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(), ElementsAre(""));

  XLS_ASSERT_OK_AND_ASSIGN(code_template,
                           CodeTemplate::Create("Two {foo} and {bar}"));
  EXPECT_EQ(code_template.Expressions().size(), 2);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("foo", "bar"));

  XLS_ASSERT_OK_AND_ASSIGN(code_template,
                           CodeTemplate::Create("Expr in brace {{{foo}}}"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("foo"));

  XLS_ASSERT_OK_AND_ASSIGN(
      code_template,
      CodeTemplate::Create("Two {foo} {{not-an-expr}} and {bar}"));
  EXPECT_EQ(code_template.Expressions().size(), 2);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("foo", "bar"));

  XLS_ASSERT_OK_AND_ASSIGN(code_template,
                           CodeTemplate::Create("Two ({foo} and {bar})"));
  EXPECT_EQ(code_template.Expressions().size(), 2);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("foo", "bar"));

  XLS_ASSERT_OK_AND_ASSIGN(code_template,
                           CodeTemplate::Create("nested function {fun()}"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("fun()"));

  XLS_ASSERT_OK_AND_ASSIGN(
      code_template, CodeTemplate::Create("nested  {some{inner, braces};}"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("some{inner, braces};"));

  XLS_ASSERT_OK_AND_ASSIGN(
      code_template,
      CodeTemplate::Create("bar {inner {is not} a problem} bar"));
  EXPECT_EQ(code_template.Expressions().size(), 1);
  EXPECT_THAT(code_template.Expressions(),
              ElementsAre("inner {is not} a problem"));
}

TEST(CodeTemplateTest, DetectErrors) {
  EXPECT_THAT(CodeTemplate::Create("foo (() {x}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("4: Parenthesis opened here missing")));
  EXPECT_THAT(CodeTemplate::Create("foo {{abc}x"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("4: Brace opened here missing")));
  EXPECT_THAT(CodeTemplate::Create("foo (( {x}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("5: Parenthesis opened here missing")));
  EXPECT_THAT(CodeTemplate::Create("foo (( {x})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("4: Parenthesis opened here missing")));
  EXPECT_THAT(CodeTemplate::Create("foo (())) {x}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("8: Too many closing parentheses")));
  EXPECT_THAT(CodeTemplate::Create("foo (() {x}}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("11: Too many closing braces")));
  EXPECT_THAT(CodeTemplate::Create("}}{{"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("0: Too many closing braces")));
  EXPECT_THAT(CodeTemplate::Create("foo {"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("4: Dangling opened")));
  EXPECT_THAT(CodeTemplate::Create("foo {x"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("4: Template expression not closed")));
}

TEST(CodeTemplateTest, ExtractErrorColumn) {
  using CT = CodeTemplate;
  EXPECT_EQ(5, CT::ExtractErrorColumn(CT::Create("foo ({").status()));
  EXPECT_EQ(0, CT::ExtractErrorColumn(absl::OkStatus()));  // no contained error
}

TEST(CodeTemplateTest, TemplateFilling) {
  CodeTemplate code_template = *CodeTemplate::Create("");

  // Typical use
  XLS_ASSERT_OK_AND_ASSIGN(
      code_template, CodeTemplate::Create("foo #(.w({width})) {fn} (.a({x}))"));
  EXPECT_EQ(code_template.Expressions().size(), 3);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("width", "fn", "x"));

  const std::string result = code_template.Substitute(
      [](std::string_view name) { return absl::StrCat(name, "_replaced"); });
  EXPECT_EQ(result, "foo #(.w(width_replaced)) fn_replaced (.a(x_replaced))");
}

TEST(CodeTemplateTest, UnescapingBraces) {
  CodeTemplate code_template = *CodeTemplate::Create("");

  XLS_ASSERT_OK_AND_ASSIGN(
      code_template,
      CodeTemplate::Create("foo {e1} {{bar}} {e2} {{{{baz}}}} {{{e3}}}"));
  EXPECT_EQ(code_template.Expressions().size(), 3);
  EXPECT_THAT(code_template.Expressions(), ElementsAre("e1", "e2", "e3"));

  absl::flat_hash_map<std::string, std::string> replacements{
      {"e1", "answer"}, {"e2", "life"}, {"e3", "42"}};
  const std::string result = code_template.Substitute(
      [&replacements](std::string_view name) { return replacements[name]; });
  EXPECT_EQ(result, "foo answer {bar} life {{baz}} {42}");
}

TEST(CodeTemplateTest, ToStringRecreatesOriginalTemplate) {
  // Note, this implictly also tests Substitute() as it is used underneath.
  CodeTemplate code_template = *CodeTemplate::Create("");
  for (std::string_view test_template : {"",                          //
                                         "((){x})"                    //
                                         "{foo}",                     //
                                         "{foo} {{justbraces}}",      //
                                         "{foo} {{{{morebraces}}}}",  //
                                         "{foo} suffix text",         //
                                         "xy {bar}",                  //
                                         "ab {bar}{baz}"}) {
    XLS_ASSERT_OK_AND_ASSIGN(code_template,
                             CodeTemplate::Create(test_template));
    EXPECT_EQ(code_template.ToString(), test_template);
  }
}

}  // namespace
}  // namespace xls
