// Copyright 2026 The XLS Authors
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

#include "xls/dslx/type_system_v2/invocation_collector.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {
namespace {

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class InvocationCollectorTest : public ::testing::Test {
 public:
  absl::StatusOr<std::unique_ptr<Module>> Parse(std::string_view program) {
    scanner_.emplace(file_table_, Fileno(0),
                     absl::StrCat("#![feature(generics)]\n\n", program));
    parser_.emplace("test", &*scanner_);
    return parser_->ParseModule();
  }

  // Returns the `Param` nodes of `fn` whose names are in `names`.
  static absl::flat_hash_set<const Param*> GetParams(
      const Function* fn, const std::vector<std::string>& names) {
    absl::flat_hash_set<const Param*> result;
    for (const Param* param : fn->params()) {
      for (const std::string& name : names) {
        if (param->identifier() == name) {
          result.insert(param);
        }
      }
    }
    return result;
  }

  // Runs the collector over the body of `fn` and returns the string form of
  // each collected invocation.
  static std::vector<std::string> CollectFromBody(
      const Function* fn, absl::flat_hash_set<const Param*> params) {
    std::vector<std::string> result;
    for (const Invocation* invocation :
         CollectInvocationsReferencingParams(fn->body(), std::move(params))) {
      result.push_back(invocation->ToString());
    }
    return result;
  }

  FileTable file_table_;
  std::optional<Scanner> scanner_;
  std::optional<Parser> parser_;
};

TEST_F(InvocationCollectorTest, CallOfFunctionalParam) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(f: fn(u32) -> u32, x: u32) -> u32 {
  f(x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f"})),
              UnorderedElementsAre("f(x)"));
}

TEST_F(InvocationCollectorTest, ParamPassedAsArgument) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn foo(x: u32) -> u32 {
  bar(x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"x"})),
              UnorderedElementsAre("bar(x)"));
}

TEST_F(InvocationCollectorTest, NoInvocations) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(x: u32) -> u32 {
  x + u32:1
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"x"})), IsEmpty());
}

TEST_F(InvocationCollectorTest, ParamReferencedOutsideInvocationIsIgnored) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn foo(x: u32) -> u32 {
  bar(u32:1) + x
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"x"})), IsEmpty());
}

TEST_F(InvocationCollectorTest, EmptyParamSet) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(f: fn(u32) -> u32, x: u32) -> u32 {
  f(x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, {}), IsEmpty());
}

TEST_F(InvocationCollectorTest, OnlyInvocationsReferencingGivenParams) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn foo(f: fn(u32) -> u32, g: fn(u32) -> u32, x: u32, y: u32) -> u32 {
  f(u32:1) + g(u32:2) + bar(x) + bar(y)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f", "y"})),
              UnorderedElementsAre("f(u32:1)", "bar(y)"));
}

TEST_F(InvocationCollectorTest, SameInvocationCollectedOnce) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(f: fn(u32, u32) -> u32, x: u32) -> u32 {
  f(x, x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f", "x"})),
              UnorderedElementsAre("f(x, x)"));
}

TEST_F(InvocationCollectorTest, NestedInvocationOnlyInnermostCollected) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn foo(f: fn(u32) -> u32, x: u32) -> u32 {
  bar(f(x))
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f"})),
              UnorderedElementsAre("f(x)"));
}

TEST_F(InvocationCollectorTest, OuterInvocationCollectedAfterNestedOne) {
  // The reference to `x` comes after the nested `bar(u32:1)` call, so this
  // verifies that the outer invocation is restored as current after visiting
  // the inner one.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn baz(a: u32, b: u32) -> u32 { a + b }
fn foo(x: u32) -> u32 {
  baz(bar(u32:1), x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"x"})),
              UnorderedElementsAre("baz(bar(u32:1), x)"));
}

TEST_F(InvocationCollectorTest, InvocationsInStatements) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn bar(a: u32) -> u32 { a }
fn foo(f: fn(u32) -> u32, x: u32) -> u32 {
  let y = f(x);
  let z = bar(y);
  f(z)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  // `bar(y)` references a local, not a param, so it is not collected.
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f"})),
              UnorderedElementsAre("f(x)", "f(z)"));
}

TEST_F(InvocationCollectorTest, InvocationsInControlFlow) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(f: fn(u32) -> u32, g: fn(u32) -> u32, c: bool, x: u32) -> u32 {
  if c { f(x) } else { g(x) }
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"f", "g"})),
              UnorderedElementsAre("f(x)", "g(x)"));
}

TEST_F(InvocationCollectorTest, BuiltinInvocationReferencingParam) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> module, Parse(R"(
fn foo(x: u32) -> u32 {
  let _ = trace!(x);
  rev(x)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * foo,
                           module->GetMemberOrError<Function>("foo"));
  EXPECT_THAT(CollectFromBody(foo, GetParams(foo, {"x"})),
              UnorderedElementsAre("trace!(x)", "rev(x)"));
}

}  // namespace
}  // namespace xls::dslx
