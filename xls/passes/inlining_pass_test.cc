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

#include "xls/passes/inlining_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"
#include "xls/passes/dce_pass.h"

namespace xls {
namespace {

void Inline(absl::string_view program, std::string* output) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("caller"));
  PassResults results;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool changed, InliningPass().RunOnFunction(f, PassOptions(), &results));
  EXPECT_TRUE(changed);
  XLS_ASSERT_OK(DeadCodeEliminationPass()
                    .RunOnFunction(f, PassOptions(), &results)
                    .status());
  *output = f->DumpIr();
}

TEST(InliningPassTest, AddWrapper) {
  const std::string program = R"(
package some_package

fn callee(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn caller() -> bits[32] {
  literal.2: bits[32] = literal(value=2)
  ret invoke.3: bits[32] = invoke(literal.2, literal.2, to_apply=callee)
}
)";

  std::string output;
  Inline(program, &output);

  const std::string expected = R"(fn caller() -> bits[32] {
  literal.2: bits[32] = literal(value=2)
  ret add.4: bits[32] = add(literal.2, literal.2)
}
)";
  EXPECT_EQ(expected, output);
}

TEST(InliningPassTest, Transitive) {
  const std::string program = R"(
package some_package

fn callee2(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn callee1(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=callee2)
}

fn caller() -> bits[32] {
  literal.3: bits[32] = literal(value=2)
  ret invoke.4: bits[32] = invoke(literal.3, literal.3, to_apply=callee1)
}
)";

  std::string output;
  Inline(program, &output);

  const std::string expected = R"(fn caller() -> bits[32] {
  literal.3: bits[32] = literal(value=2)
  ret add.7: bits[32] = add(literal.3, literal.3)
}
)";
  EXPECT_EQ(expected, output);
}

}  // namespace
}  // namespace xls
