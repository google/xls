// Copyright 2025 The XLS Authors
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

#include "xls/dslx/diagnostics/warn_on_defined_but_unused.h"

#include <string>
#include <optional>
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(WarnOnDefinedButUnusedTest, SimpleUnusedLocalLetBinding) {
  const std::string program = "fn f() { let x = u32:1; }";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test.x", &import_data));

  WarningCollector warnings{kAllWarningsSet};
  std::optional<Function*> f = tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());
  XLS_ASSERT_OK(WarnOnDefinedButUnused(*f.value(), *tm.type_info, warnings));
  ASSERT_EQ(warnings.warnings().size(), 1);
  EXPECT_EQ(warnings.warnings()[0].message,
            "Definition of `x` (type `uN[32]`) is not used in function `f`");
}

TEST(WarnOnDefinedButUnusedTest, MatchWithUnusedNameDefArm) {
  const std::string program = "fn f(x: u32) -> u32 { match x { y => x } }";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test.x", &import_data));

  WarningCollector warnings{kAllWarningsSet};
  std::optional<Function*> f = tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());
  XLS_ASSERT_OK(WarnOnDefinedButUnused(*f.value(), *tm.type_info, warnings));
  ASSERT_EQ(warnings.warnings().size(), 1);
  EXPECT_EQ(warnings.warnings()[0].message,
            "Definition of `y` (type `uN[32]`) is not used in function `f`");
}

// This one looks like the arms could be making a binding but they are in fact
// doing an equality comparison as the names are already bound.
TEST(WarnOnDefinedButUnusedTest, MatchWithArmsDoingEqualityComparisonVsParams) {
  const std::string program = R"(fn f(x: bool, y: bool, z: bool) -> u32 {
  let u = u32:42;
  match true {
    x => u32:42,
    y => u32:43,
    z => u32:44,
    _ => u32:45
  }
})";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "test.x", "test.x", &import_data));
  WarningCollector warnings{kAllWarningsSet};
  std::optional<Function*> f = tm.module->GetFunction("f");
  ASSERT_TRUE(f.has_value());
  XLS_ASSERT_OK(WarnOnDefinedButUnused(*f.value(), *tm.type_info, warnings));
  EXPECT_EQ(warnings.warnings().size(), 1);
  EXPECT_EQ(warnings.warnings()[0].message,
            "Definition of `u` (type `uN[32]`) is not used in function `f`");
}

}  // namespace
}  // namespace xls::dslx
