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

#include "xls/dslx/parse_and_typecheck.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

ImportData MakeFakeImportData(
    absl::flat_hash_map<std::filesystem::path, std::string> files,
    WarningKindSet warnings = kDefaultWarningsSet) {
  auto vfs = std::make_unique<FakeFilesystem>(std::move(files), "/root");
  return CreateImportData(
      /*stdlib_path=*/"/root/stdlib",
      /*additional_search_paths=*/{std::filesystem::path("/root")}, warnings,
      std::move(vfs));
}

absl::StatusOr<InterpValue> GetConstantValue(const TypecheckedModule& tm,
                                             std::string_view name) {
  XLS_ASSIGN_OR_RETURN(ConstantDef * cd, tm.module->GetConstantDef(name));
  return tm.type_info->GetConstExpr(cd->value());
}

TEST(ParseAndTypecheckTest,
     UnscopedConfiguredValueAppliesOnlyToEntryModuleNotImportedModule) {
  ImportData import_data = MakeFakeImportData({
      {"/root/dep_mod.x",
       R"(pub const DEP_VAL: u32 =
    configured_value_or<u32>("shared_key", 10);)"},
  });

  constexpr std::string_view kEntryProgram = R"(
import dep_mod;
pub const ENTRY_VAL: u32 = configured_value_or<u32>("shared_key", 20);
pub const IMPORTED_VAL: u32 = dep_mod::DEP_VAL;
)";

  ConvertOptions options;
  options.configured_values = {"shared_key:u32:99"};
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue entry_val,
                           GetConstantValue(tm, "ENTRY_VAL"));
  EXPECT_EQ(entry_val, InterpValue::MakeU32(99));

  // The imported module should keep its default value 10 because the
  // configured value was unscoped.
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue imported_val,
                           GetConstantValue(tm, "IMPORTED_VAL"));
  EXPECT_EQ(imported_val, InterpValue::MakeU32(10));
}

TEST(ParseAndTypecheckTest, ScopedConfiguredValueAppliesToImportedModule) {
  ImportData import_data = MakeFakeImportData({
      {"/root/dep_mod.x",
       R"(pub const DEP_VAL: u32 =
    configured_value_or<u32>("shared_key", 10);)"},
  });

  constexpr std::string_view kEntryProgram = R"(
import dep_mod;
pub const ENTRY_VAL: u32 = configured_value_or<u32>("shared_key", 20);
pub const IMPORTED_VAL: u32 = dep_mod::DEP_VAL;
)";

  ConvertOptions options;
  options.configured_values = {"shared_key@dep_mod:u32:77"};
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue entry_val,
                           GetConstantValue(tm, "ENTRY_VAL"));
  EXPECT_EQ(entry_val, InterpValue::MakeU32(20));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue imported_val,
                           GetConstantValue(tm, "IMPORTED_VAL"));
  EXPECT_EQ(imported_val, InterpValue::MakeU32(77));
}

TEST(ParseAndTypecheckTest,
     MultiModuleScopedConfiguredValueEmitsNoWarningWhenOneModuleUsesKey) {
  ImportData import_data = MakeFakeImportData({
      {"/root/mod_a.x",
       R"(pub const A_VAL: u32 =
    configured_value_or<u32>("shared_key", 1);)"},
      {"/root/mod_b.x", R"(pub const B_VAL: u32 = 2;)"},
  });

  constexpr std::string_view kEntryProgram = R"(
import mod_a;
import mod_b;
pub const RESULT: u32 = mod_a::A_VAL + mod_b::B_VAL;
)";

  ConvertOptions options;
  options.configured_values = {"shared_key@mod_a+mod_b:u32:40"};
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result_val,
                           GetConstantValue(tm, "RESULT"));
  EXPECT_EQ(result_val, InterpValue::MakeU32(42));
}

TEST(ParseAndTypecheckTest, ConflictingScopedValuesFailWhileDiamondsSucceed) {
  {
    ImportData import_data = MakeFakeImportData({
        {"/root/dep_mod.x",
         R"(pub const DEP_VAL: u32 = configured_value_or<u32>("k", 10);)"},
    });

    constexpr std::string_view kEntryProgram = R"(
import dep_mod;
pub const V: u32 = dep_mod::DEP_VAL;
)";

    ConvertOptions conflict_options;
    conflict_options.configured_values = {"k@dep_mod:u32:1", "k@dep_mod:u32:2"};
    EXPECT_THAT(
        ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod",
                          &import_data, /*comments=*/nullptr, conflict_options),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Conflict")));
  }

  {
    ImportData import_data = MakeFakeImportData({
        {"/root/dep_mod.x",
         R"(pub const DEP_VAL: u32 = configured_value_or<u32>("k", 10);)"},
    });

    constexpr std::string_view kEntryProgram = R"(
import dep_mod;
pub const V: u32 = dep_mod::DEP_VAL;
)";

    ConvertOptions diamond_options;
    diamond_options.configured_values = {"k@dep_mod:u32:88",
                                         "k@dep_mod:u32:88"};
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod",
                          &import_data, /*comments=*/nullptr, diamond_options));
    EXPECT_TRUE(tm.warnings.warnings().empty());
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue v, GetConstantValue(tm, "V"));
    EXPECT_EQ(v, InterpValue::MakeU32(88));
  }
}

TEST(ParseAndTypecheckTest,
     EntryModuleUnscopedOverridePrecedenceAndSurvivesTestFunctionTransformer) {
  ImportData import_data = MakeFakeImportData({});

  // Includes a proc-spawning #[test] function so that TestFunctionTransformer
  // rewrites and re-parses the module during TypecheckModule.
  constexpr std::string_view kEntryProgram = R"(
pub const K_VAL: u32 = configured_value_or<u32>("key", 0);

proc Counter {
  out_ch: chan<u32> out;
  init { () }
  config(out_ch: chan<u32> out) { (out_ch,) }
  next(state: ()) {
    let tok = send(join(), out_ch, K_VAL);
  }
}

#[test]
fn test_proc_spawning() {
  let (s, r) = chan<u32>("ch");
  spawn Counter(s);
  let (tok, v) = recv(join(), r);
  assert_eq(v, 55);
}
)";

  ConvertOptions options;
  options.configured_values = {"key@entry_mod:u32:10", "key:u32:55"};
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue k_val, GetConstantValue(tm, "K_VAL"));
  EXPECT_EQ(k_val, InterpValue::MakeU32(55));
}

TEST(ParseAndTypecheckTest, PartialImportOfMultiModuleScopeDoesNotWarn) {
  ImportData import_data = MakeFakeImportData({
      {"/root/mod_a.x",
       R"(pub const A_VAL: u32 =
    configured_value_or<u32>("shared_k", 1);)"},
      {"/root/mod_b.x", R"(pub const B_VAL: u32 = 2;)"},
  });

  // Entry module only imports mod_b (which does not use shared_k) while mod_a
  // (which uses shared_k) is not imported. No unused warning should be emitted.
  constexpr std::string_view kEntryProgram = R"(
import mod_b;
pub const RESULT: u32 = mod_b::B_VAL;
)";

  ConvertOptions options;
  options.configured_values = {"shared_k@mod_a+mod_b:u32:77"};
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram, "entry_mod.x", "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result_val,
                           GetConstantValue(tm, "RESULT"));
  EXPECT_EQ(result_val, InterpValue::MakeU32(2));
}

TEST(ParseAndTypecheckTest,
     SearchPathBasedNormalizationMatchesGeneratedAndWorkspacePaths) {
  auto vfs = std::make_unique<FakeFilesystem>(
      absl::flat_hash_map<std::filesystem::path, std::string>{
          {"/custom_out/arch_opt/gen_pkg/sub/dep_mod.x",
           R"(pub const DEP_VAL: u32 =
    configured_value_or<u32>("gen_key", 10);)"},
      },
      "/workspace");
  ImportData import_data = CreateImportData(
      /*stdlib_path=*/"/workspace/stdlib",
      /*additional_search_paths=*/
      {std::filesystem::path("/workspace"),
       std::filesystem::path("/custom_out/arch_opt")},
      kDefaultWarningsSet, std::move(vfs));

  constexpr std::string_view kEntryProgram = R"(
import gen_pkg.sub.dep_mod;
pub const ENTRY_VAL: u32 = configured_value_or<u32>("entry_key", 20);
pub const RESULT: u32 = dep_mod::DEP_VAL + ENTRY_VAL;
)";

  ConvertOptions options;
  options.configured_values = {
      "gen_key@gen_pkg.sub.dep_mod:u32:40",
      "entry_key@gen_pkg.sub.entry_mod:u32:2",
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kEntryProgram,
                        "/custom_out/arch_opt/gen_pkg/sub/entry_mod.x",
                        "entry_mod", &import_data,
                        /*comments=*/nullptr, options));

  EXPECT_TRUE(tm.warnings.warnings().empty());
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue result_val,
                           GetConstantValue(tm, "RESULT"));
  EXPECT_EQ(result_val, InterpValue::MakeU32(42));
}

}  // namespace
}  // namespace xls::dslx
