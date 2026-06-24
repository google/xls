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

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/semantics_analysis.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/typecheck_module_v2.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

TEST(TypecheckV2WarningTest,
     CanonicalizationPreservesDisabledSemanticAnalysis) {
  constexpr std::string_view kProgram = R"(#![feature(type_inference_v2)]
enum E { V(u32) }
fn main() -> E {
  let unused = u32:0;
  E::V(u32:1)
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "main.x", "main",
                                                    import_data.file_table()));
  WarningCollector warnings(import_data.enabled_warnings());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module_info,
      TypecheckModuleV2(std::move(module), "main.x", &import_data, &warnings,
                        /*semantics_analysis=*/nullptr,
                        /*error_handler=*/nullptr,
                        /*trait_deriver=*/std::nullopt));
  auto main = module_info->module().GetFunction("main");
  ASSERT_TRUE(main.has_value());
  EXPECT_NE(dynamic_cast<SumInstance*>(
                ToAstNode((*main)->body()->statements().back()->wrapped())),
            nullptr);
  EXPECT_TRUE(warnings.warnings().empty());
}

TEST(TypecheckV2WarningTest, ImportedGenericWarningAfterModuleTypecheck) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
pub fn f<N: u32>(x: uN[N]) -> u1 {
  x[N+:u1]
}
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));

  // Loading the import and instantiating f happen in one call, with the
  // caller's collector alive throughout. The imported module has already been
  // checked when f's concrete width causes its warning.
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(
import imported;
fn main() -> u1 { imported::f(u8:0) }
)",
                                       "main", &import_data));
  const auto& warnings = result.tm.warnings.warnings();
  ASSERT_EQ(warnings.size(), 1);
  EXPECT_EQ(warnings[0].kind, WarningKind::kWidthSliceOutOfRange);
  EXPECT_EQ(warnings[0].message,
            "Slice range out of bounds for array of size 8");
  EXPECT_EQ(warnings[0].span.GetFilename(import_data.file_table()),
            "imported.x");
  EXPECT_EQ(warnings[0].span.start().lineno(), 2);
  EXPECT_EQ(warnings[0].span.start().colno(), 3);
  EXPECT_EQ(warnings[0].span.limit().lineno(), 2);
  EXPECT_EQ(warnings[0].span.limit().colno(), 9);
}

TEST(TypecheckV2WarningTest,
     CanonicalizationPreservesDistinctAndEarlierWarnings) {
  constexpr std::string_view kImported = R"(#![feature(type_inference_v2)]
pub const TOO_FAR = (u6:0)[6+:u1];
pub fn f<N: u32>(x: uN[N]) -> u1 { x[N+:u1] }
)";
  constexpr std::string_view kProgram = R"(#![feature(type_inference_v2)]
import imported;
enum E { V(u32) }
fn main() -> E {
  let unused = u32:0;
  let _a = imported::f(u8:0);
  let _b = imported::f(u16:0);
  E::V(u32:0)
}
)";
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {"/imported.x", std::string(kImported)},
  };
  ImportData import_data = CreateImportDataForTest(
      std::make_unique<FakeFilesystem>(std::move(files), "/"));
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "main.x", "main",
                                                    import_data.file_table()));
  WarningCollector warnings(import_data.enabled_warnings());
  const Span prefix_span = module->span();
  warnings.Add(prefix_span, WarningKind::kWidthSliceOutOfRange, "earlier");
  warnings.Add(prefix_span, WarningKind::kWidthSliceOutOfRange, "earlier");
  XLS_ASSERT_OK_AND_ASSIGN(
      auto module_info,
      TypecheckModuleV2(std::move(module), "main.x", &import_data, &warnings,
                        std::make_unique<SemanticsAnalysis>(), nullptr,
                        std::nullopt));
  auto main = module_info->module().GetFunction("main");
  ASSERT_TRUE(main.has_value());
  ASSERT_NE(dynamic_cast<SumInstance*>(
                ToAstNode((*main)->body()->statements().back()->wrapped())),
            nullptr);

  const auto& entries = warnings.warnings();
  ASSERT_EQ(entries.size(), 6);
  EXPECT_EQ(entries[0].span, prefix_span);
  EXPECT_EQ(entries[1].span, prefix_span);
  EXPECT_EQ(entries[0].message, "earlier");
  EXPECT_EQ(entries[1].message, "earlier");
  std::vector<std::string> messages;
  for (size_t i = 2; i < entries.size(); ++i) {
    messages.push_back(entries[i].message);
    if (entries[i].kind == WarningKind::kWidthSliceOutOfRange) {
      EXPECT_EQ(entries[i].span.GetFilename(import_data.file_table()),
                "imported.x");
    }
  }
  // The import's constant warning cannot be reconstructed from a cached import.
  // Instantiating f at two widths must keep both messages at the same span.
  // The final module's unused-definition warning must appear exactly once.
  EXPECT_THAT(messages, UnorderedElementsAre(
                            "Slice range out of bounds for array of size 6",
                            "Slice range out of bounds for array of size 8",
                            "Slice range out of bounds for array of size 16",
                            HasSubstr("unused")));
}

}  // namespace
}  // namespace xls::dslx
