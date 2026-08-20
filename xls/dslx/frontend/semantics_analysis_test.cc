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

#include "xls/dslx/frontend/semantics_analysis.h"

#include <filesystem>
#include <memory>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {
namespace {

TEST(SemanticsAnalysisTest, StatefulProcDoubleTypecheck) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]
proc Counter {
    state: u32,
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kProgram, "fake_path.x", "the_module", &import_data));

  // Run semantics analysis on the module again. This runs ProcStateVisitor.
  // It should succeed and not double-wrap the state member.
  WarningCollector warnings(import_data.enabled_warnings());
  auto dummy_typecheck =
      [](std::unique_ptr<Module>,
         std::filesystem::path) -> absl::StatusOr<std::unique_ptr<ModuleInfo>> {
    return absl::InternalError("Dummy typecheck should not be called");
  };
  SemanticsAnalysis semantics_analysis;
  XLS_EXPECT_OK(semantics_analysis.RunPreTypeCheckPass(
      *tm.module, warnings, import_data, dummy_typecheck));

  // Verify structurally that the member is not double-wrapped.
  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * proc,
                           tm.module->GetMemberOrError<ProcDef>("Counter"));
  ASSERT_EQ(proc->members().size(), 1);
  StructMemberNode* member = proc->members()[0];
  EXPECT_EQ(member->type()->ToString(), "State<u32>");
}

}  // namespace
}  // namespace xls::dslx
