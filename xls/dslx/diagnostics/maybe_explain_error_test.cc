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

#include "xls/dslx/diagnostics/maybe_explain_error.h"

#include "absl/log/log.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_module.h"

namespace xls::dslx {
namespace {

using ::testing::HasSubstr;

void TypecheckAndExplain(std::string_view program,
                         std::string_view want_substr) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Module> module,
      ParseModule(program, "test.x", "test", import_data.file_table()));
  ASSERT_EQ(import_data.file_table().Get(Fileno(1)), "test.x");

  // Scaffolding for typechecking.
  TypeInfoOwner type_info_owner;
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfo * type_info,
                           type_info_owner.New(module.get()));
  WarningCollector warnings(kAllWarningsSet);
  DeduceCtx ctx(/*type_info=*/type_info, /*module=*/module.get(),
                /*deduce_function=*/&Deduce,
                /*typecheck_function=*/nullptr, /*typecheck_module=*/nullptr,
                /*typecheck_invocation=*/nullptr, /*import_data=*/&import_data,
                /*warnings=*/&warnings, /*parent=*/nullptr);
  ctx.AddFnStackEntry(FnStackEntry::MakeTop(module.get()));

  // Typecheck the first module member.
  ASSERT_EQ(module->top().size(), 1);
  const ModuleMember& member = module->top()[0];
  absl::Status status = typecheck_internal::TypecheckModuleMember(
      member, module.get(), &import_data, &ctx);

  // Ensure that we got a type mismatch error and extract the data for it.
  ASSERT_FALSE(status.ok());
  ASSERT_TRUE(typecheck_internal::IsTypeMismatchStatus(status));
  const std::optional<TypeMismatchErrorData>& data =
      ctx.type_mismatch_error_data();
  ASSERT_TRUE(data.has_value());
  const TypeMismatchErrorData& mismatch_data = data.value();

  // Explain the error via the diagnostic machinery.
  absl::Status explained = MaybeExplainError(mismatch_data, ctx.file_table());
  ASSERT_FALSE(explained.ok());
  EXPECT_THAT(explained.message(), HasSubstr(want_substr));
}

TEST(MaybeExplainErrorTest, ExplainIfBlockWithTrailingSemi) {
  constexpr std::string_view kProgram = R"(
    fn f() -> u32 {
        let x = {
            u32:42;
        };
        x + u32:1
    }
)";
  TypecheckAndExplain(
      kProgram,
      "note that \"x\" was defined by a block with a trailing semicolon");
}

TEST(MaybeExplainErrorTest,
     ExplainIfOneOfConditionalArmsIsUnitWithTrailingSemi) {
  constexpr std::string_view kProgram = R"(fn f() -> u32 {
    let x: u32 = if true {
        u32:42;
    } else {
        u32:1
    };
    x + u32:1
}
)";
  TypecheckAndExplain(
      kProgram,
      "note that conditional block @ test.x:2:26-4:6 had a trailing semicolon");
}

}  // namespace
}  // namespace xls::dslx