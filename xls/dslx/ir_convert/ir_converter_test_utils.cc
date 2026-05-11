// Copyright 2021 The XLS Authors
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

#include "xls/dslx/ir_convert/ir_converter_test_utils.h"

#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/ir_convert/test_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {

absl::StatusOr<TestResultData> ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) {
  // Other interpreters rely on ir_convert so we can't test with them.
  return DslxInterpreterTestRunner().ParseAndTest(program, module_name,
                                                  filename, options);
}

void ExpectIr(std::string_view got) {
  return ::xls::dslx::ExpectIr(got, TestName(), "ir_converter_test");
}

absl::StatusOr<std::string> IrConverterTest::ConvertOneFunctionForTest(
    std::string_view program, std::string_view fn_name, ImportData& import_data,
    const ConvertOptions& options) {
  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ::xls::dslx::ParseAndTypecheck(
                           program, /*path=*/"test_module.x",
                           /*module_name=*/"test_module", &import_data,
                           /*comments=*/nullptr));
  return ConvertOneFunction(tm.module, /*entry_function_name=*/fn_name,
                            &import_data,
                            /*parametric_env=*/nullptr, options);
}

absl::StatusOr<std::string> IrConverterTest::ConvertOneFunctionForTest(
    std::string_view program, std::string_view fn_name,
    const ConvertOptions& options) {
  auto import_data = CreateImportDataForTest();
  return ConvertOneFunctionForTest(program, fn_name, import_data, options);
}

absl::StatusOr<std::string> IrConverterTest::ConvertModuleForTest(
    std::string_view program, const ConvertOptions& options,
    ImportData* import_data) {
  std::optional<ImportData> import_data_value;
  if (import_data == nullptr) {
    import_data_value.emplace(CreateImportDataForTest());
    import_data = &*import_data_value;
  }
  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ::xls::dslx::ParseAndTypecheck(
                           program, "test_module.x", "test_module", import_data,
                           /*comments=*/nullptr, options));
  XLS_ASSIGN_OR_RETURN(std::string converted,
                       ConvertModule(tm.module, import_data, options));
  return converted;
}

absl::StatusOr<TypecheckedModule> IrConverterTest::ParseAndTypecheck(
    std::string_view program, std::string_view path,
    std::string_view module_name, ImportData* import_data) {
  return ::xls::dslx::ParseAndTypecheck(program, path, module_name, import_data,
                                        /*comments=*/nullptr);
}

}  // namespace xls::dslx
