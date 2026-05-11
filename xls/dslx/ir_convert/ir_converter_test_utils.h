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

#ifndef XLS_DSLX_IR_CONVERT_IR_CONVERTER_TEST_UTILS_H_
#define XLS_DSLX_IR_CONVERT_IR_CONVERTER_TEST_UTILS_H_

#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {

absl::StatusOr<TestResultData> ParseAndTest(std::string_view program,
                                            std::string_view module_name,
                                            std::string_view filename,
                                            const ParseAndTestOptions& options);

void ExpectIr(std::string_view got);

constexpr ConvertOptions kProcScopedChannelOptions = {
    .emit_positions = false,
    .lower_to_proc_scoped_channels = true,
};

constexpr ConvertOptions kNoVerifyOptions = {
    .emit_positions = false,
    .verify_ir = false,
    .lower_to_proc_scoped_channels = true,
};

class IrConverterTest : public ::testing::Test {
 public:
  absl::StatusOr<std::string> ConvertOneFunctionForTest(
      std::string_view program, std::string_view fn_name,
      ImportData& import_data,
      const ConvertOptions& options = kProcScopedChannelOptions);

  absl::StatusOr<std::string> ConvertOneFunctionForTest(
      std::string_view program, std::string_view fn_name,
      const ConvertOptions& options = kProcScopedChannelOptions);

  absl::StatusOr<std::string> ConvertModuleForTest(
      std::string_view program,
      const ConvertOptions& options = kProcScopedChannelOptions,
      ImportData* import_data = nullptr);

  absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
      std::string_view program, std::string_view path,
      std::string_view module_name, ImportData* import_data = nullptr);
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_IR_CONVERTER_TEST_UTILS_H_
