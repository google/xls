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

#include <optional>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
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

// Common infrastructure for IR converter tests.

inline constexpr std::string_view kProgramToVerifyTestConversion = R"(
#[cfg(test)]
fn test_utility_function() { trace_fmt!("Message from a test utility function"); }

fn normal_function() { trace_fmt!("Message from a normal function"); }

#[test]
fn test_func() {
    test_utility_function();
    normal_function();
    trace_fmt!("Message from a test function");
}

#[cfg(test)]
proc TestUtilityProc {
    req_r: chan<()> in;
    resp_s: chan<()> out;

    config(req_r: chan<()> in, resp_s: chan<()> out) { (req_r, resp_s) }

    init {  }

    next(state: ()) {
        let (tok, _) = recv(join(), req_r);
        trace_fmt!("Message from a TestUtilityProc");
        send(tok, resp_s, ());
    }
}

proc NormalProc {
    req_r: chan<()> in;
    resp_s: chan<()> out;

    config(req_r: chan<()> in, resp_s: chan<()> out) { (req_r, resp_s) }

    init {  }

    next(state: ()) {
        let (tok, _) = recv(join(), req_r);
        trace_fmt!("Message from a NormalProc");
        send(tok, resp_s, ());
    }
}

#[test_proc]
proc TestProc {
    terminator: chan<bool> out;
    tester_req_s: chan<()> out;
    tester_resp_r: chan<()> in;
    user_req_s: chan<()> out;
    user_resp_r: chan<()> in;

    config(terminator: chan<bool> out) {
        let (tester_req_s, tester_req_r) = chan<()>("tester_req");
        let (tester_resp_s, tester_resp_r) = chan<()>("tester_resp");

        spawn TestUtilityProc(tester_req_r, tester_resp_s);

        let (user_req_s, user_req_r) = chan<()>("user_req");
        let (user_resp_s, user_resp_r) = chan<()>("user_resp");

        spawn NormalProc(user_req_r, user_resp_s);

        (terminator, tester_req_s, tester_resp_r, user_req_s, user_resp_r)
    }

    init {  }

    next(state: ()) {
        let tok = send(join(), tester_req_s, ());
        let (tok, _) = recv(join(), tester_resp_r);

        let tok = send(join(), user_req_s, ());
        let (tok, _) = recv(join(), user_resp_r);

        trace_fmt!("Message from a TestProc");

        send(tok, terminator, true);
    }
}
)";

inline absl::StatusOr<TestResultData> ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) {
  return DslxInterpreterTestRunner().ParseAndTest(program, module_name,
                                                  filename, options);
}

inline constexpr ConvertOptions kProcScopedChannelOptions = {
    .emit_positions = false,
    .lower_to_proc_scoped_channels = true,
};

inline constexpr ConvertOptions kNoVerifyOptions = {
    .emit_positions = false,
    .verify_ir = false,
    .lower_to_proc_scoped_channels = true,
};

inline void ExpectIr(std::string_view got) {
  return ::xls::dslx::ExpectIr(got, TestName(), "ir_converter_test");
}

class IrConverterTest : public ::testing::Test {
 public:
  absl::StatusOr<std::string> ConvertOneFunctionForTest(
      std::string_view program, std::string_view fn_name,
      ImportData& import_data,
      const ConvertOptions& options = kProcScopedChannelOptions) {
    XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                         ::xls::dslx::ParseAndTypecheck(
                             program, /*path=*/"test_module.x",
                             /*module_name=*/"test_module", &import_data,
                             /*comments=*/nullptr));
    return ConvertOneFunction(tm.module, /*entry_function_name=*/fn_name,
                              &import_data,
                              /*parametric_env=*/nullptr, options);
  }

  absl::StatusOr<std::string> ConvertOneFunctionForTest(
      std::string_view program, std::string_view fn_name,
      const ConvertOptions& options = kProcScopedChannelOptions) {
    auto import_data = CreateImportDataForTest();
    return ConvertOneFunctionForTest(program, fn_name, import_data, options);
  }

  absl::StatusOr<std::string> ConvertModuleForTest(
      std::string_view program,
      const ConvertOptions& options = kProcScopedChannelOptions,
      ImportData* import_data = nullptr) {
    std::optional<ImportData> import_data_value;
    if (import_data == nullptr) {
      import_data_value.emplace(CreateImportDataForTest());
      import_data = &*import_data_value;
    }
    XLS_ASSIGN_OR_RETURN(
        TypecheckedModule tm,
        ::xls::dslx::ParseAndTypecheck(program, "test_module.x", "test_module",
                                       import_data,
                                       /*comments=*/nullptr, options));
    XLS_ASSIGN_OR_RETURN(std::string converted,
                         ConvertModule(tm.module, import_data, options));
    return converted;
  }

  absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
      std::string_view program, std::string_view path,
      std::string_view module_name, ImportData* import_data = nullptr) {
    return ::xls::dslx::ParseAndTypecheck(program, path, module_name,
                                          import_data,
                                          /*comments=*/nullptr);
  }
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_IR_CONVERTER_TEST_UTILS_H_
