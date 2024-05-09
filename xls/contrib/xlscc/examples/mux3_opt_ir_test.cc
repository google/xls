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

#include <cstdio>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xlscc {
namespace {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

constexpr const char kIrPath[] =
    "xls/contrib/xlscc/examples/mux3_opt_ir.opt.ir";

TEST(Mux3IrTest, BasicSignaturePresent) {
  XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                           xls::GetXlsRunfilePath(kIrPath));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, xls::GetFileContents(ir_path));
  XLS_VLOG_LINES(2, ir_text);

  EXPECT_THAT(ir_text,
              ContainsRegex(
                  R"(chan csrs\(\(bits\[1\], bits\[1\]\),.*ops=receive_only)"));
  EXPECT_THAT(ir_text,
              ContainsRegex(R"(chan mux_in0\(\(bits\[8\]\),.*)"
                            R"(ops=receive_only,.*flow_control=ready_valid)"));
  EXPECT_THAT(ir_text,
              ContainsRegex(R"(chan mux_in1\(\(bits\[8\]\),.*)"
                            R"(ops=receive_only,.*flow_control=ready_valid)"));
  EXPECT_THAT(ir_text,
              ContainsRegex(R"(chan mux_in2\(\(bits\[8\]\),.*)"
                            R"(ops=receive_only,.*flow_control=ready_valid)"));
  EXPECT_THAT(ir_text,
              ContainsRegex(R"(chan mux_out\(\(bits\[8\]\),.*)"
                            R"(ops=send_only,.*flow_control=ready_valid)"));
  EXPECT_THAT(ir_text, HasSubstr(R"(proc Mux3_proc())"));
}

}  // namespace
}  // namespace xlscc
