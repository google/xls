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

#include "xls/codegen/combinational_generator.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/package.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "fast_hsv2rgb.h"

using xls::Value;
using xls::UBits;

class XlsColorsTest : public xls::IrTestBase {};

TEST_F(XlsColorsTest, Hsv2RgbTest) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path path,
      xls::GetXlsRunfilePath("third_party/xls_colors/hsv2rgb.x"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string moduleText, xls::GetFileContents(path));
  xls::dslx::ImportData import_data;
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::dslx::TypecheckedModule tm,
      xls::dslx::ParseAndTypecheck(moduleText, "hsv2rgb.x", "hsv2rgb", &import_data));
  const xls::dslx::ConvertOptions& options = xls::dslx::ConvertOptions{};
  XLS_ASSERT_OK_AND_ASSIGN(std::string package_text,
                           xls::dslx::ConvertModule(tm.module, &import_data, options));
  const uint8_t s = 255;
  const uint8_t v = 255;
  const uint16_t hstep = 8;
  for (uint16_t h = 0; h < 256 * 6; h+=hstep) {
    uint8_t r, g, b;
    fast_hsv2rgb_32bit(h, s, v, &r, &g, &b);
    RunAndExpectEq({{"h", Value(UBits(h, 16))}, {"s", Value(UBits(s, 8))}, {"v", Value(UBits(v, 8))}},
                   Value::Tuple({Value(UBits(r, 8)), Value(UBits(g, 8)), Value(UBits(b, 8))}),
                   package_text, true, true);
  }
}
