// The MIT License (MIT)
//
// Copyright (c) 2021 The XLS Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>

#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/simulation/sim_test_base.h"
#include "third_party/xls_colors/fast_hsv2rgb.h"

namespace {

using xls::UBits;
using xls::Value;

class XlsColorsTest : public xls::SimTestBase {
 public:
  void Run(uint16_t hstart, uint16_t hlimit) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::filesystem::path path,
        xls::GetXlsRunfilePath("third_party/xls_colors/hsv2rgb.x"));
    XLS_ASSERT_OK_AND_ASSIGN(std::string moduleText,
                             xls::GetFileContents(path));
    auto import_data = xls::dslx::CreateImportDataForTest();
    XLS_ASSERT_OK_AND_ASSIGN(
        xls::dslx::TypecheckedModule tm,
        xls::dslx::ParseAndTypecheck(moduleText, "hsv2rgb.x", "hsv2rgb",
                                     &import_data));
    const xls::dslx::ConvertOptions& options = xls::dslx::ConvertOptions{};
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string package_text,
        xls::dslx::ConvertOneFunction(tm.module, "hsv2rgb", &import_data,
                                      /*symbolic_bindings=*/nullptr, options));
    const uint8_t kS = 255;
    const uint8_t kV = 255;
    const uint16_t kHStep = 8;
    for (uint16_t h = hstart; h < hlimit; h += kHStep) {
      uint8_t r, g, b;
      fast_hsv2rgb_32bit(h, kS, kV, &r, &g, &b);
      RunAndExpectEq({{"h", Value(UBits(h, 16))},
                      {"s", Value(UBits(kS, 8))},
                      {"v", Value(UBits(kV, 8))}},
                     Value::Tuple({Value(UBits(r, 8)), Value(UBits(g, 8)),
                                   Value(UBits(b, 8))}),
                     package_text, true, true);
    }
  }
};

TEST_F(XlsColorsTest, Hsv2RgbTest0) { Run(256 * 0, 256 * 1); }
TEST_F(XlsColorsTest, Hsv2RgbTest1) { Run(256 * 1, 256 * 2); }
TEST_F(XlsColorsTest, Hsv2RgbTest2) { Run(256 * 2, 256 * 3); }
TEST_F(XlsColorsTest, Hsv2RgbTest3) { Run(256 * 3, 256 * 4); }
TEST_F(XlsColorsTest, Hsv2RgbTest4) { Run(256 * 4, 256 * 5); }
TEST_F(XlsColorsTest, Hsv2RgbTest5) { Run(256 * 5, 256 * 6); }

}  // namespace
