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

#include "xls/dslx/ir_convert/test_utils.h"

#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
#include "re2/re2.h"

namespace xls::dslx {

void ExpectIr(std::string_view actual_ir, std::string_view test_name,
              std::string_view prefix) {
  std::string test_name_without_param(test_name);
  RE2::GlobalReplace(&test_name_without_param, R"(/\d+)", "");
  ExpectEqualToGoldenFile(
      absl::StrFormat("xls/dslx/ir_convert/testdata/%s_%s.ir", prefix,
                      test_name_without_param),
      actual_ir);
}

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

}  // namespace xls::dslx
