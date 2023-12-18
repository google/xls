// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/contrib/xlscc/unit_tests/cc_generator.h"

const char kUsage[] = R"(
Generates XLScc fuzz samples.
)";

ABSL_FLAG(int, seed, 1, "seed for pseudo-randomizer");
ABSL_FLAG(bool, test_ac_int, true, "tests will be run for ac_int");
ABSL_FLAG(bool, test_ac_fixed, false, "tests will be run for ac_fixed");
ABSL_FLAG(std::string, cc_filepath, "", "output file name");

namespace {

enum class TestType: uint8_t { kAcInt, kAcFixed };

static bool GenerateTest(int seed, const std::string& filename, TestType type) {
  std::string content = type == TestType::kAcInt ? xlscc::GenerateIntTest(seed)
                                       : xlscc::GenerateFixedTest(seed);
  std::cout << filename << '\n';
  absl::Status contents_set = xls::SetFileContents(filename, content);
  return contents_set.ok();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat(
        "Expected: %s -seed=1 -cc_filepath=xxx", argv[0]);
  }
  int seed = absl::GetFlag(FLAGS_seed);
  std::string cc_filepath = absl::GetFlag(FLAGS_cc_filepath);

  if (absl::GetFlag(FLAGS_test_ac_fixed) &&
      GenerateTest(seed, cc_filepath, TestType::kAcFixed)) {
    return EXIT_SUCCESS;
  }
  if (absl::GetFlag(FLAGS_test_ac_int) &&
      GenerateTest(seed, cc_filepath, TestType::kAcInt)) {
    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
