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

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/contrib/xlscc/cc_generator.h"

const char kUsage[] = R"(
Generates XLScc fuzz samples.
)";

ABSL_FLAG(int, seed, 1, "seed for pseudo-randomizer");
ABSL_FLAG(std::string, cc_filepath, "", "output file name");

namespace {

static bool GenerateTest(int seed, const std::string& filename) {
  std::string content = xlscc::GenerateIntTest(seed);
  std::cout << filename << std::endl;
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

  if (GenerateTest(seed, cc_filepath)) {
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
