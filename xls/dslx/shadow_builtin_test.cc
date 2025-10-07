// Copyright 2023 The XLS Authors
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

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/run_routines/run_routines.h"

namespace xls::dslx {
namespace {

absl::StatusOr<TestResultData> ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) {
  return DslxInterpreterTestRunner().ParseAndTest(program, module_name,
                                                  filename, options);
}

template <typename K, typename V>
std::vector<K> MapKeysSorted(const absl::flat_hash_map<K, V>& m) {
  std::vector<K> keys;
  for (const auto& [k, v] : m) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

TEST(ShadowBuiltinTest, ShadowBuiltinsWithFnDefn) {
  const auto& builtins_map = GetParametricBuiltins();
  std::vector<std::string> builtins_names = MapKeysSorted(builtins_map);

  constexpr const char kTemplate[] = R"(fn %s(x: u32) -> u32 {
  x
}

fn main(x: u32) -> u32 {
  %s(x)
}

#[test]
fn test_main() {
  assert_eq(main(u32:42), u32:42)
})";
  const std::string kFakePath = "/path/to/fake.x";

  for (const std::string& name : builtins_names) {
    if (name == "assert_eq") {
      continue;  // Used in our expectation.
    }
    if (name == "token") {
      continue;  // Type name, cannot be shadowed.
    }
    LOG(INFO) << "builtin: " << name;
    std::string program = absl::StrFormat(kTemplate, name, name);
    ParseAndTestOptions options;
    XLS_ASSERT_OK_AND_ASSIGN(
        TestResultData test_result,
        ParseAndTest(program, /*module_name=*/"fake", kFakePath, options));
    EXPECT_EQ(test_result.result(), TestResult::kAllPassed);
  }
}

TEST(ShadowBuiltinTest, ShadowBuiltinsWithLetBinding) {
  const auto& builtins_map = GetParametricBuiltins();
  std::vector<std::string> builtins_names = MapKeysSorted(builtins_map);

  constexpr const char kTemplate[] = R"(fn id(x: u32) -> u32 {
  let %s = x;
  %s
}

fn main(x: u32) -> u32 {
  id(x)
}

#[test]
fn test_main() {
  assert_eq(main(u32:42), u32:42)
})";
  const std::string kFakePath = "/path/to/fake.x";

  for (const std::string& name : builtins_names) {
    if (name == "assert_eq") {
      continue;  // Used in our expectation.
    }
    if (name == "token") {
      continue;  // Type name, cannot be shadowed.
    }
    LOG(INFO) << "builtin: " << name;
    std::string program = absl::StrFormat(kTemplate, name, name);
    ParseAndTestOptions options;
    XLS_ASSERT_OK_AND_ASSIGN(
        TestResultData test_result,
        ParseAndTest(program, /*module_name=*/"fake", kFakePath, options));
    EXPECT_EQ(test_result.result(), TestResult::kAllPassed);
  }
}

}  // namespace
}  // namespace xls::dslx
