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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace xlscc {
namespace {

class SynthOnlyTest : public XlsccTestBase {};

TEST_F(SynthOnlyTest, CstdintStdNamespace) {
  constexpr std::string_view content = R"(
    #include <cstdint>

    long long my_package(long long a) {
      std::int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(SynthOnlyTest, StdintNotInStdNamespace) {
  constexpr std::string_view content = R"(
    #include "stdint.h"

    long long my_package(long long a) {
      int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(SynthOnlyTest, IntTypesIncludesStdint) {
  constexpr std::string_view content = R"(
    #include "inttypes.h"

    long long my_package(long long a) {
      int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content, xabsl::SourceLocation::current());
}

TEST_F(SynthOnlyTest, StdioHTest) {
  constexpr std::string_view content = R"(
  #include <stdio.h>
    int my_package() {
      int a = 10;
      return a;
    })";
  RunAcDatatypeTest({}, 10, content);
}

TEST_F(SynthOnlyTest, StdintHTest) {
  constexpr std::string_view content = R"(
  #include <stdint.h>
    int my_package() {
      return INT32_MAX;
    })";
  RunAcDatatypeTest({}, 2147483647, content);
}

TEST_F(SynthOnlyTest, IomanipTest) {
  constexpr std::string_view content = R"(
  #include <iomanip>
  #include <iostream>
    int my_package() {
      std::cerr << std::setprecision(10) << std::endl;
      return 1;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::string> clang_args,
                           GetClangArgForIntTest());
  std::vector<std::string_view> clang_argv(clang_args.begin(),
                                           clang_args.end());
  absl::StatusOr<std::string> ret = SourceToIr(content, nullptr, clang_argv);
  ASSERT_THAT(ret.status(), xls::status_testing::StatusIs(
                                absl::StatusCode::kUnimplemented,
                                testing::HasSubstr("Unimplemented marker")));
}

TEST_F(SynthOnlyTest, StdintHUint64Max) {
  constexpr std::string_view content = R"(
  #include <stdint.h>
    int my_package() {
      return UINT64_MAX == 18446744073709551615UL;
    })";
  RunAcDatatypeTest({}, 1, content);
}

}  // namespace

}  // namespace xlscc
