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

#include "xls/common/source_location.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace xlscc {
namespace {

class SynthOnlyTest : public XlsccTestBase {};

TEST_F(SynthOnlyTest, CstdintStdNamespace) {
  const std::string content = R"(
    #include <cstdint>

    long long my_package(long long a) {
      std::int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content,
                    xabsl::SourceLocation::current());
}

TEST_F(SynthOnlyTest, StdintNotInStdNamespace) {
  const std::string content = R"(
    #include "stdint.h"

    long long my_package(long long a) {
      int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content,
                    xabsl::SourceLocation::current());
}

TEST_F(SynthOnlyTest, IntTypesIncludesStdint) {
  const std::string content = R"(
    #include "inttypes.h"

    long long my_package(long long a) {
      int64_t result = a;
      return result;
    })";
  RunAcDatatypeTest({{"a", 3}}, 3, content,
                    xabsl::SourceLocation::current());
}

}  // namespace

}  // namespace xlscc
