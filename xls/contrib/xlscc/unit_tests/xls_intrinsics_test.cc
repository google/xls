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

#include <string>

#include "gtest/gtest.h"
#include "xls/common/source_location.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace xlscc {
namespace {

class XlsIntrinsicsTest : public XlsccTestBase {};

TEST_F(XlsIntrinsicsTest, Clz) {
  const std::string content = R"(
    #include <xls_int.h>
    #include <xls_intrinsics.h>

    unsigned long long my_package(unsigned long long a) {
      XlsInt<8, false> ax = a;
      return xls_intrinsics::clz(ax);
    })";
  RunAcDatatypeTest({{"a", 0b1000}}, 4, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0b10000000}}, 0, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0}}, 8, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 7, content, xabsl::SourceLocation::current());
}

TEST_F(XlsIntrinsicsTest, Ctz) {
  const std::string content = R"(
    #include <xls_int.h>
    #include <xls_intrinsics.h>

    unsigned long long my_package(unsigned long long a) {
      XlsInt<8, false> ax = a;
      return xls_intrinsics::ctz(ax);
    })";
  RunAcDatatypeTest({{"a", 0b1000}}, 3, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0b10000000}}, 7, content,
                    xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 0}}, 8, content, xabsl::SourceLocation::current());
  RunAcDatatypeTest({{"a", 1}}, 0, content, xabsl::SourceLocation::current());
}

}  // namespace

}  // namespace xlscc
