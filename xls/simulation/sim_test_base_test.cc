// Copyright 2024 The XLS Authors
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

#include "xls/simulation/sim_test_base.h"

#include <memory>

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

class SimTestBaseTest : public SimTestBase {
 protected:
  static constexpr char kTestPackage[] = R"(
package test_package

top fn main(p: bits[8], q: bits[8]) -> bits[8] {
  add.1: bits[8] = add(p, q)
  ret add.2: bits[8] = add(add.1, q)
}
)";

  // Creates an invalid VerifiedPackage, then destroys it.
  void DestructInvalidVerifiedPackage() {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                             ParsePackage(kTestPackage));
    // Set an id to a duplicate value.
    FindNode("add.2", p.get())->SetId(1);
  }
};

TEST_F(SimTestBaseTest, RunAndExpectEqRightValue) {
  RunAndExpectEq({{"p", 3}, {"q", 10}}, 23, kTestPackage);
}

TEST_F(SimTestBaseTest, RunAndExpectEqWrongValue) {
  EXPECT_FATAL_FAILURE(RunAndExpectEq({{"p", 3}, {"q", 10}}, 55, kTestPackage),
                       "bits[8]:55 != bits[8]:23");
}

TEST_F(SimTestBaseTest, RunAndExpectEqArgDoesNotFit) {
  EXPECT_FATAL_FAILURE(
      RunAndExpectEq({{"p", 12345}, {"q", 10}}, 23, kTestPackage),
      "Argument value 12345 for parameter 'p' does not fit in type bits[8]");
}

TEST_F(SimTestBaseTest, RunAndExpectEqExpectedResultDoesNotFit) {
  EXPECT_FATAL_FAILURE(
      RunAndExpectEq({{"p", 3}, {"q", 10}}, 12345, kTestPackage),
      "Value 12345 does not fit in return type bits[8]");
}

TEST_F(SimTestBaseTest, RunAndExpectEqMissingArg) {
  EXPECT_FATAL_FAILURE(RunAndExpectEq({{"p", 3}}, 10, kTestPackage),
                       "Missing argument 'q'");
}

}  // namespace
}  // namespace xls
