// Copyright 2020 The XLS Authors
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
#include "xls/solvers/z3_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

// Verifies that z3 boolean values are hexified correctly.
TEST(Z3UtilsTest, Hexifies) {
  std::string input_text =
      "This is some fun text. It has a boolean string, #b1010010110100101 .";
  std::string output_text = xls::solvers::z3::HexifyOutput(input_text);
  EXPECT_EQ(output_text,
            "This is some fun text. It has a boolean string, #xa5a5 .");
}

}  // namespace
