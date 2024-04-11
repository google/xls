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

#include "xls/solvers/z3_ir_translator_matchers.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {
namespace {
using ::testing::Not;
using ::testing::StrEq;

TEST(ProverResultTest, Stringify) {
  EXPECT_EQ(absl::StrCat(ProverResult(ProvenTrue{})), "[ProvenTrue]");
  EXPECT_EQ(absl::StrCat(ProverResult(ProvenFalse{.message = "foobar"})),
            "[ProvenFalse: foobar]");
}

TEST(ProverResultTest, Matchers) {
  EXPECT_THAT(ProverResult(ProvenTrue{}), IsProvenTrue());
  EXPECT_THAT(ProverResult(ProvenFalse{.message = "foobar"}),
              Not(IsProvenTrue()));
  EXPECT_THAT(ProverResult(ProvenFalse{.message = "foobar"}), IsProvenFalse());
  EXPECT_THAT(ProverResult(ProvenFalse{.message = "foobar"}),
              IsProvenFalse(StrEq("foobar")));
  EXPECT_THAT(ProverResult(ProvenFalse{.message = "foobar"}),
              Not(IsProvenTrue()));
}

}  // namespace
}  // namespace xls::solvers::z3
