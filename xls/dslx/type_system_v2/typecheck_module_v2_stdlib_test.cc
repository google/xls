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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

namespace xls::dslx {
namespace {

TEST(TypecheckV2StdlibTest, ImportStd) {
  XLS_EXPECT_OK(TypecheckV2("import std;"));
}

TEST(TypecheckV2StdlibTest, ImportAbsDiff) {
  // This is an auxiliary lib that imports std. Importing this one proves that
  // cross-module use of entities in std works.
  XLS_EXPECT_OK(TypecheckV2("import abs_diff;"));
}

TEST(TypecheckV2StdlibTest, UseFloat32Flatten) {
  // This proves that leakage of constants like F32_TOTAL_SZ doesn't cause
  // issues in a consuming module.
  EXPECT_THAT(R"(
import float32;
const X = float32::flatten(float32::zero(0));
  )",
              TypecheckSucceeds(HasNodeWithType("X", "uN[32]")));
}

}  // namespace
}  // namespace xls::dslx
