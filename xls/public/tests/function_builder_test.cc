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

#include "xls/public/function_builder.h"

#include "gtest/gtest.h"
#include "xls/public/ir.h"
#include "xls/public/status_matchers.h"

namespace {

TEST(FunctionBuilderTest, SimpleFunctionToIrText) {
  xls::Package package("test_package");
  xls::FunctionBuilder builder("f", &package);
  xls::BitsType* b32 = package.GetBitsType(32);
  xls::BValue x = builder.Param("x", b32);
  xls::BValue y = builder.Param("y", b32);
  builder.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, builder.Build());
  ASSERT_NE(f, nullptr);
  EXPECT_EQ(package.DumpIr(), R"(package test_package

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.3: bits[32] = add(x, y, id=3)
}
)");
}

}  // namespace
