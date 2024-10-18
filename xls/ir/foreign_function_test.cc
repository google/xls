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

#include "xls/ir/foreign_function.h"

#include "gtest/gtest.h"
#include "xls/ir/bits.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/value.h"

namespace xls {
namespace {
TEST(ForeignFunctionTest, PartialValueSubstituteHelper) {
  ForeignFunctionData ffi;
  ffi.set_code_template("Some {foo} at {bar} with {baz}");
  // Setting some non-zero delay to make sure it is preserved.
  ffi.set_delay_ps(1536);
  FfiPartialValueSubstituteHelper substitute(ffi);
  substitute.SetNamedValue("foo", Value(UBits(0xf00d, 16)));
  // '{bar}', we want to keep as-is.
  substitute.SetNamedValue("baz", Value(UBits(0xc0ffee, 24)));
  EXPECT_EQ(substitute.GetUpdatedFfiData()->code_template(),
            "Some 16'hf00d at {bar} with 24'hc0ffee");
  EXPECT_EQ(ffi.delay_ps(), 1536);
  EXPECT_EQ(substitute.GetUpdatedFfiData()->delay_ps(), 1536);
}
}  // namespace
}  // namespace xls
