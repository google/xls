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

#ifndef XLS_IR_VALUE_TEST_UTIL_H_
#define XLS_IR_VALUE_TEST_UTIL_H_

#include "gtest/gtest.h"
#include "xls/ir/value.h"

namespace xls {

// Returns an assertion result indicating whether the given two values were
// equal. If equal the return value is AssertionSuccess, otherwise
// AssertionFailure. For large Values (arrays, tuples, and very wide bit widths)
// this method can give a more readable message than EXPECT_EQ(a, b) by
// highlighting the differences between the two Values rather than just dumping
// the string representation of the Values.
::testing::AssertionResult ValuesEqual(const Value& a, const Value& b);

}  // namespace xls

#endif  // XLS_IR_VALUE_TEST_UTIL_H_
