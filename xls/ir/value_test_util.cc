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

#include "xls/ir/value_test_util.h"

#include "gtest/gtest.h"
#include "xls/ir/value.h"

namespace xls {

::testing::AssertionResult ValuesEqual(const Value& a, const Value& b) {
  if (a != b) {
    return testing::AssertionFailure()
           << a.ToString() << " != " << b.ToString();
  }
  return testing::AssertionSuccess();
}

}  // namespace xls
