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

#include "xls/dslx/bytecode/interpreter_stack.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

TEST(InterpreterStackTest, PushThenDoublePop) {
  FileTable file_table;
  InterpreterStack stack(file_table);
  EXPECT_TRUE(stack.empty());
  stack.Push(InterpValue::MakeU32(42));
  EXPECT_FALSE(stack.empty());
  EXPECT_EQ(stack.size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue popped, stack.Pop());
  EXPECT_TRUE(popped.Eq(InterpValue::MakeU32(42)));
  EXPECT_TRUE(stack.empty());
  EXPECT_THAT(stack.Pop(),
              status_testing::StatusIs(absl::StatusCode::kInternal,
                                       "Tried to pop off an empty stack."));
}

}  // namespace
}  // namespace xls::dslx
