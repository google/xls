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

#include "xls/dslx/cpp_bindings.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

TEST(CppBindingsTest, ParseErrorGetData) {
  Pos start_pos("fake_file.x", 42, 64);
  Pos limit_pos("fake_file.x", 43, 65);
  Span span(start_pos, limit_pos);
  absl::Status status = ParseError(span, "my message");
  XLS_ASSERT_OK_AND_ASSIGN(auto got, ParseErrorGetData(status));
  const auto& [got_span, got_message] = got;
  EXPECT_EQ(got_span, span);
  EXPECT_EQ(got_message, "my message");
}

}  // namespace
}  // namespace xls::dslx
