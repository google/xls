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

#include "xls/common/proto_adaptor_utils.h"

#include <string>
#include <string_view>

#include "gtest/gtest.h"

namespace xls {

// Test the ToProtoString function.
TEST(ProtoAdaptorUtilsTest, ToProtoString) {
  const char* kString = "Test";
  std::string_view string_view_value = {kString};
  std::string string_result = ToProtoString(string_view_value);

  EXPECT_EQ(string_result, kString);
}

}  // namespace xls
