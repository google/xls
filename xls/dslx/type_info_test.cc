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

#include "xls/dslx/type_info.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(TypeInfoTest, Instantiate) {
  auto module = std::make_shared<Module>("test");
  TypeInfo type_info(module);
  EXPECT_EQ(type_info.parent(), nullptr);
}

}  // namespace
}  // namespace xls::dslx
