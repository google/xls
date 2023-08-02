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

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/iactive_stub.h"

namespace xls::simulation::generic {
namespace {

class IActiveStubTest : public ::testing::Test {
 protected:
  IActiveStub stub;
};

TEST_F(IActiveStubTest, Init) { ASSERT_EQ(stub.getCnt(), 0); }

TEST_F(IActiveStubTest, Update) {
  for (auto i = 0; i < 100; ++i) {
    ASSERT_EQ(stub.getCnt(), i);
    XLS_EXPECT_OK(stub.Update());
  }
}

}  // namespace
}  // namespace xls::simulation::generic
