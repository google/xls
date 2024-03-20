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

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/generic/iirq_stub.h"

namespace xls::simulation::generic {
namespace {

class IIRQStubTest : public ::testing::Test {
 protected:
  IIRQStub stub;
};

TEST_F(IIRQStubTest, Init) {
  // stub should not request any interrupt after initialization
  ASSERT_EQ(stub.GetIRQ(), false);
}

TEST_F(IIRQStubTest, UpdateIRQ) {
  // with the default stub implementation,
  // UpdateIRQ should change the internal interrupt state from false to true
  ASSERT_EQ(stub.GetIRQ(), false);
  XLS_EXPECT_OK(stub.UpdateIRQ());
  EXPECT_EQ(stub.GetIRQ(), true);
}

TEST_F(IIRQStubTest, GetIRQ) {
  // consecutive reads using GetIRQ should have the same value
  ASSERT_EQ(stub.GetIRQ(), false);
  EXPECT_EQ(stub.GetIRQ(), false);
  EXPECT_EQ(stub.GetIRQ(), false);
  XLS_EXPECT_OK(stub.UpdateIRQ());
  EXPECT_EQ(stub.GetIRQ(), true);
  EXPECT_EQ(stub.GetIRQ(), true);
  EXPECT_EQ(stub.GetIRQ(), true);
}

TEST_F(IIRQStubTest, SetPolicy) {
  // stub should be able to change the existing policy
  bool prev = false;
  std::function<bool(void)> alternate_policy = [&prev] {
    prev = !prev;
    return prev;
  };

  stub.SetPolicy(alternate_policy);
  bool prev_policy = stub.GetIRQ();
  for (int i = 0; i < 100; ++i) {
    XLS_EXPECT_OK(stub.UpdateIRQ());
    EXPECT_EQ(stub.GetIRQ(), !prev_policy);
    prev_policy = stub.GetIRQ();
  }
}

}  // namespace
}  // namespace xls::simulation::generic
