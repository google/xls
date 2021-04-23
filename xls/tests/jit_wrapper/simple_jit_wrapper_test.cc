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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/value_helpers.h"
#include "xls/tests/jit_wrapper/fail_on_42.h"
#include "xls/tests/jit_wrapper/identity.h"
#include "xls/tests/jit_wrapper/is_inf.h"

namespace xls {
namespace {

using xls::status_testing::IsOkAndHolds;

TEST(SimpleJitWrapperTest, InvokeIdentity) {
  constexpr float kInput = 1.0;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<F32Identity> f,
                           F32Identity::Create());
  XLS_ASSERT_OK_AND_ASSIGN(float output, f->Run(kInput));
  EXPECT_EQ(output, kInput);

  Value input = F32ToTuple(kInput);
  XLS_ASSERT_OK_AND_ASSIGN(Value output_value, f->Run(input));
  EXPECT_EQ(output_value, input);
}

TEST(SimpleJitWrapperTest, IsInf) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<F32IsInf> f, F32IsInf::Create());
  EXPECT_THAT(f->Run(1.0), IsOkAndHolds(false));
  EXPECT_THAT(f->Run(F32ToTuple(1.0)), IsOkAndHolds(Value(UBits(false, 1))));

  EXPECT_THAT(f->Run(std::numeric_limits<float>::infinity()),
              IsOkAndHolds(true));
  EXPECT_THAT(f->Run(F32ToTuple(std::numeric_limits<float>::infinity())),
              IsOkAndHolds(Value(UBits(true, 1))));
}

TEST(SimpleJitWrapperTest, FailOn42) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<FailOn42> f, FailOn42::Create());

  EXPECT_THAT(f->Run(0), IsOkAndHolds(0));
  EXPECT_THAT(f->Run(Value(UBits(0, 32))), IsOkAndHolds(Value(UBits(0, 32))));

  EXPECT_THAT(f->Run(1), IsOkAndHolds(1));
  EXPECT_THAT(f->Run(Value(UBits(1, 32))), IsOkAndHolds(Value(UBits(1, 32))));

  // TODO(https://github.com/google/xls/issues/232): 2021-04-21 When DSL
  // translates `fail!()` to IR as an assert op this will become an error
  // status.
  EXPECT_THAT(f->Run(42), IsOkAndHolds(42));
  EXPECT_THAT(f->Run(Value(UBits(42, 32))), IsOkAndHolds(Value(UBits(42, 32))));
}

}  // namespace
}  // namespace xls
