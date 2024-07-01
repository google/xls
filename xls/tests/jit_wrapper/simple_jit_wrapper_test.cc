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

#include <cstdint>
#include <limits>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/ir/value_view.h"
#include "xls/tests/jit_wrapper/fail_on_42.h"
#include "xls/tests/jit_wrapper/identity.h"
#include "xls/tests/jit_wrapper/is_inf.h"
#include "xls/tests/jit_wrapper/make_tuple.h"
#include "xls/tests/jit_wrapper/wide_identity.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(SimpleJitWrapperTest, InvokeIdentity) {
  constexpr float kInput = 1.0;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::F32Identity> f,
                           xls::test::F32Identity::Create());
  XLS_ASSERT_OK_AND_ASSIGN(float output, f->Run(kInput));
  EXPECT_EQ(output, kInput);

  Value input = F32ToTuple(kInput);
  XLS_ASSERT_OK_AND_ASSIGN(Value output_value, f->Run(input));
  EXPECT_EQ(output_value, input);
}

TEST(SimpleJitWrapperTest, InvokeWideIdentity) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::WideIdentity> f,
                           xls::test::WideIdentity::Create());

  uint8_t input[64];
  uint8_t output[64];
  for (uint8_t i = 0; i < 64; ++i) {
    input[i] = i;
    output[i] = 0;
  }

  BitsView<512> input_view(input);
  MutableBitsView<512> output_view(output);
  XLS_ASSERT_OK(f->Run(input_view, output_view));

  EXPECT_THAT(output, testing::ElementsAreArray(input));
}

TEST(SimpleJitWrapperTest, IsInf) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::F32IsInf> f,
                           xls::test::F32IsInf::Create());
  EXPECT_THAT(f->Run(1.0), IsOkAndHolds(false));
  EXPECT_THAT(f->Run(F32ToTuple(1.0)), IsOkAndHolds(Value(UBits(false, 1))));

  EXPECT_THAT(f->Run(std::numeric_limits<float>::infinity()),
              IsOkAndHolds(true));
  EXPECT_THAT(f->Run(F32ToTuple(std::numeric_limits<float>::infinity())),
              IsOkAndHolds(Value(UBits(true, 1))));
}

TEST(SimpleJitWrapperTest, FailOn42) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::FailOn42> f,
                           xls::test::FailOn42::Create());

  EXPECT_THAT(f->Run(0), IsOkAndHolds(0));
  EXPECT_THAT(f->Run(Value(UBits(0, 32))), IsOkAndHolds(Value(UBits(0, 32))));

  EXPECT_THAT(f->Run(1), IsOkAndHolds(1));
  EXPECT_THAT(f->Run(Value(UBits(1, 32))), IsOkAndHolds(Value(UBits(1, 32))));

  EXPECT_THAT(f->Run(42), StatusIs(absl::StatusCode::kAborted,
                                   HasSubstr("Assertion failure via fail!")));
  EXPECT_THAT(f->Run(Value(UBits(42, 32))),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("Assertion failure via fail!")));
}

TEST(SimpleJitWrapperTest, InvokeMakeTuple) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::MakeTuple> f,
                           xls::test::MakeTuple::Create());

  // Run using views.
  int64_t x = 0x1;
  int64_t y = 0x34;
  uint8_t result[4];

  xls::PackedBitsView<1> x_view(reinterpret_cast<uint8_t*>(&x), 0);
  xls::PackedBitsView<8> y_view(reinterpret_cast<uint8_t*>(&y), 0);
  xls::PackedTupleView<xls::PackedBitsView<1>, xls::PackedBitsView<8>,
                       xls::PackedBitsView<16>>
      result_view(result, 0);

  XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

  int64_t out_0 = 0;
  int64_t out_1 = 0;
  int64_t out_2 = 0;

  result_view.Get<0>().Get(reinterpret_cast<uint8_t*>(&out_0));
  result_view.Get<1>().Get(reinterpret_cast<uint8_t*>(&out_1));
  result_view.Get<2>().Get(reinterpret_cast<uint8_t*>(&out_2));

  EXPECT_EQ(out_0, 0x1);
  EXPECT_EQ(out_1, 0x34);
  EXPECT_EQ(out_2, 0xabcd);
  EXPECT_THAT(result, testing::ElementsAreArray({0xcd, 0xab, 0x34, 0x1}));
}

}  // namespace
}  // namespace xls
