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

#include "xls/fuzzer/dslx_mutator.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/mock_distributions.h"
#include "absl/random/mocking_bit_gen.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"

using testing::_;
using testing::HasSubstr;
using testing::Return;
using xls::status_testing::IsOkAndHolds;
using xls::status_testing::StatusIs;

namespace xls::dslx {
namespace {

TEST(DslxMutator, RemoveDslxToken) {
  absl::MockingBitGen always_zero;
  ON_CALL(absl::MockUniform<int64_t>(), Call(always_zero, _, _))
      .WillByDefault(Return(0));

  EXPECT_THAT(RemoveDslxToken("", always_zero),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty")));
  EXPECT_THAT(RemoveDslxToken(":", always_zero), IsOkAndHolds(""));
  EXPECT_THAT(RemoveDslxToken("+5", always_zero), IsOkAndHolds("5"));

  absl::MockingBitGen always_two;
  ON_CALL(absl::MockUniform<int64_t>(), Call(always_two, _, _))
      .WillByDefault(Return(2));
  // "add" is the third token, it should be deleted.
  EXPECT_THAT(RemoveDslxToken("fn add(x: u32, y: u32) -> u32 {", always_two),
              IsOkAndHolds("fn (x: u32, y: u32) -> u32 {"));
}

}  // namespace
}  // namespace xls::dslx
