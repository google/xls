// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system/deduce_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {
namespace {

TEST(DeduceUtilsTest, ValidateNumber) {
  Scanner scanner("test.x", "42 256");
  Parser parser("test", &scanner);
  Bindings bindings;
  XLS_ASSERT_OK_AND_ASSIGN(Expr * e, parser.ParseExpression(bindings));
  auto* ft = down_cast<Number*>(e);

  XLS_ASSERT_OK_AND_ASSIGN(e, parser.ParseExpression(bindings));
  auto* tfs = down_cast<Number*>(e);

  auto u8 = BitsType::MakeU8();
  XLS_ASSERT_OK(ValidateNumber(*ft, *u8));

  // 256 does not fit in a u8.
  ASSERT_THAT(ValidateNumber(*tfs, *u8),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Value '256' does not fit in the bitwidth of a uN[8]")));
}

}  // namespace
}  // namespace xls::dslx
