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

#include "xls/dslx/type_system/parametric_expression.h"

#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

const Pos kFakePos("<fake>", 0, 0);
const Span kFakeSpan(kFakePos, kFakePos);

TEST(ParametricExpressionTest, SampleEvaluation) {
  auto param_0 = InterpValue::MakeUBits(32, 0);
  auto param_1 = InterpValue::MakeUBits(32, 1);
  auto param_2 = InterpValue::MakeUBits(32, 2);
  auto param_3 = InterpValue::MakeUBits(32, 3);
  auto param_6 = InterpValue::MakeUBits(32, 6);
  auto param_12 = InterpValue::MakeUBits(32, 12);
  auto e = std::make_unique<ParametricMul>(
      std::make_unique<ParametricConstant>(param_3),
      std::make_unique<ParametricAdd>(
          std::make_unique<ParametricSymbol>("M", kFakeSpan),
          std::make_unique<ParametricSymbol>("N", kFakeSpan)));
  EXPECT_EQ(*e, *e);
  EXPECT_EQ(e->ToString(), "(u32:3*(M+N))");
  EXPECT_EQ(param_6, std::get<InterpValue>(
                         e->Evaluate({{"N", param_2}, {"M", param_0}})));
  EXPECT_EQ(param_12, std::get<InterpValue>(
                          e->Evaluate({{"N", param_1}, {"M", param_3}})));
  const absl::flat_hash_set<std::string> want_freevars = {"N", "M"};
  EXPECT_EQ(e->GetFreeVariables(), want_freevars);
}

TEST(ParametricExpressionTest, TestNonIdentityEquality) {
  auto s0 = std::make_unique<ParametricSymbol>("s", kFakeSpan);
  auto s1 = std::make_unique<ParametricSymbol>("s", kFakeSpan);
  EXPECT_EQ(*s0, *s1);
  EXPECT_EQ(s0->ToRepr(), "ParametricSymbol(\"s\")");
  auto add = std::make_unique<ParametricAdd>(std::move(s0), std::move(s1));
  EXPECT_EQ(add->ToRepr(),
            "ParametricAdd(ParametricSymbol(\"s\"), ParametricSymbol(\"s\"))");
}

}  // namespace
}  // namespace xls::dslx
