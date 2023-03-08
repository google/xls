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

#include "xls/dslx/type_system/parametric_bind.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

// Tests a simple sample binding like:
//    p<X: u32>(x: uN[X]) -> ()
//    p(u8:42)
// After that invocation X should be 8.
TEST(ParametricBindTest, SampleConcreteDimBind) {
  absl::flat_hash_map<std::string, InterpValue> parametric_env;
  const absl::flat_hash_map<std::string, Expr*> parametric_default_exprs = {
      {"X", nullptr},
  };
  absl::flat_hash_map<std::string, std::unique_ptr<ConcreteType>>
      parametric_binding_types;
  parametric_binding_types.emplace("X", BitsType::MakeU32());

  const Span fake_span = Span::Fake();
  auto arg_type = BitsType::MakeU8();
  auto param_type = std::make_unique<BitsType>(
      /*is_signed=*/false,
      ConcreteTypeDim(std::make_unique<ParametricSymbol>("X", fake_span)));

  XLS_ASSERT_OK(ParametricBindConcreteTypeDim(
      *param_type, param_type->size(), *arg_type, arg_type->size(), fake_span,
      parametric_binding_types, parametric_default_exprs, parametric_env));
  ASSERT_TRUE(parametric_env.contains("X"));
  EXPECT_EQ(parametric_env.at("X"), InterpValue::MakeU32(8));
}

}  // namespace
}  // namespace xls::dslx
