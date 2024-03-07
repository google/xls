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

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/concrete_type.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_expression.h"

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
  absl::flat_hash_map<std::string, std::unique_ptr<Type>>
      parametric_binding_types;
  parametric_binding_types.emplace("X", BitsType::MakeU32());

  const Span fake_span = Span::Fake();
  auto arg_type = BitsType::MakeU8();
  auto param_type = std::make_unique<BitsType>(
      /*is_signed=*/false,
      TypeDim(std::make_unique<ParametricSymbol>("X", fake_span)));

  DeduceCtx deduce_ctx(
      /*type_info=*/nullptr, /*module=*/nullptr, /*deduce_function=*/nullptr,
      /*typecheck_function=*/nullptr, /*typecheck_module=*/nullptr,
      /*typecheck_invocation=*/nullptr, /*import_data=*/nullptr,
      /*warnings=*/nullptr, /*parent=*/nullptr);
  ParametricBindContext ctx{fake_span, parametric_binding_types,
                            parametric_default_exprs, parametric_env,
                            deduce_ctx};
  XLS_ASSERT_OK(ParametricBindTypeDim(*param_type, param_type->size(),
                                      *arg_type, arg_type->size(), ctx));
  ASSERT_TRUE(parametric_env.contains("X"));
  EXPECT_EQ(parametric_env.at("X"), InterpValue::MakeU32(8));
}

// Tests a simple sample binding that's the same as above, but triggered at the
// type level.
TEST(ParametricBindTest, SampleTypeBind) {
  absl::flat_hash_map<std::string, InterpValue> parametric_env;
  const absl::flat_hash_map<std::string, Expr*> parametric_default_exprs = {
      {"X", nullptr},
  };
  absl::flat_hash_map<std::string, std::unique_ptr<Type>>
      parametric_binding_types;
  parametric_binding_types.emplace("X", BitsType::MakeU32());

  const Span fake_span = Span::Fake();
  auto arg_type = BitsType::MakeU8();
  auto param_type = std::make_unique<BitsType>(
      /*is_signed=*/false,
      TypeDim(std::make_unique<ParametricSymbol>("X", fake_span)));

  DeduceCtx deduce_ctx(
      /*type_info=*/nullptr, /*module=*/nullptr, /*deduce_function=*/nullptr,
      /*typecheck_function=*/nullptr, /*typecheck_module=*/nullptr,
      /*typecheck_invocation=*/nullptr, /*import_data=*/nullptr,
      /*warnings=*/nullptr, /*parent=*/nullptr);
  ParametricBindContext ctx{fake_span, parametric_binding_types,
                            parametric_default_exprs, parametric_env,
                            deduce_ctx};

  XLS_ASSERT_OK(ParametricBind(*param_type, *arg_type, ctx));
  ASSERT_TRUE(parametric_env.contains("X"));
  EXPECT_EQ(parametric_env.at("X"), InterpValue::MakeU32(8));
}

}  // namespace
}  // namespace xls::dslx
