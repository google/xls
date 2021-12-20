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

#include "xls/dslx/symbolic_type.h"

#include "gmock/gmock.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;

TEST(SymbolicTypeTest, AddBitsLeaf) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/8, /*value=*/255);
  auto sym =
      SymbolicType(ff.GetBitsOrDie(), ff.GetBitCount().value(), ff.IsSigned());

  EXPECT_TRUE(sym.IsLeaf());
  XLS_EXPECT_OK(sym.ToString());
  EXPECT_EQ(sym.ToString().value(), "255");
  EXPECT_EQ(sym.IsSigned(), ff.IsSigned());
  EXPECT_EQ(sym.GetBitCount(), ff.GetBitCount().value());
  EXPECT_THAT(sym.GetBits(), IsOkAndHolds(ff.GetBitsOrDie()));
}

TEST(SymbolicTypeTest, AddParamLeaf) {
  auto module = std::make_unique<Module>("test");
  Pos fake_pos("<fake>", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  auto* u32 = module->Make<BuiltinTypeAnnotation>(fake_span, BuiltinType::kU32);
  auto* x = module->Make<NameDef>(fake_span, "x", /*definer=*/nullptr);
  Param* param = {module->Make<Param>(x, u32)};
  auto sym = SymbolicType(param, /*bit_count=*/32, /*is_signed=*/false);
  XLS_EXPECT_OK(sym.ToString());
  EXPECT_EQ(sym.ToString().value(), "x");
}

TEST(SymbolicTypeTest, CreateNode) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/255);
  auto zero = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/0);

  auto sym_ff =
      SymbolicType(ff.GetBitsOrDie(), ff.GetBitCount().value(), ff.IsSigned());
  auto sym_zero = SymbolicType(zero.GetBitsOrDie(), zero.GetBitCount().value(),
                               zero.IsSigned());
  auto node =
      SymbolicType(SymbolicType::Nodes{&sym_ff, &sym_zero}, BinopKind::kAdd,
                   sym_ff.GetBitCount(), sym_ff.IsSigned());
  XLS_EXPECT_OK(node.ToString());
  EXPECT_EQ(node.ToString().value(), "(255, 0)+");
}

}  // namespace
}  // namespace xls::dslx
