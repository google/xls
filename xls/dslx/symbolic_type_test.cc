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

TEST(SymbolicTypeTest, AddBitsLeaf) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/8, /*value=*/25);
  auto sym = SymbolicType::MakeLiteral(
      ConcreteInfo{ff.IsSigned(), ff.GetBitCount().value(),
                   ff.GetBitsOrDie().ToInt64().value()});
  EXPECT_TRUE(sym.IsLeaf());
  XLS_EXPECT_OK(sym.ToString());
  EXPECT_EQ(sym.ToString().value(), "25");
  EXPECT_EQ(sym.IsSigned(), ff.IsSigned());
  EXPECT_EQ(sym.bit_count(), ff.GetBitCount().value());
}

TEST(SymbolicTypeTest, AddParamLeaf) {
  auto module = std::make_unique<Module>("test");
  Pos fake_pos("<fake>", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  auto sym = SymbolicType::MakeParam(ConcreteInfo{/*is_signed=*/false,
                                                  /*bit_count=*/32,
                                                  /*bit_value=*/0, "x"});
  XLS_EXPECT_OK(sym.ToString());
  EXPECT_EQ(sym.ToString().value(), "x");
}

TEST(SymbolicTypeTest, CreateNode) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/25);
  auto zero = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/0);

  auto sym_ff = SymbolicType::MakeLiteral(
      ConcreteInfo{ff.IsSigned(), ff.GetBitCount().value(),
                   ff.GetBitsOrDie().ToInt64().value()});
  auto sym_zero = SymbolicType::MakeLiteral(
      ConcreteInfo{zero.IsSigned(), zero.GetBitCount().value(),
                   zero.GetBitsOrDie().ToInt64().value()});
  auto node = SymbolicType::MakeBinary(
      SymbolicType::Nodes{BinopKind::kAdd, &sym_ff, &sym_zero},
      ConcreteInfo{sym_ff.IsSigned(), sym_ff.bit_count()});
  XLS_EXPECT_OK(node.ToString());
  EXPECT_EQ(node.ToString().value(), "(25, 0)+");
}

}  // namespace
}  // namespace xls::dslx
