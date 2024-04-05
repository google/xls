// Copyright 2020 The XLS Authors
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

#include "xls/passes/stateless_query_engine.h"

#include <string_view>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

class StatelessQueryEngineTest : public IrTestBase {};

TEST_F(StatelessQueryEngineTest, ArbitraryNode) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.Add(fb.Literal(UBits(65, 7)), fb.Literal(Bits::AllOnes(7)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0bXXX_XXXX");
  EXPECT_FALSE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_FALSE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_FALSE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

TEST_F(StatelessQueryEngineTest, Literal) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.Literal(UBits(65, 7));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0b100_0001");
  EXPECT_TRUE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_FALSE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_FALSE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

TEST_F(StatelessQueryEngineTest, ZeroLiteral) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.Literal(UBits(0, 7));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0b000_0000");
  EXPECT_FALSE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_FALSE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

TEST_F(StatelessQueryEngineTest, OneHotLiteral) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.Literal(UBits(16, 7));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0b001_0000");
  EXPECT_TRUE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

TEST_F(StatelessQueryEngineTest, OneHotLsb) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.OneHot(fb.Param("p", p.GetBitsType(13)), LsbOrMsb::kLsb);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0bXX_XXXX_XXXX_XXXX");
  EXPECT_TRUE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

TEST_F(StatelessQueryEngineTest, OneHotMsb) {
  Package p("test_package");
  FunctionBuilder fb("f", &p);
  fb.OneHot(fb.Param("p", p.GetBitsType(13)), LsbOrMsb::kMsb);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  VLOG(3) << f->DumpIr();

  StatelessQueryEngine query_engine;
  EXPECT_EQ(query_engine.ToString(f->return_value()), "0bXX_XXXX_XXXX_XXXX");
  EXPECT_TRUE(query_engine.AtLeastOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.AtMostOneBitTrue(f->return_value()));
  EXPECT_TRUE(query_engine.ExactlyOneBitTrue(f->return_value()));
}

}  // namespace
}  // namespace xls
