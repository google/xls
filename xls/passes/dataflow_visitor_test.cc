// Copyright 2022 The XLS Authors
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

#include "xls/passes/dataflow_visitor.h"

#include <stdint.h>

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

class DataFlowVisitorTest : public IrTestBase {};

// Test visitor which tracks the name of the source node of elements of values
// in the graph.
class TestDataFlowVisitor : public DataFlowVisitor<std::string> {
 protected:
  absl::Status DefaultHandler(Node* node) override {
    return SetValue(
        node, LeafTypeTree<std::string>(node->GetType(), node->GetName()));
  }
};

TEST_F(DataFlowVisitorTest, NameTest) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  FunctionBuilder b(TestName(), p.get());

  BValue x = b.Param("x", u32);
  BValue y = b.Param("y", u32);
  BValue z = b.Param("z", p->GetTupleType({u32, u32}));
  BValue xy = b.Tuple({x, y});
  BValue xy_z = b.Tuple({xy, z});
  BValue xy_z_copy = b.Identity(xy_z);
  BValue y_copy = b.Identity(y);
  BValue x_not_y = b.Tuple({x, b.Not(y_copy, SourceInfo(), "not_y")});
  BValue x_from_tuple = b.TupleIndex(x_not_y, 0);
  BValue x_plus_y = b.Add(x, y, SourceInfo(), "x_plus_y");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataFlowVisitor visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(x.node()).ToString(), "x");
  EXPECT_THAT(visitor.GetValue(y.node()).ToString(), "y");
  EXPECT_THAT(visitor.GetValue(z.node()).ToString(), "(z, z)");
  EXPECT_THAT(visitor.GetValue(xy.node()).ToString(), "(x, y)");
  EXPECT_THAT(visitor.GetValue(xy_z.node()).ToString(), "((x, y), (z, z))");
  EXPECT_THAT(visitor.GetValue(xy_z_copy.node()).ToString(),
              "((x, y), (z, z))");
  EXPECT_THAT(visitor.GetValue(y_copy.node()).ToString(), "y");
  EXPECT_THAT(visitor.GetValue(x_not_y.node()).ToString(), "(x, not_y)");
  EXPECT_THAT(visitor.GetValue(x_from_tuple.node()).ToString(), "x");
  EXPECT_THAT(visitor.GetValue(x_plus_y.node()).ToString(), "x_plus_y");
}

}  // namespace
}  // namespace xls
