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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

class DataflowVisitorTest : public IrTestBase {};

// Data structure which aggregates the values of a dataflow join operation in a
// easy to ready representation.
struct DataSource {
  absl::btree_set<std::string> sources;
  absl::btree_set<std::string> controls;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DataSource& ds) {
    std::string src_str = absl::StrJoin(ds.sources, " OR ");
    std::string cntl_str = absl::StrJoin(ds.controls, ",");
    absl::Format(
        &sink, "%s",
        cntl_str.empty() ? src_str : absl::StrCat(src_str, " | ", cntl_str));
  }
};

// DefaultHandler for a dataflow visitor which creates a string for each leaf
// element which is the name of the string and the index of the leaf
// element. For example, for node `x` with type `(u32, u32)[3]` an example leaf
// element is: x[1].0.
absl::StatusOr<LeafTypeTree<DataSource>> VisitorDefaultHandler(Node* node) {
  LeafTypeTree<DataSource> result(node->GetType());
  XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachIndex(
      result.AsMutableView(),
      [&](Type* type, DataSource& element, absl::Span<const int64_t> index) {
        std::string s = node->GetName();
        Type* subtype = node->GetType();
        for (int64_t i : index) {
          if (subtype->IsTuple()) {
            subtype = subtype->AsTupleOrDie()->element_type(i);
            absl::StrAppend(&s, ".", i);
          } else {
            subtype = subtype->AsArrayOrDie()->element_type();
            absl::StrAppend(&s, "[", i, "]");
          }
        }
        element.sources = {s};
        return absl::OkStatus();
      }));

  return std::move(result);
}

// Test visitor which tracks the name of the source node of elements of values
// in the graph. The data join operator combines the names with " OR " and the
// control join operator appends the control signal name.
class TestDataflowVisitor : public DataflowVisitor<DataSource> {
 protected:
  absl::Status DefaultHandler(Node* node) final {
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<DataSource> result,
                         VisitorDefaultHandler(node));
    return SetValue(node, std::move(result));
  }

  absl::StatusOr<DataSource> JoinElements(
      Type* element_type, absl::Span<const DataSource* const> data_sources,
      absl::Span<const LeafTypeTreeView<DataSource>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    XLS_RET_CHECK(!data_sources.empty());
    DataSource result = *data_sources.front();
    for (const DataSource* other : data_sources.subspan(1)) {
      result.sources.insert(other->sources.begin(), other->sources.end());
    }
    return std::move(result);
  }
};

// Test visitor which appends the control signal to the leaf element string.
class TestDataflowVisitorWithControl : public TestDataflowVisitor {
 protected:
  absl::StatusOr<DataSource> JoinElements(
      Type* element_type, absl::Span<const DataSource* const> data_sources,
      absl::Span<const LeafTypeTreeView<DataSource>> control_sources,
      Node* node, absl::Span<const int64_t> index) final {
    XLS_RET_CHECK(!data_sources.empty());
    DataSource result = *data_sources.front();
    for (const DataSource* other : data_sources.subspan(1)) {
      result.sources.insert(other->sources.begin(), other->sources.end());
      result.controls.insert(other->controls.begin(), other->controls.end());
    }
    for (const LeafTypeTreeView<DataSource> control_source : control_sources) {
      XLS_RET_CHECK(IsLeafType(control_source.type()));
      const DataSource& control_element = control_source.elements().front();
      result.controls.insert(control_element.sources.begin(),
                             control_element.sources.end());
    }
    return std::move(result);
  }
};

TEST_F(DataflowVisitorTest, Tuples) {
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
  BValue z0 = b.TupleIndex(z, 0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitor visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(x.node()).ToString(), "x");
  EXPECT_THAT(visitor.GetValue(y.node()).ToString(), "y");
  EXPECT_THAT(visitor.GetValue(z.node()).ToString(), "(z.0, z.1)");
  EXPECT_THAT(visitor.GetValue(xy.node()).ToString(), "(x, y)");
  EXPECT_THAT(visitor.GetValue(xy_z.node()).ToString(), "((x, y), (z.0, z.1))");
  EXPECT_THAT(visitor.GetValue(xy_z_copy.node()).ToString(),
              "((x, y), (z.0, z.1))");
  EXPECT_THAT(visitor.GetValue(y_copy.node()).ToString(), "y");
  EXPECT_THAT(visitor.GetValue(x_not_y.node()).ToString(), "(x, not_y)");
  EXPECT_THAT(visitor.GetValue(x_from_tuple.node()).ToString(), "x");
  EXPECT_THAT(visitor.GetValue(x_plus_y.node()).ToString(), "x_plus_y");
  EXPECT_THAT(visitor.GetValue(z0.node()).ToString(), "z.0");
}

TEST_F(DataflowVisitorTest, Arrays) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  FunctionBuilder b(TestName(), p.get());

  BValue i = b.Param("i", u32);
  BValue j = b.Param("j", u32);
  BValue x = b.Param("x", p->GetArrayType(1, p->GetTupleType({u32, u32})));
  BValue y = b.Param("y", p->GetArrayType(3, u32));
  BValue z = b.Param("z", p->GetArrayType(2, p->GetArrayType(3, u32)));

  BValue one = b.Literal(UBits(1, 32), SourceInfo(), "one");
  BValue y_1 = b.ArrayIndex(y, {one});
  BValue y_i = b.ArrayIndex(y, {i});
  BValue z_1_i = b.ArrayIndex(z, {one, i});
  BValue y_updated_1 = b.ArrayUpdate(y, i, {one});
  BValue y_updated_i = b.ArrayUpdate(y, i, {j});
  BValue z_updated_i_1 = b.ArrayUpdate(z, i, {i, one});
  BValue z_updated_1_i = b.ArrayUpdate(z, i, {one, i});
  BValue yy = b.ArrayConcat({y, y});
  BValue ij = b.Array({i, j}, u32);
  BValue y_slice = b.ArraySlice(y, i, 2);
  BValue y_slice_past_the_end = b.ArraySlice(y, i, 5);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitor visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(x.node()).ToString(), "[(x[0].0, x[0].1)]");
  EXPECT_THAT(visitor.GetValue(y.node()).ToString(), "[y[0], y[1], y[2]]");
  EXPECT_THAT(visitor.GetValue(z.node()).ToString(),
              "[[z[0][0], z[0][1], z[0][2]], [z[1][0], z[1][1], z[1][2]]]");

  EXPECT_THAT(visitor.GetValue(y_1.node()).ToString(), "y[1]");
  EXPECT_THAT(visitor.GetValue(y_i.node()).ToString(), "y[0] OR y[1] OR y[2]");
  EXPECT_THAT(visitor.GetValue(z_1_i.node()).ToString(),
              "z[1][0] OR z[1][1] OR z[1][2]");
  EXPECT_THAT(visitor.GetValue(y_updated_1.node()).ToString(),
              "[y[0], i, y[2]]");
  EXPECT_THAT(visitor.GetValue(y_updated_i.node()).ToString(),
              "[i OR y[0], i OR y[1], i OR y[2]]");
  EXPECT_THAT(visitor.GetValue(z_updated_i_1.node()).ToString(),
              "[[z[0][0], i OR z[0][1], z[0][2]], "
              "[z[1][0], i OR z[1][1], z[1][2]]]");
  EXPECT_THAT(visitor.GetValue(z_updated_1_i.node()).ToString(),
              "[[z[0][0], z[0][1], z[0][2]], "
              "[i OR z[1][0], i OR z[1][1], i OR z[1][2]]]");

  EXPECT_THAT(visitor.GetValue(yy.node()).ToString(),
              "[y[0], y[1], y[2], y[0], y[1], y[2]]");
  EXPECT_THAT(visitor.GetValue(ij.node()).ToString(), "[i, j]");

  EXPECT_THAT(visitor.GetValue(y_slice.node()).ToString(),
              "[y[0] OR y[1] OR y[2], y[1] OR y[2]]");
  EXPECT_THAT(visitor.GetValue(y_slice_past_the_end.node()).ToString(),
              "[y[0] OR y[1] OR y[2], y[1] OR y[2], y[2], y[2], y[2]]");
}

TEST_F(DataflowVisitorTest, MultiDimensionalArrays) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u32_3 = p->GetArrayType(3, u32);
  Type* u32_3_2 = p->GetArrayType(2, u32_3);
  FunctionBuilder b(TestName(), p.get());

  BValue i = b.Param("i", u32);
  BValue j = b.Param("j", u32);
  BValue x = b.Param("x", u32_3_2);

  BValue x_empty = b.ArrayIndex(x, {});
  BValue x_i = b.ArrayIndex(x, {i});
  BValue x_i_then_j = b.ArrayIndex(x_i, {j});
  BValue x_i_j = b.ArrayIndex(x, {i, j});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitorWithControl visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(x_empty.node()).ToString(),
              "[[x[0][0], x[0][1], x[0][2]], "
              "[x[1][0], x[1][1], x[1][2]]]");
  EXPECT_THAT(visitor.GetValue(x_i_j.node()).ToString(),
              "x[0][0] OR x[0][1] OR x[0][2] OR "
              "x[1][0] OR x[1][1] OR x[1][2] | i,j");
  EXPECT_THAT(visitor.GetValue(x_i_then_j.node()).ToString(),
              "x[0][0] OR x[0][1] OR x[0][2] OR "
              "x[1][0] OR x[1][1] OR x[1][2] | i,j");
}

TEST_F(DataflowVisitorTest, Selects) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  FunctionBuilder b(TestName(), p.get());

  BValue pred = b.Param("pred", p->GetBitsType(1));
  BValue x = b.Param("x", u32);
  BValue y = b.Param("y", u32);
  BValue xx = b.Tuple({x, x});
  BValue xy = b.Tuple({x, y});
  BValue sel_xx_xy = b.Select(pred, {xx, xy});
  BValue sel_x_y = b.Select(pred, {x, y});
  BValue sel_x_x = b.Select(pred, {x, x});
  BValue sel_x_x_y = b.Select(x, {x, x}, /*default_value=*/y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitor visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(sel_x_x.node()).ToString(), "x");
  EXPECT_THAT(visitor.GetValue(sel_x_y.node()).ToString(), "x OR y");
  EXPECT_THAT(visitor.GetValue(sel_xx_xy.node()).ToString(), "(x, x OR y)");
  EXPECT_THAT(visitor.GetValue(sel_x_x_y.node()).ToString(), "x OR y");
}

TEST_F(DataflowVisitorTest, SelectsWithControl) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  FunctionBuilder b(TestName(), p.get());

  BValue pred1 = b.Param("pred1", p->GetBitsType(1));
  BValue pred2 = b.Param("pred2", p->GetBitsType(1));
  BValue x = b.Param("x", u32);
  BValue y = b.Param("y", u32);
  BValue sel_x_y_pred1 = b.Select(pred1, {x, y});
  BValue sel_x_y_pred2 = b.Select(pred2, {x, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitorWithControl visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(sel_x_y_pred1.node()).ToString(),
              "x OR y | pred1");
  EXPECT_THAT(visitor.GetValue(sel_x_y_pred2.node()).ToString(),
              "x OR y | pred2");
}

TEST_F(DataflowVisitorTest, OneHotSelects) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  FunctionBuilder b(TestName(), p.get());

  BValue selector = b.Param("selector", p->GetBitsType(2));
  BValue x = b.Param("x", u32);
  BValue y = b.Param("y", u32);
  BValue z = b.Param("z", u32);
  BValue ohs = b.OneHotSelect(selector, {x, y}, SourceInfo(), "ohs");
  BValue one_hot = b.OneHot(selector, LsbOrMsb::kLsb, SourceInfo(), "one_hot");
  BValue onehot_ohs =
      b.OneHotSelect(one_hot, {x, y, z}, SourceInfo(), "onehot_ohs");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());

  TestDataflowVisitor visitor;
  XLS_ASSERT_OK(f->Accept(&visitor));

  EXPECT_THAT(visitor.GetValue(ohs.node()).ToString(), "ohs");
  EXPECT_THAT(visitor.GetValue(onehot_ohs.node()).ToString(), "x OR y OR z");
}

}  // namespace
}  // namespace xls
