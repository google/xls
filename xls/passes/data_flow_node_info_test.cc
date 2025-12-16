// Copyright 2025 The XLS Authors
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

#include "xls/passes/data_flow_node_info.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/passes/node_source_analysis.h"
#include "xls/passes/partial_info_query_engine.h"

namespace xls {

typedef absl::flat_hash_set<NodeSource> NodeSourceSet;

template <typename Sink>
void AbslStringify(Sink& sink, const NodeSourceSet& my_object) {
  absl::Format(&sink, "{%s}", absl::StrJoin(my_object, ", "));
}

template <typename Sink>
void AbslStringify(Sink& sink, const NodeSourceSet* my_object) {
  AbslStringify(sink, *my_object);
}

namespace {

class TestParamCountInfo
    : public DataFlowLazyNodeInfo<TestParamCountInfo, int64_t> {
 public:
  TestParamCountInfo()
      : DataFlowLazyNodeInfo<TestParamCountInfo, int64_t>(
            /*default_to_leaf=*/false) {}

  int64_t ComputeInfoForLeafNode(Node* node) const override final {
    return node->Is<xls::Param>() ? 1 : 0;
  }
  int64_t ComputeInfoForBitsLiteral(
      const xls::Bits& literal) const override final {
    return 0;
  }
  xls::LeafTypeTree<int64_t> ComputeInfoTreeForLeafNode(
      xls::Node* node) const override final {
    LOG(FATAL)
        << "ComputeInfoTreeForLeafNode should be unused for TestNodeSourceInfo";
    return xls::LeafTypeTree<int64_t>();
  }
  int64_t MergeInfos(
      const absl::Span<const int64_t>& infos) const override final {
    int64_t result = 0;
    for (int64_t info : infos) {
      result += info;
    }
    return result;
  }
};

#if 0
std::string NodeSourceSetToString(const NodeSourceSet& node_set) {
  return absl::StrFormat(
      "{%s}", absl::StrJoin(node_set, ", ",
                            [](std::string* out, const xls::Node* node) {
                              absl::StrAppend(out, node->GetName());
                            }));
}
#endif

class TestNodeSourceInfo
    : public DataFlowLazyNodeInfo<TestNodeSourceInfo, NodeSourceSet> {
 public:
  TestNodeSourceInfo()
      : DataFlowLazyNodeInfo<TestNodeSourceInfo, NodeSourceSet>(
            /*default_to_leaf=*/true) {}

  NodeSourceSet ComputeInfoForLeafNode(Node* node) const override final {
    LOG(FATAL)
        << "ComputeInfoForLeafNode should be unused for TestNodeSourceInfo";
    return NodeSourceSet{};
  }
  NodeSourceSet ComputeInfoForBitsLiteral(
      const xls::Bits& literal) const override final {
    LOG(FATAL)
        << "ComputeInfoForBitsLiteral should be unused for TestNodeSourceInfo";
    return NodeSourceSet{};
  }
  xls::LeafTypeTree<NodeSourceSet> ComputeInfoTreeForLeafNode(
      xls::Node* node) const override final {
    xls::LeafTypeTree<NodeSourceSet> result(node->GetType());
    CHECK_OK(xls::leaf_type_tree::ForEachIndex(
        result.AsMutableView(),
        [&](xls::Type* element_type, NodeSourceSet& element,
            absl::Span<const int64_t> index) {
          element = {
              xls::NodeSource(node, std::vector(index.begin(), index.end()))};
          return absl::OkStatus();
        }));
    return result;
  }

  NodeSourceSet MergeInfos(
      const absl::Span<const NodeSourceSet>& infos) const override final {
    NodeSourceSet ret;
    for (const NodeSourceSet& info : infos) {
      ret.insert(info.begin(), info.end());
    }
    return ret;
  }
};

class DataFlowNodeInfoTest : public IrTestBase {
 public:
  DataFlowNodeInfoTest()
      : query_engine_(std::make_unique<PartialInfoQueryEngine>()) {}

  QueryEngine* query_engine() const { return query_engine_.get(); }

 private:
  std::unique_ptr<PartialInfoQueryEngine> query_engine_;
};

TEST_F(DataFlowNodeInfoTest, Identity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue id = fb.Identity(x, SourceInfo(), "id");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(id));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());

  EXPECT_EQ(x_count, 1);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                        p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, Add) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(add));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  int64_t add_count = node_info.GetSingleInfoForNode(add.node());
  SharedLeafTypeTree<int64_t> add_tree = node_info.GetInfo(add.node());

  EXPECT_EQ(add_count, 2);
  EXPECT_EQ(add_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                          p->GetBitsType(32), 2));
}

TEST_F(DataFlowNodeInfoTest, ModifyNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(32, 32)));
  BValue add = fb.Add(x, y, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(add));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    int64_t add_count = node_info.GetSingleInfoForNode(add.node());
    SharedLeafTypeTree<int64_t> add_tree = node_info.GetInfo(add.node());
    EXPECT_EQ(add_count, 2);
    EXPECT_EQ(add_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                            p->GetBitsType(32), 2));
  }

  EXPECT_TRUE(add.node()->ReplaceOperand(x.node(), l.node()));

  {
    int64_t add_count = node_info.GetSingleInfoForNode(add.node());
    SharedLeafTypeTree<int64_t> add_tree = node_info.GetInfo(add.node());
    EXPECT_EQ(add_count, 1);
    EXPECT_EQ(add_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                            p->GetBitsType(32), 1));
  }
}

TEST_F(DataFlowNodeInfoTest, AddLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(32, 32)));
  BValue add = fb.Add(x, l, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(add));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  int64_t add_count = node_info.GetSingleInfoForNode(add.node());
  SharedLeafTypeTree<int64_t> add_tree = node_info.GetInfo(add.node());

  EXPECT_EQ(add_count, 1);
  EXPECT_EQ(add_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                          p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, Select) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue c = fb.Param("c", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue eq = fb.Eq(c, l, SourceInfo(), "eq");

  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel = fb.Select(eq, x, y, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  int64_t sel_count = node_info.GetSingleInfoForNode(sel.node());
  SharedLeafTypeTree<int64_t> sel_tree = node_info.GetInfo(sel.node());

  // Should not include the selector
  EXPECT_EQ(sel_count, 2);
  EXPECT_EQ(sel_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                          p->GetBitsType(32), 2));

  int64_t eq_count = node_info.GetSingleInfoForNode(eq.node());
  SharedLeafTypeTree<int64_t> eq_tree = node_info.GetInfo(eq.node());

  EXPECT_EQ(eq_count, 1);
  EXPECT_EQ(eq_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                         p->GetBitsType(1), 1));
}

TEST_F(DataFlowNodeInfoTest, Invoke) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb(absl::StrCat(TestName(), "_subroutine"), p.get());
  BValue x = sub_fb.Param("x", p->GetBitsType(32));
  BValue y = sub_fb.Param("y", p->GetBitsType(32));
  BValue add = sub_fb.Add(x, y, SourceInfo(), "add");
  BValue sub_ret = sub_fb.Tuple({x, y, add}, SourceInfo(), "sub_ret");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * sub_fn,
                           sub_fb.BuildWithReturnValue(sub_ret));

  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue invoke_literal =
      fb.Invoke({a, l}, sub_fn, SourceInfo(), "invoke_literal");
  BValue y_returned_literal =
      fb.TupleIndex(invoke_literal, 1, SourceInfo(), "y_returned_literal");
  BValue sum_returned_literal =
      fb.TupleIndex(invoke_literal, 2, SourceInfo(), "sum_returned_literal");

  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue invoke = fb.Invoke({a, b}, sub_fn, SourceInfo(), "invoke");
  BValue y_returned = fb.TupleIndex(invoke, 1, SourceInfo(), "y_returned");
  BValue sum_returned = fb.TupleIndex(invoke, 2, SourceInfo(), "sum_returned");

  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(sum_returned));

  TestParamCountInfo sub_node_info;
  sub_node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(sub_node_info.Attach(sub_fn));
  XLS_ASSERT_OK(query_engine()->Populate(sub_fn).status());

  int64_t sub_ret_count = sub_node_info.GetSingleInfoForNode(sub_ret.node());
  SharedLeafTypeTree<int64_t> sub_ret_tree =
      sub_node_info.GetInfo(sub_ret.node());
  EXPECT_EQ(sub_ret_count, 4);
  EXPECT_EQ(sub_ret_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32),
                                 p->GetBitsType(32)}),
                {1, 1, 2}));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  int64_t invoke_count_literal =
      node_info.GetSingleInfoForNode(invoke_literal.node());
  SharedLeafTypeTree<int64_t> invoke_literal_tree =
      node_info.GetInfo(invoke_literal.node());
  EXPECT_EQ(invoke_count_literal, 2);
  EXPECT_EQ(invoke_literal_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32),
                                 p->GetBitsType(32)}),
                {1, 0, 1}));

  int64_t sum_returned_count_literal =
      node_info.GetSingleInfoForNode(sum_returned_literal.node());
  SharedLeafTypeTree<int64_t> sum_returned_literal_tree =
      node_info.GetInfo(sum_returned_literal.node());
  EXPECT_EQ(sum_returned_count_literal, 1);
  EXPECT_EQ(
      sum_returned_literal_tree,
      LeafTypeTree<int64_t>::CreateSingleElementTree(p->GetBitsType(32), 1));

  int64_t y_returned_count_literal =
      node_info.GetSingleInfoForNode(y_returned_literal.node());
  SharedLeafTypeTree<int64_t> y_returned_literal_tree =
      node_info.GetInfo(y_returned_literal.node());

  EXPECT_EQ(y_returned_count_literal, 0);
  EXPECT_EQ(
      y_returned_literal_tree,
      LeafTypeTree<int64_t>::CreateSingleElementTree(p->GetBitsType(32), 0));

  int64_t invoke_count = node_info.GetSingleInfoForNode(invoke.node());
  SharedLeafTypeTree<int64_t> invoke_tree = node_info.GetInfo(invoke.node());
  EXPECT_EQ(invoke_count, 4);
  EXPECT_EQ(invoke_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32),
                                 p->GetBitsType(32)}),
                {1, 1, 2}));

  int64_t sum_returned_count =
      node_info.GetSingleInfoForNode(sum_returned.node());
  SharedLeafTypeTree<int64_t> sum_returned_tree =
      node_info.GetInfo(sum_returned.node());
  EXPECT_EQ(sum_returned_count, 2);
  EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 2));

  int64_t y_returned_count = node_info.GetSingleInfoForNode(y_returned.node());
  SharedLeafTypeTree<int64_t> y_returned_tree =
      node_info.GetInfo(y_returned.node());
  EXPECT_EQ(y_returned_count, 1);
  EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                 p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, ModifyInvoke) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb(absl::StrCat(TestName(), "_subroutine"), p.get());
  BValue x = sub_fb.Param("x", p->GetBitsType(32));
  BValue y = sub_fb.Param("y", p->GetBitsType(32));
  BValue add = sub_fb.Add(x, y, SourceInfo(), "add");
  BValue sub_ret = sub_fb.Tuple({x, y, add}, SourceInfo(), "sub_ret");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * sub_fn,
                           sub_fb.BuildWithReturnValue(sub_ret));

  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue invoke = fb.Invoke({a, b}, sub_fn, SourceInfo(), "invoke");
  BValue y_returned = fb.TupleIndex(invoke, 1, SourceInfo(), "y_returned");
  BValue sum_returned = fb.TupleIndex(invoke, 2, SourceInfo(), "sum_returned");

  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(sum_returned));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 2);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 2));

    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    EXPECT_EQ(y_returned_count, 1);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
  }

  EXPECT_TRUE(invoke.node()->ReplaceOperand(b.node(), l.node()));

  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 1);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 1));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 0);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 0));
  }
}

TEST_F(DataFlowNodeInfoTest, ModifyInvokeCallee) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb(absl::StrCat(TestName(), "_subroutine"), p.get());
  BValue x = sub_fb.Param("x", p->GetBitsType(32));
  BValue y = sub_fb.Param("y", p->GetBitsType(32));
  BValue l = sub_fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue add = sub_fb.Add(x, y, SourceInfo(), "add");
  BValue sub_ret = sub_fb.Tuple({x, y, add}, SourceInfo(), "sub_ret");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * sub_fn,
                           sub_fb.BuildWithReturnValue(sub_ret));

  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue invoke = fb.Invoke({a, b}, sub_fn, SourceInfo(), "invoke");
  BValue y_returned = fb.TupleIndex(invoke, 1, SourceInfo(), "y_returned");
  BValue sum_returned = fb.TupleIndex(invoke, 2, SourceInfo(), "sum_returned");

  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(sum_returned));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 2);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 2));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 1);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
  }

  EXPECT_TRUE(add.node()->ReplaceOperand(y.node(), l.node()));
  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 1);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 1));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 1);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
  }
}

TEST_F(DataFlowNodeInfoTest, ModifyInvokeParam) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb(absl::StrCat(TestName(), "_subroutine"), p.get());
  BValue x = sub_fb.Param("x", p->GetBitsType(32));
  BValue y = sub_fb.Param("y", p->GetBitsType(32));
  BValue add = sub_fb.Add(x, y, SourceInfo(), "add");
  BValue sub_ret = sub_fb.Tuple({x, y, add}, SourceInfo(), "sub_ret");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * sub_fn,
                           sub_fb.BuildWithReturnValue(sub_ret));

  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue invoke = fb.Invoke({a, b}, sub_fn, SourceInfo(), "invoke");
  BValue y_returned = fb.TupleIndex(invoke, 1, SourceInfo(), "y_returned");
  BValue sum_returned = fb.TupleIndex(invoke, 2, SourceInfo(), "sum_returned");

  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(sum_returned));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 2);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 2));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 1);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
  }

  EXPECT_TRUE(invoke.node()->ReplaceOperand(b.node(), l.node()));
  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 1);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 1));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 0);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 0));
  }
}

TEST_F(DataFlowNodeInfoTest, DeleteInvoke) {
  auto p = CreatePackage();
  FunctionBuilder sub_fb(absl::StrCat(TestName(), "_subroutine"), p.get());
  BValue x = sub_fb.Param("x", p->GetBitsType(32));
  BValue y = sub_fb.Param("y", p->GetBitsType(32));
  BValue add = sub_fb.Add(x, y, SourceInfo(), "add");
  BValue sub_ret = sub_fb.Tuple({x, y, add}, SourceInfo(), "sub_ret");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * sub_fn,
                           sub_fb.BuildWithReturnValue(sub_ret));

  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  auto lx = xls::Value(xls::UBits(50, 32));
  auto ly = xls::Value(xls::UBits(100, 32));
  auto lsum = xls::Value(xls::UBits(200, 32));
  BValue l = fb.Literal(xls::Value(xls::Value::Tuple({lx, ly, lsum})));
  BValue invoke = fb.Invoke({a, b}, sub_fn, SourceInfo(), "invoke");
  BValue y_returned = fb.TupleIndex(invoke, 1, SourceInfo(), "y_returned");
  BValue sum_returned = fb.TupleIndex(invoke, 2, SourceInfo(), "sum_returned");

  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(sum_returned));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 2);

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 1);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
  }

  XLS_EXPECT_OK(invoke.node()->ReplaceUsesWith(l.node()));
  XLS_EXPECT_OK(f->RemoveNode(invoke.node()));
  XLS_EXPECT_OK(p->RemoveFunction(sub_fn));

  {
    SharedLeafTypeTree<int64_t> sum_returned_tree =
        node_info.GetInfo(sum_returned.node());
    int64_t sum_returned_count =
        node_info.GetSingleInfoForNode(sum_returned.node());
    EXPECT_EQ(sum_returned_count, 0);
    EXPECT_EQ(sum_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                     p->GetBitsType(32), 0));

    SharedLeafTypeTree<int64_t> y_returned_tree =
        node_info.GetInfo(y_returned.node());
    int64_t y_returned_count =
        node_info.GetSingleInfoForNode(y_returned.node());
    EXPECT_EQ(y_returned_count, 0);
    EXPECT_EQ(y_returned_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 0));
  }
}

TEST_F(DataFlowNodeInfoTest, Tuple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue tuple = fb.Tuple({x, y}, SourceInfo(), "tuple");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(tuple));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  SharedLeafTypeTree<int64_t> tuple_tree = node_info.GetInfo(tuple.node());
  int64_t tuple_node_count = node_info.GetSingleInfoForNode(tuple.node());

  EXPECT_EQ(tuple_node_count, 2);
  EXPECT_EQ(
      tuple_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}), {1, 1}));
}

TEST_F(DataFlowNodeInfoTest, TupleOfTuples) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue tuple_inner = fb.Tuple({x, y}, SourceInfo(), "tuple");
  BValue tuple_outer = fb.Tuple({x, tuple_inner, y}, SourceInfo(), "tuple");
  BValue tuple_index =
      fb.TupleIndex(tuple_outer, 1, SourceInfo(), "tuple_index");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(tuple_outer));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  int64_t tuple_node_count = node_info.GetSingleInfoForNode(tuple_outer.node());
  SharedLeafTypeTree<int64_t> tuple_node_tree =
      node_info.GetInfo(tuple_outer.node());
  EXPECT_EQ(tuple_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType(
                    {p->GetBitsType(32),
                     p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}),
                     p->GetBitsType(64)}),
                {1, 1, 1, 1}));
  EXPECT_EQ(tuple_node_count, 4);

  SharedLeafTypeTree<int64_t> tuple_index_tree =
      node_info.GetInfo(tuple_index.node());
  int64_t tuple_index_count =
      node_info.GetSingleInfoForNode(tuple_index.node());

  EXPECT_EQ(tuple_index_count, 2);
  EXPECT_EQ(
      tuple_index_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}), {1, 1}));
}

TEST_F(DataFlowNodeInfoTest, TupleParam) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x =
      fb.Param("x", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}));
  BValue tuple_index0 = fb.TupleIndex(x, 0, SourceInfo(), "tuple_index0");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(tuple_index0));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  EXPECT_EQ(
      x_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}), {1, 1}));

  EXPECT_EQ(x_count, 2);

  SharedLeafTypeTree<int64_t> tuple_index0_tree =
      node_info.GetInfo(tuple_index0.node());
  int64_t tuple_index0_count =
      node_info.GetSingleInfoForNode(tuple_index0.node());

  EXPECT_EQ(tuple_index0_count, 1);
  EXPECT_EQ(tuple_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, TupleWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue tuple = fb.Tuple({x, l, y}, SourceInfo(), "tuple");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(tuple));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> tuple_node_tree = node_info.GetInfo(tuple.node());
  int64_t tuple_node_count = node_info.GetSingleInfoForNode(tuple.node());

  EXPECT_EQ(tuple_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32),
                                 p->GetBitsType(64)}),
                {1, 0, 1}));
  EXPECT_EQ(tuple_node_count, 2);
}

TEST_F(DataFlowNodeInfoTest, TupleIdentity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x =
      fb.Param("x", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}));
  BValue id = fb.Identity(x, SourceInfo(), "id");
  BValue tuple_index0 = fb.TupleIndex(id, 0, SourceInfo(), "tuple_index0");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(id));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_count_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 2);
  EXPECT_EQ(
      x_count_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}), {1, 1}));

  SharedLeafTypeTree<int64_t> id_count_tree = node_info.GetInfo(id.node());
  int64_t id_count = node_info.GetSingleInfoForNode(id.node());
  EXPECT_EQ(id_count, 2);
  EXPECT_EQ(
      x_count_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}), {1, 1}));

  SharedLeafTypeTree<int64_t> tuple_index0_count_tree =
      node_info.GetInfo(tuple_index0.node());
  int64_t tuple_index0_count =
      node_info.GetSingleInfoForNode(tuple_index0.node());
  EXPECT_EQ(tuple_index0_count, 1);
  EXPECT_EQ(
      tuple_index0_count_tree,
      LeafTypeTree<int64_t>::CreateSingleElementTree(p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, ArrayParam) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue l = fb.Literal(xls::Value(xls::UBits(0, 32)));
  BValue array_index0 = fb.ArrayIndex(x, {l}, /*assumed_in_bounds=*/false,
                                      SourceInfo(), "array_index0");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_index0));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_index0_tree =
      node_info.GetInfo(array_index0.node());
  int64_t array_index0_count =
      node_info.GetSingleInfoForNode(array_index0.node());

  EXPECT_EQ(array_index0_count, 1);
  EXPECT_EQ(array_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateDynamic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue i = fb.Param("i", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(0, 32)));
  BValue array_update = fb.ArrayUpdate(x, /*update_value=*/l, /*indices=*/{i},
                                       SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());

  // Each element in the array ends up with a count of 10
  EXPECT_EQ(array_update_count, 10 * 10);

  EXPECT_EQ(array_update_tree, LeafTypeTree<int64_t>::CreateFromVector(
                                   p->GetArrayType(10, p->GetBitsType(32)),
                                   {10, 10, 10, 10, 10, 10, 10, 10, 10, 10}));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateDynamic2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue i = fb.Param("i", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue array_update = fb.ArrayUpdate(x, /*update_value=*/y, /*indices=*/{i},
                                       SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());

  // Each element in the array ends up with a count of 11
  EXPECT_EQ(array_update_count, 11 * 10);

  EXPECT_EQ(array_update_tree, LeafTypeTree<int64_t>::CreateFromVector(
                                   p->GetArrayType(10, p->GetBitsType(32)),
                                   {11, 11, 11, 11, 11, 11, 11, 11, 11, 11}));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue l = fb.Literal(xls::Value(xls::UBits(0, 32)));
  BValue array_update = fb.ArrayUpdate(x, /*update_value=*/l, /*indices=*/{l},
                                       SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());
  EXPECT_EQ(array_update_count, 9);
  EXPECT_EQ(array_update_tree, LeafTypeTree<int64_t>::CreateFromVector(
                                   p->GetArrayType(10, p->GetBitsType(32)),
                                   {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateLiteral2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(0, 32)));
  BValue array_update = fb.ArrayUpdate(x, /*update_value=*/y, /*indices=*/{l},
                                       SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());
  EXPECT_EQ(array_update_count, 10);
  EXPECT_EQ(array_update_tree, LeafTypeTree<int64_t>::CreateFromVector(
                                   p->GetArrayType(10, p->GetBitsType(32)),
                                   {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateLiteralOutOfBounds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue l = fb.Literal(xls::Value(xls::UBits(100, 32)));
  BValue array_update = fb.ArrayUpdate(x, /*update_value=*/l, /*indices=*/{l},
                                       SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_tree, LeafTypeTree<int64_t>::CreateFromVector(
                        p->GetArrayType(10, p->GetBitsType(32)),
                        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());

  // Index should get clipped
  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  EXPECT_EQ(array_update_count, 9);
  EXPECT_EQ(array_update_tree, LeafTypeTree<int64_t>::CreateFromVector(
                                   p->GetArrayType(10, p->GetBitsType(32)),
                                   {1, 1, 1, 1, 1, 1, 1, 1, 1, 0}));
}

TEST_F(DataFlowNodeInfoTest, ArrayUpdateLiteralOutOfBounds2) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(4, p->GetBitsType(32)));
  auto vi = xls::Value(xls::UBits(5, 32));
  BValue la =
      fb.Literal(xls::Value::ArrayOrDie({vi, vi, vi, vi}), SourceInfo(), "la");
  BValue arr0 = fb.Array({x, x, x}, x.GetType(), SourceInfo(), "arr0");
  BValue arr1 =
      fb.Array({arr0, arr0, arr0}, arr0.GetType(), SourceInfo(), "arr0");
  BValue i0 = fb.Literal(xls::Value(xls::UBits(100, 32)));
  BValue i1 = fb.Literal(xls::Value(xls::UBits(100, 32)));
  BValue array_update =
      fb.ArrayUpdate(arr1, /*update_value=*/la, /*indices=*/{i0, i1},
                     SourceInfo(), /*name=*/"array_update");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_update));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> arr1_tree = node_info.GetInfo(arr1.node());
  int64_t arr1_count = node_info.GetSingleInfoForNode(arr1.node());
  EXPECT_EQ(arr1_count, 4 * 3 * 3);
  EXPECT_EQ(
      arr1_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetArrayType(
              3, p->GetArrayType(3, p->GetArrayType(4, p->GetBitsType(32)))),
          {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_update_tree =
      node_info.GetInfo(array_update.node());
  int64_t array_update_count =
      node_info.GetSingleInfoForNode(array_update.node());

  // Index should get clipped
  EXPECT_EQ(array_update_count, 4 * 3 * 3 - 4);
  EXPECT_EQ(
      array_update_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetArrayType(
              3, p->GetArrayType(3, p->GetArrayType(4, p->GetBitsType(32)))),
          {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}));
}

TEST_F(DataFlowNodeInfoTest, ArrayWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue array = fb.Array({x, l, y}, x.GetType(), SourceInfo(), "array");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(array));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_node_tree = node_info.GetInfo(array.node());
  int64_t array_node_count = node_info.GetSingleInfoForNode(array.node());
  EXPECT_EQ(array_node_count, 2);
  EXPECT_EQ(array_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(3, p->GetBitsType(32)), {1, 0, 1}));
}

TEST_F(DataFlowNodeInfoTest, ArrayIndexDynamic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue i = fb.Param("i", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue array = fb.Array({x, l, y}, x.GetType(), SourceInfo(), "array");
  BValue array_index0 = fb.ArrayIndex(array, {i}, /*assumed_in_bounds=*/false,
                                      SourceInfo(), "array_index0");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_index0));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_node_tree = node_info.GetInfo(array.node());
  int64_t array_node_count = node_info.GetSingleInfoForNode(array.node());
  EXPECT_EQ(array_node_count, 2);
  EXPECT_EQ(array_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(3, p->GetBitsType(32)), {1, 0, 1}));

  SharedLeafTypeTree<int64_t> array_index0_tree =
      node_info.GetInfo(array_index0.node());
  int64_t array_index0_count =
      node_info.GetSingleInfoForNode(array_index0.node());
  EXPECT_EQ(array_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 2));

  // The index variable should not be included
  EXPECT_EQ(array_index0_count, 2);
}

TEST_F(DataFlowNodeInfoTest, ArrayIndexWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue l0 = fb.Literal(xls::Value(xls::UBits(0, 32)));
  BValue l1 = fb.Literal(xls::Value(xls::UBits(1, 32)));
  BValue array = fb.Array({x, l, y}, x.GetType(), SourceInfo(), "array");
  BValue array_index0 = fb.ArrayIndex(array, {l0}, /*assumed_in_bounds=*/false,
                                      SourceInfo(), "array_index0");
  BValue array_index1 = fb.ArrayIndex(array, {l1}, /*assumed_in_bounds=*/false,
                                      SourceInfo(), "array_index1");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(array));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_node_tree = node_info.GetInfo(array.node());
  int64_t array_node_count = node_info.GetSingleInfoForNode(array.node());
  EXPECT_EQ(array_node_count, 2);
  EXPECT_EQ(array_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(3, p->GetBitsType(32)), {1, 0, 1}));

  SharedLeafTypeTree<int64_t> array_index0_tree =
      node_info.GetInfo(array_index0.node());
  int64_t array_index0_count =
      node_info.GetSingleInfoForNode(array_index0.node());
  EXPECT_EQ(array_index0_count, 1);
  EXPECT_EQ(array_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));

  SharedLeafTypeTree<int64_t> array_index1_tree =
      node_info.GetInfo(array_index1.node());
  int64_t array_index1_count =
      node_info.GetSingleInfoForNode(array_index1.node());
  EXPECT_EQ(array_index1_count, 0);
  EXPECT_EQ(array_index1_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 0));
}

TEST_F(DataFlowNodeInfoTest, ArrayIndexWithLiteralOutOfBounds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue l0 = fb.Literal(xls::Value(xls::UBits(100, 32)));
  BValue array = fb.Array({x, l, y}, x.GetType(), SourceInfo(), "array");
  BValue array_index0 = fb.ArrayIndex(array, {l0}, /*assumed_in_bounds=*/false,
                                      SourceInfo(), "array_index0");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(array));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_node_tree = node_info.GetInfo(array.node());
  int64_t array_node_count = node_info.GetSingleInfoForNode(array.node());
  EXPECT_EQ(array_node_count, 2);
  EXPECT_EQ(array_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(3, p->GetBitsType(32)), {1, 0, 1}));

  int64_t array_index0_count =
      node_info.GetSingleInfoForNode(array_index0.node());
  SharedLeafTypeTree<int64_t> array_index0_tree =
      node_info.GetInfo(array_index0.node());

  EXPECT_EQ(array_index0_count, 1);
  EXPECT_EQ(array_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));
}

TEST_F(DataFlowNodeInfoTest, TupleNested) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue z = fb.Param("z", p->GetBitsType(64));
  BValue tuple = fb.Tuple({x, y}, SourceInfo(), "tuple");
  BValue tuple2 = fb.Tuple({tuple, z}, SourceInfo(), "tuple");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(tuple2));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> tuple2_tree = node_info.GetInfo(tuple2.node());
  int64_t tuple2_node_count = node_info.GetSingleInfoForNode(tuple2.node());
  EXPECT_EQ(tuple2_node_count, 3);
  EXPECT_EQ(tuple2_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType(
                    {p->GetTupleType({p->GetBitsType(32), p->GetBitsType(64)}),
                     p->GetBitsType(64)}),
                {1, 1, 1}));
}

TEST_F(DataFlowNodeInfoTest, TupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(55, 32)));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue tuple = fb.Tuple({x, l, y}, SourceInfo(), "tuple");
  BValue tuple_index0 = fb.TupleIndex(tuple, 0, SourceInfo(), "tuple_index0");
  BValue tuple_index1 = fb.TupleIndex(tuple, 1, SourceInfo(), "tuple_index1");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(tuple));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> tuple_node_tree = node_info.GetInfo(tuple.node());
  int64_t tuple_node_count = node_info.GetSingleInfoForNode(tuple.node());
  EXPECT_EQ(tuple_node_count, 2);
  EXPECT_EQ(tuple_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32),
                                 p->GetBitsType(64)}),
                {1, 0, 1}));

  int64_t tuple_index0_count =
      node_info.GetSingleInfoForNode(tuple_index0.node());
  SharedLeafTypeTree<int64_t> tuple_index0_tree =
      node_info.GetInfo(tuple_index0.node());
  EXPECT_EQ(tuple_index0_count, 1);
  EXPECT_EQ(tuple_index0_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 1));

  int64_t tuple_index1_count =
      node_info.GetSingleInfoForNode(tuple_index1.node());
  SharedLeafTypeTree<int64_t> tuple_index1_tree =
      node_info.GetInfo(tuple_index1.node());
  EXPECT_EQ(tuple_index1_count, 0);
  EXPECT_EQ(tuple_index1_tree, LeafTypeTree<int64_t>::CreateSingleElementTree(
                                   p->GetBitsType(32), 0));
}

TEST_F(DataFlowNodeInfoTest, LiteralTupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue tuple = fb.Literal(xls::Value::Tuple(
      {xls::Value(xls::UBits(0, 32)), xls::Value(xls::UBits(0, 32))}));
  BValue tuple_index = fb.TupleIndex(tuple, 0, SourceInfo(), "tuple_index");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(tuple_index));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> tuple_index_node_tree =
      node_info.GetInfo(tuple_index.node());
  int64_t tuple_index_node_count =
      node_info.GetSingleInfoForNode(tuple_index.node());
  EXPECT_EQ(tuple_index_node_count, 0);
  EXPECT_EQ(
      tuple_index_node_tree,
      LeafTypeTree<int64_t>::CreateSingleElementTree(p->GetBitsType(32), 0));
}

TEST_F(DataFlowNodeInfoTest, TupleOfArrays) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue array_inner =
      fb.Array({x, y}, p->GetBitsType(32), SourceInfo(), "array");
  BValue tuple_outer = fb.Tuple({x, array_inner, y}, SourceInfo(), "tuple");
  BValue tuple_index =
      fb.TupleIndex(tuple_outer, 1, SourceInfo(), "tuple_index");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(tuple_outer));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> tuple_outer_node_tree =
      node_info.GetInfo(tuple_outer.node());
  int64_t tuple_node_count = node_info.GetSingleInfoForNode(tuple_outer.node());
  EXPECT_EQ(tuple_node_count, 4);
  EXPECT_EQ(tuple_outer_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32),
                                 p->GetArrayType(2, p->GetBitsType(32)),
                                 p->GetBitsType(32)}),
                {1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> tuple_index_node_tree =
      node_info.GetInfo(tuple_index.node());
  int64_t tuple_index_count =
      node_info.GetSingleInfoForNode(tuple_index.node());
  EXPECT_EQ(tuple_index_count, 2);
  EXPECT_EQ(tuple_index_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(2, p->GetBitsType(32)), {1, 1}));
}

TEST_F(DataFlowNodeInfoTest, ArrayOfTuples) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue l1 = fb.Literal(xls::Value(xls::UBits(1, 32)));
  BValue tuple_inner = fb.Tuple({x, y}, SourceInfo(), "tuple");
  BValue array_outer = fb.Array({tuple_inner, tuple_inner},
                                tuple_inner.GetType(), SourceInfo(), "array");
  BValue array_index =
      fb.ArrayIndex(array_outer, {l1}, /*assumed_in_bounds=*/false,
                    SourceInfo(), "tuple_index");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_index));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_outer_node_tree =
      node_info.GetInfo(array_outer.node());
  int64_t tuple_node_count = node_info.GetSingleInfoForNode(array_outer.node());
  EXPECT_EQ(tuple_node_count, 4);
  EXPECT_EQ(array_outer_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(2, p->GetTupleType({p->GetBitsType(32),
                                                    p->GetBitsType(32)})),
                {1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_index_node_tree =
      node_info.GetInfo(array_index.node());
  int64_t array_index_count =
      node_info.GetSingleInfoForNode(array_index.node());
  EXPECT_EQ(
      array_index_node_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}), {1, 1}));
  EXPECT_EQ(array_index_count, 2);
}

TEST_F(DataFlowNodeInfoTest, ArrayOfTuplesDynamicIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue i = fb.Param("i", p->GetBitsType(32));
  BValue tuple_inner = fb.Tuple({x, y}, SourceInfo(), "tuple");
  BValue array_outer = fb.Array({tuple_inner, tuple_inner},
                                tuple_inner.GetType(), SourceInfo(), "array");
  BValue array_index =
      fb.ArrayIndex(array_outer, {i}, /*assumed_in_bounds=*/false, SourceInfo(),
                    "tuple_index");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_index));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> array_node_tree =
      node_info.GetInfo(array_outer.node());
  int64_t array_node_count = node_info.GetSingleInfoForNode(array_outer.node());
  EXPECT_EQ(array_node_count, 4);
  EXPECT_EQ(array_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(2, p->GetTupleType({p->GetBitsType(32),
                                                    p->GetBitsType(32)})),
                {1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_index_node_tree =
      node_info.GetInfo(array_index.node());
  int64_t array_index_count =
      node_info.GetSingleInfoForNode(array_index.node());

  // Each element gets a value of 4
  EXPECT_EQ(array_index_count, 8);

  EXPECT_EQ(
      array_index_node_tree,
      LeafTypeTree<int64_t>::CreateFromVector(
          p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}), {4, 4}));
}

TEST_F(DataFlowNodeInfoTest, ArraySlice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(10, p->GetBitsType(32)));
  BValue s = fb.Param("s", p->GetBitsType(32));
  BValue array_slice = fb.ArraySlice(x, /*start=*/s, /*width=*/3);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f,
                           fb.BuildWithReturnValue(array_slice));

  TestParamCountInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<int64_t> x_node_tree = node_info.GetInfo(x.node());
  int64_t x_count = node_info.GetSingleInfoForNode(x.node());
  EXPECT_EQ(x_count, 10);
  EXPECT_EQ(x_node_tree, LeafTypeTree<int64_t>::CreateFromVector(
                             p->GetArrayType(10, p->GetBitsType(32)),
                             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  SharedLeafTypeTree<int64_t> array_slice_node_tree =
      node_info.GetInfo(array_slice.node());
  int64_t array_slice_count =
      node_info.GetSingleInfoForNode(array_slice.node());

  // Each element in the array ends up with a count of 11
  // This is because array_slice is not directly supported,
  // so the default action includes the slice start, and treats
  // the operation as dynamic.
  EXPECT_EQ(array_slice_count, 3 * 11);
  EXPECT_EQ(array_slice_node_tree,
            LeafTypeTree<int64_t>::CreateFromVector(
                p->GetArrayType(3, p->GetBitsType(32)), {11, 11, 11}));
}

TEST_F(DataFlowNodeInfoTest, AddNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(add));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  NodeSourceSet add_sources = node_info.GetSingleInfoForNode(add.node());
  SharedLeafTypeTree<NodeSourceSet> add_sources_tree =
      node_info.GetInfo(add.node());

  NodeSourceSet add_ref_sources = {NodeSource(add.node(), /*tree_index=*/{})};

  EXPECT_EQ(add_sources, add_ref_sources);
  EXPECT_EQ(add_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                p->GetBitsType(32), add_ref_sources));
}

TEST_F(DataFlowNodeInfoTest, AddInSelectNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue c = fb.Param("c", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue eq = fb.Eq(c, l, SourceInfo(), "eq");

  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "add");
  BValue sel = fb.Select(eq, x, add, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  NodeSourceSet sel_sources = node_info.GetSingleInfoForNode(sel.node());
  SharedLeafTypeTree<NodeSourceSet> sel_sources_tree =
      node_info.GetInfo(sel.node());

  NodeSourceSet sel_ref_sources = {NodeSource(x.node(), /*tree_index=*/{}),
                                   NodeSource(add.node(), /*tree_index=*/{})};

  EXPECT_EQ(sel_sources, sel_ref_sources);
  EXPECT_EQ(sel_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                p->GetBitsType(32), sel_ref_sources));
}

TEST_F(DataFlowNodeInfoTest, SelectNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue c = fb.Param("c", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue eq = fb.Eq(c, l, SourceInfo(), "eq");

  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel = fb.Select(eq, x, y, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  NodeSourceSet sel_sources = node_info.GetSingleInfoForNode(sel.node());
  SharedLeafTypeTree<NodeSourceSet> sel_sources_tree =
      node_info.GetInfo(sel.node());

  NodeSourceSet sel_ref_sources = {NodeSource(x.node(), /*tree_index=*/{}),
                                   NodeSource(y.node(), /*tree_index=*/{})};

  // Should not include the selector
  EXPECT_EQ(sel_sources, sel_ref_sources);
  EXPECT_EQ(sel_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                p->GetBitsType(32), sel_ref_sources));
}

TEST_F(DataFlowNodeInfoTest, SelectConstantSelectorNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l0 = fb.Literal(xls::Value(xls::UBits(0, 1)));
  BValue l1 = fb.Literal(xls::Value(xls::UBits(1, 1)));

  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel =
      fb.Select(l1, /*on_true=*/x, /*on_false=*/y, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    NodeSourceSet sel_sources = node_info.GetSingleInfoForNode(sel.node());
    SharedLeafTypeTree<NodeSourceSet> sel_sources_tree =
        node_info.GetInfo(sel.node());

    NodeSourceSet sel_on_1_ref_sources = {
        NodeSource(x.node(), /*tree_index=*/{})};

    EXPECT_EQ(sel_sources, sel_on_1_ref_sources);
    EXPECT_EQ(sel_sources_tree,
              LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                  p->GetBitsType(32), sel_on_1_ref_sources));
  }

  // Change the selector and check updated
  XLS_ASSERT_OK(l1.node()->ReplaceUsesWith(l0.node()));

  {
    NodeSourceSet sel_sources = node_info.GetSingleInfoForNode(sel.node());
    SharedLeafTypeTree<NodeSourceSet> sel_sources_tree =
        node_info.GetInfo(sel.node());

    NodeSourceSet sel_on_0_ref_sources = {
        NodeSource(y.node(), /*tree_index=*/{})};

    EXPECT_EQ(sel_sources, sel_on_0_ref_sources);
    EXPECT_EQ(sel_sources_tree,
              LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                  p->GetBitsType(32), sel_on_0_ref_sources));
  }
}

TEST_F(DataFlowNodeInfoTest, SelectConstExprNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l0 = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue l1 = fb.Literal(xls::Value(xls::UBits(7, 32)));
  BValue eq = fb.Eq(l0, l1, SourceInfo(), "eq");

  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel =
      fb.Select(eq, /*on_true=*/x, /*on_false=*/y, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  NodeSourceSet sel_sources = node_info.GetSingleInfoForNode(sel.node());
  SharedLeafTypeTree<NodeSourceSet> sel_sources_tree =
      node_info.GetInfo(sel.node());

  NodeSourceSet sel_ref_sources = {NodeSource(y.node(), /*tree_index=*/{})};

  // Should not include the selector
  EXPECT_EQ(sel_sources, sel_ref_sources);
  EXPECT_EQ(sel_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateSingleElementTree(
                p->GetBitsType(32), sel_ref_sources));
}

TEST_F(DataFlowNodeInfoTest, SwizzleNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue tup =
      fb.Param("x", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}));
  BValue ti0 = fb.TupleIndex(tup, 0, SourceInfo(), "ti0");
  BValue ti1 = fb.TupleIndex(tup, 1, SourceInfo(), "ti1");
  BValue swizzle = fb.Tuple({ti1, ti0}, SourceInfo(), "swizzle");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(swizzle));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  NodeSourceSet tup_sources = node_info.GetSingleInfoForNode(swizzle.node());
  SharedLeafTypeTree<NodeSourceSet> tup_sources_tree =
      node_info.GetInfo(swizzle.node());

  NodeSourceSet tup_ref_sources = {NodeSource(tup.node(), /*tree_index=*/{0}),
                                   NodeSource(tup.node(), /*tree_index=*/{1})};

  EXPECT_EQ(tup_sources, tup_ref_sources);
  EXPECT_EQ(tup_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}),
                {{NodeSource(tup.node(), /*tree_index=*/{1})},
                 {NodeSource(tup.node(), /*tree_index=*/{0})}}));
}

TEST_F(DataFlowNodeInfoTest, SelectTupleNoDefaultLeaf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x =
      fb.Param("x", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}));
  BValue y =
      fb.Param("y", p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}));
  BValue c = fb.Param("c", p->GetBitsType(32));
  BValue l = fb.Literal(xls::Value(xls::UBits(5, 32)));
  BValue eq = fb.Eq(c, l, SourceInfo(), "eq");
  BValue sel =
      fb.Select(eq, /*on_true=*/x, /*on_false=*/y, SourceInfo(), "sel");
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * f, fb.BuildWithReturnValue(sel));

  TestNodeSourceInfo node_info;
  node_info.set_query_engine(query_engine());
  XLS_ASSERT_OK(node_info.Attach(f));
  XLS_ASSERT_OK(query_engine()->Populate(f).status());
  NodeSourceSet tup_sources = node_info.GetSingleInfoForNode(sel.node());
  SharedLeafTypeTree<NodeSourceSet> tup_sources_tree =
      node_info.GetInfo(sel.node());

  NodeSourceSet tup_ref_sources = {NodeSource(x.node(), /*tree_index=*/{0}),
                                   NodeSource(x.node(), /*tree_index=*/{1}),
                                   NodeSource(y.node(), /*tree_index=*/{0}),
                                   NodeSource(y.node(), /*tree_index=*/{1})};

  // fprintf(stderr, "tup_sources_tree: %s\n",
  //         tup_sources_tree.ToString().c_str());

  EXPECT_EQ(tup_sources, tup_ref_sources);
  EXPECT_EQ(tup_sources_tree,
            LeafTypeTree<NodeSourceSet>::CreateFromVector(
                p->GetTupleType({p->GetBitsType(32), p->GetBitsType(32)}),
                {{NodeSource(x.node(), /*tree_index=*/{0}),
                  NodeSource(y.node(), /*tree_index=*/{0})},
                 {NodeSource(x.node(), /*tree_index=*/{1}),
                  NodeSource(y.node(), /*tree_index=*/{1})}}));
}

}  // namespace
}  // namespace xls
