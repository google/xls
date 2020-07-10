// Copyright 2020 Google LLC
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

#include "xls/ir/node_iterator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

TEST(NodeIteratorTest, ReordersViaDependencies) {
  Package p("p");
  Function f("f", &p);
  absl::optional<SourceLocation> loc = absl::nullopt;
  XLS_ASSERT_OK_AND_ASSIGN(Node * literal,
                           f.MakeNode<Literal>(loc, Value(UBits(3, 2))));
  XLS_ASSERT_OK_AND_ASSIGN(Node * neg,
                           f.MakeNode<UnOp>(loc, literal, OP_NEG));

  f.set_return_value(neg);

  // Literal should precede the negation in RPO although we added those nodes in
  // the opposite order.
  NodeIterator rni = TopoSort(&f);
  auto it = rni.begin();
  EXPECT_EQ(*it, literal);
  ++it;
  EXPECT_EQ(*it, neg);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, Diamond) {
  std::string program = R"(
  fn diamond(x: bits[32]) -> bits[32] {
    neg.1: bits[32] = neg(x)
    neg.2: bits[32] = neg(x)
    ret add.3: bits[32] = add(neg.1, neg.2)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  NodeIterator rni = TopoSort(f);
  auto it = rni.begin();
  EXPECT_EQ((*it)->As<Param>()->name(), "x");
  ++it;
  EXPECT_EQ((*it)->id(), 1);
  ++it;
  EXPECT_EQ((*it)->id(), 2);
  ++it;
  EXPECT_EQ((*it)->id(), 3);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//        A
//      /   \
//      \    B
//       \  /
//        \/
//         C
//
// Topological order: A B C
TEST(NodeIteratorTest, PostOrderNotPreOrder) {
  Package p("p");
  Function f("f", &p);
  absl::optional<SourceLocation> loc = absl::nullopt;
  XLS_ASSERT_OK_AND_ASSIGN(Node * a,
                           f.MakeNode<Literal>(loc, Value(UBits(0, 2))));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b, f.MakeNode<BinOp>(loc, a, a, OP_ADD));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c, f.MakeNode<BinOp>(loc, a, b, OP_ADD));

  f.set_return_value(c);

  NodeIterator rni = TopoSort(&f);
  auto it = rni.begin();
  EXPECT_EQ(*it, a);
  ++it;
  EXPECT_EQ(*it, b);
  ++it;
  EXPECT_EQ(*it, c);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//         A --
//        / \  \
//        | |   \
//        \ /   |
//         B    C
//          \  /
//            D
//
// Topo: D B C A =(reverse)=> A C B D
//                              2 1 3
TEST(NodeIteratorTest, TwoOfSameOperandLinks) {
  std::string program = R"(
  fn computation(a: bits[32]) -> bits[32] {
    b.1: bits[32] = add(a, a)
    c.2: bits[32] = neg(a)
    ret d.3: bits[32] = add(b.1, c.2)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  NodeIterator rni = TopoSort(f);
  auto it = rni.begin();
  EXPECT_EQ((*it)->As<Param>()->name(), "a");
  ++it;
  EXPECT_EQ((*it)->id(), 1);
  ++it;
  EXPECT_EQ((*it)->id(), 2);
  ++it;
  EXPECT_EQ((*it)->id(), 3);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, UselessParamsUnrelatedReturn) {
  std::string program = R"(
  fn computation(a: bits[32], b: bits[32]) -> bits[32] {
    ret r.1: bits[32] = literal(value=2)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  NodeIterator rni = TopoSort(f);
  auto it = rni.begin();
  EXPECT_EQ((*it)->As<Param>()->name(), "a");
  ++it;
  EXPECT_EQ((*it)->As<Param>()->name(), "b");
  ++it;
  EXPECT_EQ((*it)->id(), 1);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

// Constructs a test as follows:
//
//      A
//     / \
//    T   C
//     \ / \
//      B   E
//       \ /
//        D
TEST(NodeIteratorTest, ExtendedDiamond) {
  std::string program = R"(
  fn computation(a: bits[32]) -> bits[32] {
    t.1: bits[32] = neg(a)
    c.2: bits[32] = neg(a)
    b.3: bits[32] = add(t.1, c.2)
    e.4: bits[32] = neg(c.2)
    ret d.5: bits[32] = add(b.3, e.4)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  NodeIterator rni = TopoSort(f);
  auto it = rni.begin();
  EXPECT_EQ((*it)->As<Param>()->name(), "a");
  ++it;
  EXPECT_EQ((*it)->id(), 1);
  ++it;
  EXPECT_EQ((*it)->id(), 2);
  ++it;
  EXPECT_EQ((*it)->id(), 3);
  ++it;
  EXPECT_EQ((*it)->id(), 4);
  ++it;
  EXPECT_EQ((*it)->id(), 5);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

TEST(NodeIteratorTest, ExtendedDiamondReverse) {
  std::string program = R"(
  fn computation(a: bits[32]) -> bits[32] {
    t.1: bits[32] = neg(a)
    c.2: bits[32] = neg(a)
    b.3: bits[32] = add(t.1, c.2)
    e.4: bits[32] = neg(c.2)
    ret d.5: bits[32] = add(b.3, e.4)
  })";
  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  // ReverseTopoSort should produce the same order but in reverse.
  NodeIterator fwd_it = TopoSort(f);
  NodeIterator rev_it = ReverseTopoSort(f);
  std::vector<Node*> fwd(fwd_it.begin(), fwd_it.end());
  std::vector<Node*> rev(rev_it.begin(), rev_it.end());
  std::reverse(fwd.begin(), fwd.end());
  EXPECT_EQ(fwd, rev);
}

// Constructs a test as follows:
//
//      D
//      | \
//      C  \
//      |   \
//      B    T
//       \  /
//        \/
//         A
//
// Given that we know we visit operands in left-to-right order, this example
// points out the discrepancy between the RPO ordering and what our algorithm
// produces. The depth-first traversal RPO necessitates would have us visit the
// whole D,C,B chain before T.
//
// Post-Order:     D C B T A =(rev)=> A T B C D
//                                      1 2 3 4
// Our topo order: D T C B A =(rev)=> A B C T D
//                                      2 3 1 4
TEST(NodeIteratorTest, RpoVsTopo) {
  std::string program = R"(
  fn computation(a: bits[32]) -> bits[32] {
    t.1: bits[32] = neg(a)
    b.2: bits[32] = neg(a)
    c.3: bits[32] = neg(b.2)
    ret d.4: bits[32] = add(c.3, t.1)
  })";

  Package p("p");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(program, &p));

  NodeIterator rni = TopoSort(f);
  auto it = rni.begin();
  EXPECT_EQ((*it)->As<Param>()->name(), "a");
  ++it;
  EXPECT_EQ((*it)->id(), 2);
  ++it;
  EXPECT_EQ((*it)->id(), 3);
  ++it;
  EXPECT_EQ((*it)->id(), 1);
  ++it;
  EXPECT_EQ((*it)->id(), 4);
  ++it;
  EXPECT_EQ(rni.end(), it);
}

}  // namespace
}  // namespace xls
